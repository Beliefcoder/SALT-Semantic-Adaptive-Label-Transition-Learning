import argparse
import json
import os
import shutil
import sys

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from llama3_hidden_pipeline_utils import (
    EmbeddingBundle,
    build_label_prompt,
    build_label_prompt_styled,
    ensure_dir,
    infer_hidden_dataset_root,
    l2_normalize,
    load_code_metadata,
    masked_mean_pool,
    save_embedding_bundle,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract label embeddings with local Llama hidden states.")
    parser.add_argument("--source_dataset", type=str, default="car")
    parser.add_argument("--variant_dataset", type=str, default="car_llama3_hidden_v1")
    parser.add_argument("--model_path", type=str, default="Llama-3-8B-Instruct")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument(
        "--label_prompt_style",
        type=str,
        default="fault_code_part_name_zh",
        choices=[
            "fault_code_part_name_zh",
            "fault_code_part_name_en",
            "structured_automotive_zh",
            "structured_clinical_en",
            "diagnosis_name_only_en",
            "part_name_only_zh",
            "part_name_only_en",
            "code_only",
        ],
    )
    parser.add_argument(
        "--hidden_layer",
        type=int,
        default=-1,
        help="Which hidden_states index to pool. -1 means the final hidden layer.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "last", "first"],
        help="Token pooling strategy over the selected hidden layer.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def repair_sharded_model_dir(model_path: str) -> None:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return

    with open(index_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    weight_map = payload.get("weight_map", {})
    referenced_files = sorted(set(weight_map.values()))
    missing = [name for name in referenced_files if not os.path.exists(os.path.join(model_path, name))]
    if not missing:
        return

    temp_dir = os.path.join(model_path, "._____temp")
    if not os.path.isdir(temp_dir):
        missing_str = ", ".join(missing)
        raise FileNotFoundError(f"Missing model shard(s) in {model_path}: {missing_str}")

    copied = []
    unresolved = []
    for filename in missing:
        source = os.path.join(temp_dir, filename)
        target = os.path.join(model_path, filename)
        if os.path.exists(source):
            shutil.copy2(source, target)
            copied.append(filename)
        else:
            unresolved.append(filename)

    if copied:
        print(f"copied missing shard(s) from temp dir: {', '.join(copied)}")
    if unresolved:
        unresolved_str = ", ".join(unresolved)
        raise FileNotFoundError(f"Missing model shard(s) after repair attempt: {unresolved_str}")


def main():
    args = parse_args()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required for embedding extraction") from exc

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(repo_root, "data")
    source_dataset_root = os.path.join(data_root, args.source_dataset)
    target_dataset_root = infer_hidden_dataset_root(data_root, args.variant_dataset)
    artifacts_dir = os.path.join(target_dataset_root, "artifacts")
    ensure_dir(artifacts_dir)

    output_path = args.output_path or os.path.join(artifacts_dir, "label_embeddings_llama3_fp16.pt")
    codes, descriptions = load_code_metadata(source_dataset_root)

    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(repo_root, model_path)
    repair_sharded_model_dir(model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = resolve_dtype(args.torch_dtype)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    model.to(device)

    prompts = [
        build_label_prompt_styled(code, desc, args.label_prompt_style)
        for code, desc in zip(codes, descriptions)
    ]
    embedding_batches = []
    with torch.no_grad():
        for start in range(0, len(prompts), args.batch_size):
            end = min(start + args.batch_size, len(prompts))
            batch_prompts = prompts[start:end]
            encoded = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded, output_hidden_states=True, return_dict=True)
            hidden = outputs.hidden_states[args.hidden_layer]
            if args.pooling == "mean":
                pooled = masked_mean_pool(hidden, encoded["attention_mask"])
            elif args.pooling == "first":
                pooled = hidden[:, 0, :]
            else:
                lengths = encoded["attention_mask"].sum(dim=1).clamp_min(1) - 1
                pooled = hidden[torch.arange(hidden.size(0), device=device), lengths]
            pooled = l2_normalize(pooled.float()).to(torch.float16)
            embedding_batches.append(pooled.cpu())
            print(f"embedded labels: {end}/{len(prompts)}")

    embeddings = torch.cat(embedding_batches, dim=0)
    bundle = EmbeddingBundle(
        codes=codes,
        descriptions=descriptions,
        embeddings=embeddings,
        model_path=model_path,
        prompt_style=f"{args.label_prompt_style}|layer={args.hidden_layer}|pooling={args.pooling}",
    )
    save_embedding_bundle(output_path, bundle)
    print(f"saved embeddings to {output_path}")


if __name__ == "__main__":
    main()
