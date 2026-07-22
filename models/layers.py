import math

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from scipy import sparse

from models.utils import SingleHeadAttentionLayer


class EmbeddingLayer(nn.Module):
    def __init__(self, code_num, code_size, graph_size):
        super().__init__()
        self.code_num = code_num
        self.c_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, code_size)))
        self.n_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, code_size)))
        self.u_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, graph_size)))

    def forward(self):
        return self.c_embeddings, self.n_embeddings, self.u_embeddings


def _coerce_transition_tensor(value, device, dtype=torch.float32):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    array = value.toarray() if sparse.issparse(value) else np.array(value)
    return torch.as_tensor(array, device=device, dtype=dtype)


class EndToEndSemanticTransitionGenerator(nn.Module):
    def __init__(
        self,
        label_embeddings,
        bottleneck_dim=128,
        hidden_dim=256,
        dropout=0.1,
        chunk_size=64,
        candidate_mask=None,
    ):
        super().__init__()
        label_tensor = torch.as_tensor(label_embeddings, dtype=torch.float32)
        self.register_buffer("label_embeddings", label_tensor)
        if candidate_mask is not None:
            mask_tensor = torch.as_tensor(candidate_mask, dtype=torch.float32)
            self.register_buffer("candidate_mask", mask_tensor)
        else:
            self.candidate_mask = None
        self.embedding_dim = label_tensor.shape[1]
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.chunk_size = chunk_size
        self.input_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(bottleneck_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self):
        projected = self.input_proj(self.label_embeddings)
        num_labels = projected.size(0)
        rows = []
        for start in range(0, num_labels, self.chunk_size):
            end = min(start + self.chunk_size, num_labels)
            lhs = projected[start:end].unsqueeze(1)
            rhs = projected.unsqueeze(0)
            diff = torch.abs(lhs - rhs)
            prod = lhs * rhs
            features = torch.cat(
                [lhs.expand(-1, num_labels, -1), rhs.expand(end - start, -1, -1), diff, prod],
                dim=-1,
            )
            logits = self.head(features).squeeze(-1)
            scores = torch.sigmoid(logits)
            rows.append(scores)
        transition = torch.cat(rows, dim=0)
        if self.candidate_mask is not None:
            transition = transition * self.candidate_mask.to(device=transition.device, dtype=transition.dtype)
        transition.fill_diagonal_(0.0)
        return transition


class GraphLayer(nn.Module):
    def __init__(self, adj, code_size, graph_size):
        super().__init__()
        self.adj = adj
        self.dense = nn.Linear(code_size, graph_size) # å¨è¿æ¥å±
        self.activation = nn.LeakyReLU() # æ¿æ´»å½æ°

    def forward(self, code_x, neighbor, c_embeddings, n_embeddings):
        center_codes = torch.unsqueeze(code_x, dim=-1) #
        neighbor_codes = torch.unsqueeze(neighbor, dim=-1)

        center_embeddings = center_codes * c_embeddings # 1 èµ·ä½ç¨
        neighbor_embeddings = neighbor_codes * n_embeddings
        cc_embeddings = center_codes * torch.matmul(self.adj, center_embeddings)
        cn_embeddings = center_codes * torch.matmul(self.adj, neighbor_embeddings)
        nn_embeddings = neighbor_codes * torch.matmul(self.adj, neighbor_embeddings)
        nc_embeddings = neighbor_codes * torch.matmul(self.adj, center_embeddings)

        co_embeddings = self.activation(self.dense(center_embeddings + cc_embeddings + cn_embeddings)) # Zd
        no_embeddings = self.activation(self.dense(neighbor_embeddings + nn_embeddings + nc_embeddings)) # Zn
        return co_embeddings, no_embeddings


class TransitionLayer(nn.Module):
    def __init__(self, code_num, graph_size, hidden_size, t_attention_size, t_output_size):
        super().__init__()
        self.gru = nn.GRUCell(input_size=graph_size, hidden_size=hidden_size)
        self.single_head_attention = SingleHeadAttentionLayer(graph_size, graph_size, t_output_size, t_attention_size)
        self.activation = nn.Tanh()

        self.code_num = code_num
        self.hidden_size = hidden_size

    def forward(self, t, co_embeddings, divided, no_embeddings, unrelated_embeddings, hidden_state=None):
        m1, m2, m3 = divided[:, 0], divided[:, 1], divided[:, 2] # åå«ååºä¸ç±»ç¾ççdivided
        m1_index = torch.where(m1 > 0)[0] # é¡½åºç¾ç
        m2_index = torch.where(m2 > 0)[0] # æ°å´ç¾ç
        m3_index = torch.where(m3 > 0)[0] # æ°å´æ å³ç¾ç
        h_new = torch.zeros((self.code_num, self.hidden_size), dtype=co_embeddings.dtype).to(co_embeddings.device)
        output_m1 = 0
        output_m23 = 0
        if len(m1_index) > 0:
            m1_embedding = co_embeddings[m1_index] #ç­éåºé¡½åºç¾çå¯¹åºçå¨å±è¯æ­ä¸ä¸æ
            h = hidden_state[m1_index] if hidden_state is not None else None
            h_m1 = self.gru(m1_embedding, h)
            h_new[m1_index] = h_m1
            output_m1, _ = torch.max(h_m1, dim=-2) # æå¤§æ± å
        if t > 0 and len(m2_index) + len(m3_index) > 0:
            q = torch.vstack([no_embeddings[m2_index], unrelated_embeddings[m3_index]]) # æ¨ªçæ¼æ¥
            v = torch.vstack([co_embeddings[m2_index], co_embeddings[m3_index]])
            h_m23 = self.activation(self.single_head_attention(q, q, v))
            h_new[m2_index] = h_m23[:len(m2_index)]
            h_new[m3_index] = h_m23[len(m2_index):]
            output_m23, _ = torch.max(h_m23, dim=-2)
        if len(m1_index) == 0:
            output = output_m23
        elif len(m2_index) + len(m3_index) == 0:
            output = output_m1
        else:
            output, _ = torch.max(torch.vstack([output_m1, output_m23]), dim=-2)
        return output, h_new


class SimpleTransitionLayer(nn.Module):
    def __init__(self, code_num, hidden_size, init_scale=1.0, dropout=0.1):
        super().__init__()
        self.code_num = code_num
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        self.linear = nn.Linear(hidden_size, code_num)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.output_linear = nn.Linear(code_num, hidden_size)
        nn.init.xavier_uniform_(self.output_linear.weight)
        nn.init.zeros_(self.output_linear.bias)
        self.scale = nn.Parameter(torch.tensor(init_scale))
        self.dropout = nn.Dropout(dropout)
        self._transition_T = None
        self._T_tensor = None

    def set_transition_matrix(self, T):
        if T is not None:
            self._transition_T = T
            if sparse.issparse(T):
                self._T_tensor = torch.from_numpy(T.toarray()).float()
            else:
                self._T_tensor = torch.from_numpy(T).float()

    def forward(self, h, transition_T=None):
        T = transition_T if transition_T is not None else self._transition_T
        if T is None:
            return h
        
        if self._T_tensor is None:
            if sparse.issparse(T):
                self._T_tensor = torch.from_numpy(T.toarray()).float()
            else:
                self._T_tensor = torch.from_numpy(T).float()
        
        T_device = self._T_tensor.to(h.device)
        
        is_2d = h.dim() == 2
        if is_2d:
            batch_seq, hidden = h.shape
            h = h.unsqueeze(1)
        
        code_rep = self.linear(h)
        transition_out = torch.matmul(code_rep, T_device)
        transition_out = self.output_linear(transition_out)
        transition_out = self.dropout(transition_out)
        output = h + self.scale * transition_out
        
        if is_2d:
            output = output.squeeze(1)
        
        return output


class SoftPositionFusionLayer(nn.Module):
    def __init__(self, code_num, hidden_size, max_seq_len=70, 
                 base_alpha=0.5, position_factor=0.2, 
                 confidence_adjust_range=0.2, use_confidence=True, dropout=0.1):
        super().__init__()
        
        self.code_num = code_num
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.use_confidence = use_confidence
        
        self.base_alpha = nn.Parameter(torch.tensor(base_alpha))
        self.position_factor = nn.Parameter(torch.tensor(position_factor), requires_grad=False)
        self.confidence_scale = nn.Parameter(torch.tensor(1.0))
        self.confidence_adjust_range = confidence_adjust_range
        
        self.linear = nn.Linear(hidden_size, code_num)
        self.output_linear = nn.Linear(code_num, hidden_size)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.xavier_uniform_(self.output_linear.weight)
        nn.init.zeros_(self.output_linear.bias)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(dropout)
        
        self._T_data_tensor = None
        self._T_llm_tensor = None
    
    def set_transition_matrices(self, T_data, T_llm=None):
        if T_data is not None:
            if sparse.issparse(T_data):
                self._T_data_tensor = torch.from_numpy(T_data.toarray()).float()
            else:
                self._T_data_tensor = torch.from_numpy(T_data).float()
        
        if T_llm is not None:
            if sparse.issparse(T_llm):
                self._T_llm_tensor = torch.from_numpy(T_llm.toarray()).float()
            else:
                self._T_llm_tensor = torch.from_numpy(T_llm).float()
    
    def _compute_confidence_scalar(self, h_t):
        h_mean = torch.mean(torch.abs(h_t))
        return h_mean
    
    def forward(self, h, transition_T=None, transition_T_llm=None):
        T_data = transition_T if transition_T is not None else self._T_data_tensor
        T_llm = transition_T_llm if transition_T_llm is not None else self._T_llm_tensor
        
        if T_data is None:
            return h
        
        if self._T_data_tensor is None:
            if sparse.issparse(T_data):
                self._T_data_tensor = torch.from_numpy(T_data.toarray()).float()
            else:
                self._T_data_tensor = torch.from_numpy(T_data).float()
            T_data = self._T_data_tensor
        
        if T_llm is not None and self._T_llm_tensor is None:
            if sparse.issparse(T_llm):
                self._T_llm_tensor = torch.from_numpy(T_llm.toarray()).float()
            else:
                self._T_llm_tensor = torch.from_numpy(T_llm).float()
            T_llm = self._T_llm_tensor
        
        is_2d = h.dim() == 2
        if is_2d:
            batch_seq, hidden = h.shape
            h = h.unsqueeze(1)
        
        batch_size, seq_len, hidden_size = h.shape
        
        T_data_device = T_data.to(h.device)
        T_llm_device = T_llm.to(h.device) if T_llm is not None else None
        
        position_factor = torch.sigmoid(self.position_factor) * 0.3
        base_alpha = torch.sigmoid(self.base_alpha)
        
        outputs = []
        
        for t in range(seq_len):
            h_t = h[:, t, :]
            
            pos_factor = t / max(1, seq_len - 1)
            base_alpha_t = base_alpha + pos_factor * position_factor
            
            if self.use_confidence:
                batch_confidence = torch.mean(torch.abs(h_t))
                confidence_adj = torch.tanh(batch_confidence) * self.confidence_adjust_range * self.confidence_scale
                alpha_t = torch.sigmoid(base_alpha_t + confidence_adj)
            else:
                alpha_t = torch.sigmoid(base_alpha_t)
            
            alpha_expanded = alpha_t.unsqueeze(-1).unsqueeze(-1)
            
            if T_llm_device is not None:
                T_fused_base = alpha_expanded * T_data_device + (1 - alpha_expanded) * T_llm_device
            else:
                T_fused_base = T_data_device
            
            h_t_expanded = h_t.unsqueeze(1)
            code_rep = self.linear(h_t_expanded)
            transition_out = torch.matmul(code_rep, T_fused_base)
            transition_out = self.output_linear(transition_out)
            transition_out = self.dropout(transition_out)
            h_out = h_t_expanded + self.scale * transition_out
            
            outputs.append(h_out)
        
        output = torch.cat(outputs, dim=1)
        
        if is_2d:
            output = output.squeeze(1)
        
        return output


class FusionTransitionLayer(nn.Module):
    def __init__(self, code_num, hidden_size, init_scale=1.0, fusion_mode='learnable', init_alpha=0.5, dropout=0.1):
        super().__init__()
        self.code_num = code_num
        self.hidden_size = hidden_size
        self.fusion_mode = fusion_mode
        
        self.linear = nn.Linear(hidden_size, code_num)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.output_linear = nn.Linear(code_num, hidden_size)
        nn.init.xavier_uniform_(self.output_linear.weight)
        nn.init.zeros_(self.output_linear.bias)
        self.scale = nn.Parameter(torch.tensor(init_scale))
        self.dropout = nn.Dropout(dropout)
        
        if fusion_mode == 'learnable':
            self.alpha = nn.Parameter(torch.tensor(init_alpha))
        else:
            self.register_buffer('alpha', torch.tensor(init_alpha))
        
        self._T_data = None
        self._T_llm = None
        self._T_data_tensor = None
        self._T_llm_tensor = None

    def set_transition_matrices(self, T_data, T_llm=None):
        if T_data is not None:
            self._T_data = T_data
            if sparse.issparse(T_data):
                T_data_arr = T_data.toarray()
            else:
                T_data_arr = T_data
            self._T_data_tensor = torch.from_numpy(T_data_arr).float()
        
        if T_llm is not None:
            self._T_llm = T_llm
            if sparse.issparse(T_llm):
                T_llm_arr = T_llm.toarray()
            else:
                T_llm_arr = T_llm
            self._T_llm_tensor = torch.from_numpy(T_llm_arr).float()

    def forward(self, h, transition_T=None, transition_T_llm=None):
        T_data = transition_T if transition_T is not None else self._T_data
        T_llm = transition_T_llm if transition_T_llm is not None else self._T_llm
        
        if T_data is None:
            return h
        
        if self._T_data_tensor is None:
            T_data_arr = T_data.toarray() if sparse.issparse(T_data) else np.array(T_data)
            T_data_tensor = torch.from_numpy(T_data_arr).float().to(h.device)
        else:
            T_data_tensor = self._T_data_tensor.to(h.device)
        
        if T_llm is not None:
            if self._T_llm_tensor is None:
                T_llm_arr = T_llm.toarray() if sparse.issparse(T_llm) else np.array(T_llm)
                T_llm_tensor = torch.from_numpy(T_llm_arr).float().to(h.device)
            else:
                T_llm_tensor = self._T_llm_tensor.to(h.device)
            
            alpha = torch.sigmoid(self.alpha) if isinstance(self.alpha, nn.Parameter) else self.alpha
            T_fused = alpha * T_data_tensor + (1 - alpha) * T_llm_tensor
        else:
            T_fused = T_data_tensor
        
        is_2d = h.dim() == 2
        if is_2d:
            batch_seq, hidden = h.shape
            h = h.unsqueeze(1)
        
        code_rep = self.linear(h)
        transition_out = torch.matmul(code_rep, T_fused)
        transition_out = self.output_linear(transition_out)
        transition_out = self.dropout(transition_out)
        output = h + self.scale * transition_out
        
        if is_2d:
            output = output.squeeze(1)
        
        return output


class DynamicResidualFusionTransitionLayer(nn.Module):
    def __init__(self, code_num, hidden_size, context_size=None, init_alpha=0.5, init_beta=0.5, dropout=0.1):
        super().__init__()
        self.code_num = code_num
        self.hidden_size = hidden_size
        self.context_size = hidden_size if context_size is None else context_size

        self.linear_data = nn.Linear(hidden_size, code_num)
        self.linear_llm = nn.Linear(hidden_size, code_num)
        self.output_linear_data = nn.Linear(code_num, hidden_size)
        self.output_linear_llm = nn.Linear(code_num, hidden_size)

        for layer in [self.linear_data, self.linear_llm, self.output_linear_data, self.output_linear_llm]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.gate_mlp = nn.Sequential(
            nn.Linear(self.context_size, max(8, hidden_size)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, hidden_size), 1)
        )
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))
        self.gate_bias = nn.Parameter(torch.tensor(float(init_alpha)))
        self.dropout = nn.Dropout(dropout)

        self._T_data_tensor = None
        self._T_llm_tensor = None

    def set_transition_matrices(self, T_data, T_llm):
        if T_data is not None:
            self._T_data_tensor = torch.from_numpy(T_data.toarray() if sparse.issparse(T_data) else np.array(T_data)).float()
        if T_llm is not None:
            self._T_llm_tensor = torch.from_numpy(T_llm.toarray() if sparse.issparse(T_llm) else np.array(T_llm)).float()

    def _masked_mean(self, h, mask=None):
        if mask is None:
            return h.mean(dim=1)
        mask_f = mask.unsqueeze(-1).to(h.dtype)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        return (h * mask_f).sum(dim=1) / denom

    def forward(self, h, mask=None, transition_T=None, transition_T_llm=None, return_extras=False):
        T_data_tensor = self._T_data_tensor if transition_T is None else _coerce_transition_tensor(
            transition_T, device=h.device
        )
        T_llm_tensor = self._T_llm_tensor if transition_T_llm is None else _coerce_transition_tensor(
            transition_T_llm, device=h.device
        )

        if T_data_tensor is None or T_llm_tensor is None:
            return (h, None) if return_extras else h

        T_data_tensor = T_data_tensor.to(h.device)
        T_llm_tensor = T_llm_tensor.to(h.device)

        code_rep_data = self.linear_data(h)
        code_rep_llm = self.linear_llm(h)
        z_data = self.output_linear_data(torch.matmul(code_rep_data, T_data_tensor))
        z_llm = self.output_linear_llm(torch.matmul(code_rep_llm, T_llm_tensor))

        context = self._masked_mean(h, mask)
        gate = torch.sigmoid(self.gate_mlp(context) + self.gate_bias).view(-1, 1, 1)
        beta = torch.sigmoid(self.beta)

        fused = gate * z_data + (1.0 - gate) * z_llm
        fused = self.dropout(fused)
        output = h + beta * fused

        if mask is not None:
            output = output * mask.unsqueeze(-1).to(output.dtype)

        extras = {
            'fusion_gate_mean': gate.mean().item(),
            'fusion_gate_std': gate.std().item(),
            'residual_beta': beta.item(),
        }
        return (output, extras) if return_extras else output


class DynamicVectorResidualFusionTransitionLayer(nn.Module):
    def __init__(self, code_num, hidden_size, context_size=None, init_alpha=0.5, init_beta=0.5, dropout=0.1):
        super().__init__()
        self.code_num = code_num
        self.hidden_size = hidden_size
        self.context_size = hidden_size if context_size is None else context_size

        self.linear_data = nn.Linear(hidden_size, code_num)
        self.linear_llm = nn.Linear(hidden_size, code_num)
        self.output_linear_data = nn.Linear(code_num, hidden_size)
        self.output_linear_llm = nn.Linear(code_num, hidden_size)

        for layer in [self.linear_data, self.linear_llm, self.output_linear_data, self.output_linear_llm]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.gate_mlp = nn.Sequential(
            nn.Linear(self.context_size, max(16, hidden_size * 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, hidden_size * 2), hidden_size)
        )
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))
        self.gate_bias = nn.Parameter(torch.full((hidden_size,), float(init_alpha)))
        self.dropout = nn.Dropout(dropout)

        self._T_data_tensor = None
        self._T_llm_tensor = None

    def set_transition_matrices(self, T_data, T_llm):
        if T_data is not None:
            self._T_data_tensor = torch.from_numpy(T_data.toarray() if sparse.issparse(T_data) else np.array(T_data)).float()
        if T_llm is not None:
            self._T_llm_tensor = torch.from_numpy(T_llm.toarray() if sparse.issparse(T_llm) else np.array(T_llm)).float()

    def _select_last_valid(self, h, mask=None):
        if mask is None:
            return h[:, -1, :]
        lengths = mask.long().sum(dim=1).clamp(min=1)
        batch_idx = torch.arange(h.size(0), device=h.device)
        last_idx = lengths - 1
        return h[batch_idx, last_idx]

    def forward(self, h, mask=None, transition_T=None, transition_T_llm=None, return_extras=False):
        T_data_tensor = self._T_data_tensor if transition_T is None else _coerce_transition_tensor(
            transition_T, device=h.device
        )
        T_llm_tensor = self._T_llm_tensor if transition_T_llm is None else _coerce_transition_tensor(
            transition_T_llm, device=h.device
        )

        if T_data_tensor is None or T_llm_tensor is None:
            return (h, None) if return_extras else h

        T_data_tensor = T_data_tensor.to(h.device)
        T_llm_tensor = T_llm_tensor.to(h.device)

        code_rep_data = self.linear_data(h)
        code_rep_llm = self.linear_llm(h)
        z_data = self.output_linear_data(torch.matmul(code_rep_data, T_data_tensor))
        z_llm = self.output_linear_llm(torch.matmul(code_rep_llm, T_llm_tensor))

        last_context = self._select_last_valid(h, mask)
        gate = torch.sigmoid(self.gate_mlp(last_context) + self.gate_bias).unsqueeze(1)
        beta = torch.sigmoid(self.beta)

        fused = gate * z_data + (1.0 - gate) * z_llm
        fused = self.dropout(fused)
        output = h + beta * fused

        if mask is not None:
            output = output * mask.unsqueeze(-1).to(output.dtype)

        extras = {
            'fusion_gate_mean': gate.mean().item(),
            'fusion_gate_std': gate.std().item(),
            'residual_beta': beta.item(),
        }
        return (output, extras) if return_extras else output


class SampleAdaptiveFusionTransitionLayer(nn.Module):
    def __init__(
        self,
        code_num,
        hidden_size,
        init_alpha=0.5,
        init_beta=0.5,
        dropout=0.1,
        gate_mode='sample_adaptive',
        shared_projection=False,
    ):
        super().__init__()
        self.code_num = code_num
        self.hidden_size = hidden_size
        self.gate_mode = gate_mode
        self.shared_projection = shared_projection

        self.linear_data = nn.Linear(hidden_size, code_num)
        self.linear_llm = nn.Linear(hidden_size, code_num)
        self.output_linear_data = nn.Linear(code_num, hidden_size)
        self.output_linear_llm = nn.Linear(code_num, hidden_size)
        for layer in [self.linear_data, self.linear_llm, self.output_linear_data, self.output_linear_llm]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.alpha_mlp = nn.Sequential(
            nn.Linear(hidden_size, max(8, hidden_size)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, hidden_size), 1)
        )
        self.alpha_bias = nn.Parameter(torch.tensor(float(init_alpha)))
        init_alpha_clamped = min(max(float(init_alpha), 1e-4), 1.0 - 1e-4)
        self.global_alpha_logit = nn.Parameter(torch.tensor(math.log(init_alpha_clamped / (1.0 - init_alpha_clamped))))
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))
        self.dropout = nn.Dropout(dropout)

        self._T_data_tensor = None
        self._T_llm_tensor = None

    def set_transition_matrices(self, T_data, T_llm):
        if T_data is not None:
            self._T_data_tensor = torch.from_numpy(T_data.toarray() if sparse.issparse(T_data) else np.array(T_data)).float()
        if T_llm is not None:
            self._T_llm_tensor = torch.from_numpy(T_llm.toarray() if sparse.issparse(T_llm) else np.array(T_llm)).float()

    def _select_last_valid(self, h, mask=None):
        if mask is None:
            return h[:, -1, :]
        lengths = mask.long().sum(dim=1).clamp(min=1)
        batch_idx = torch.arange(h.size(0), device=h.device)
        return h[batch_idx, lengths - 1]

    def forward(self, h, mask=None, transition_T=None, transition_T_llm=None, return_extras=False):
        T_data_tensor = self._T_data_tensor if transition_T is None else _coerce_transition_tensor(
            transition_T, device=h.device
        )
        T_llm_tensor = self._T_llm_tensor if transition_T_llm is None else _coerce_transition_tensor(
            transition_T_llm, device=h.device
        )

        if T_data_tensor is None or T_llm_tensor is None:
            return (h, None) if return_extras else h

        T_data_tensor = T_data_tensor.to(h.device)
        T_llm_tensor = T_llm_tensor.to(h.device)

        z_data = self.output_linear_data(torch.matmul(self.linear_data(h), T_data_tensor))
        if self.shared_projection:
            z_llm = self.output_linear_data(torch.matmul(self.linear_data(h), T_llm_tensor))
        else:
            z_llm = self.output_linear_llm(torch.matmul(self.linear_llm(h), T_llm_tensor))

        if self.gate_mode == 'fixed':
            alpha = h.new_full((h.size(0), 1, 1), 0.5)
        elif self.gate_mode == 'global':
            alpha = torch.sigmoid(self.global_alpha_logit).view(1, 1, 1).expand(h.size(0), 1, 1)
        else:
            last_context = self._select_last_valid(h, mask)
            alpha = torch.sigmoid(self.alpha_mlp(last_context) + self.alpha_bias).view(-1, 1, 1)
        beta = torch.sigmoid(self.beta)

        fused = alpha * z_data + (1.0 - alpha) * z_llm
        fused = self.dropout(fused)
        output = h + beta * fused
        if mask is not None:
            output = output * mask.unsqueeze(-1).to(output.dtype)

        extras = {
            'fusion_alpha_mean': alpha.mean().item(),
            'fusion_alpha_std': alpha.std().item(),
            'fusion_alpha_values': alpha.view(-1).detach().cpu().numpy(),
            'residual_beta': beta.item(),
            'gate_mode': self.gate_mode,
            'shared_projection': self.shared_projection,
        }
        if transition_T_llm is not None and isinstance(transition_T_llm, torch.Tensor):
            extras['end_to_end_transition_mean'] = transition_T_llm.mean().item()
            extras['end_to_end_transition_density'] = (transition_T_llm > 0).float().mean().item()
        return (output, extras) if return_extras else output


class ConfidenceAwareFusionTransitionLayer(nn.Module):
    def __init__(self, code_num, hidden_size, init_alpha=0.5, init_beta=0.5, dropout=0.1):
        super().__init__()
        self.code_num = code_num
        self.hidden_size = hidden_size

        self.linear_data = nn.Linear(hidden_size, code_num)
        self.linear_llm = nn.Linear(hidden_size, code_num)
        self.output_linear_data = nn.Linear(code_num, hidden_size)
        self.output_linear_llm = nn.Linear(code_num, hidden_size)
        for layer in [self.linear_data, self.linear_llm, self.output_linear_data, self.output_linear_llm]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.alpha_mlp = nn.Sequential(
            nn.Linear(hidden_size + 2, max(8, hidden_size)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, hidden_size), 1)
        )
        self.alpha_bias = nn.Parameter(torch.tensor(float(init_alpha)))
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))
        self.dropout = nn.Dropout(dropout)

        self._T_data_tensor = None
        self._T_llm_tensor = None
        self._T_data_conf = None
        self._T_llm_conf = None

    def _compute_row_confidence(self, T):
        if T is None:
            return None
        T_arr = T.toarray() if sparse.issparse(T) else np.array(T)
        T_arr = T_arr.astype(np.float32)
        row_sum = T_arr.sum(axis=1, keepdims=True)
        normalized = np.divide(T_arr, row_sum + 1e-12, where=row_sum > 0)
        entropy = -np.sum(normalized * np.log(normalized + 1e-12), axis=1)
        max_entropy = np.log(T_arr.shape[1] + 1e-12)
        confidence = 1.0 - (entropy / max(max_entropy, 1e-12))
        confidence[row_sum.squeeze(-1) <= 0] = 0.0
        return torch.from_numpy(confidence.astype(np.float32))

    def set_transition_matrices(self, T_data, T_llm):
        if T_data is not None:
            self._T_data_tensor = torch.from_numpy(T_data.toarray() if sparse.issparse(T_data) else np.array(T_data)).float()
            self._T_data_conf = self._compute_row_confidence(T_data)
        if T_llm is not None:
            self._T_llm_tensor = torch.from_numpy(T_llm.toarray() if sparse.issparse(T_llm) else np.array(T_llm)).float()
            self._T_llm_conf = self._compute_row_confidence(T_llm)

    def _select_last_valid(self, h, mask=None):
        if mask is None:
            return h[:, -1, :]
        lengths = mask.long().sum(dim=1).clamp(min=1)
        batch_idx = torch.arange(h.size(0), device=h.device)
        return h[batch_idx, lengths - 1]

    def _sample_confidence(self, last_visit_codes, conf_tensor):
        if conf_tensor is None:
            return torch.zeros(last_visit_codes.size(0), 1, device=last_visit_codes.device)
        conf_tensor = conf_tensor.to(last_visit_codes.device)
        weights = last_visit_codes.float()
        denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        conf = torch.matmul(weights, conf_tensor.unsqueeze(-1)) / denom
        return conf

    def forward(self, h, mask=None, last_visit_codes=None, transition_T=None, transition_T_llm=None, return_extras=False):
        T_data_tensor = self._T_data_tensor if transition_T is None else _coerce_transition_tensor(
            transition_T, device=h.device
        )
        T_llm_tensor = self._T_llm_tensor if transition_T_llm is None else _coerce_transition_tensor(
            transition_T_llm, device=h.device
        )

        if T_data_tensor is None or T_llm_tensor is None:
            return (h, None) if return_extras else h

        T_data_tensor = T_data_tensor.to(h.device)
        T_llm_tensor = T_llm_tensor.to(h.device)

        z_data = self.output_linear_data(torch.matmul(self.linear_data(h), T_data_tensor))
        z_llm = self.output_linear_llm(torch.matmul(self.linear_llm(h), T_llm_tensor))

        last_context = self._select_last_valid(h, mask)
        if last_visit_codes is None:
            conf_data = torch.zeros(h.size(0), 1, device=h.device)
            conf_llm = torch.zeros(h.size(0), 1, device=h.device)
        else:
            conf_data = self._sample_confidence(last_visit_codes, self._T_data_conf)
            conf_llm = self._sample_confidence(last_visit_codes, self._T_llm_conf)
        alpha_input = torch.cat([last_context, conf_data, conf_llm], dim=-1)
        alpha = torch.sigmoid(self.alpha_mlp(alpha_input) + self.alpha_bias).view(-1, 1, 1)
        beta = torch.sigmoid(self.beta)

        fused = alpha * z_data + (1.0 - alpha) * z_llm
        fused = self.dropout(fused)
        output = h + beta * fused
        if mask is not None:
            output = output * mask.unsqueeze(-1).to(output.dtype)

        extras = {
            'fusion_alpha_mean': alpha.mean().item(),
            'fusion_alpha_std': alpha.std().item(),
            'residual_beta': beta.item(),
            'conf_data_mean': conf_data.mean().item() if last_visit_codes is not None else 0.0,
            'conf_llm_mean': conf_llm.mean().item() if last_visit_codes is not None else 0.0,
        }
        return (output, extras) if return_extras else output


class LogitPriorLayer(nn.Module):
    def __init__(self, init_lambda=0.5, learnable=True):
        super().__init__()
        if learnable:
            self.prior_lambda = nn.Parameter(torch.tensor(float(init_lambda)))
        else:
            self.register_buffer('prior_lambda', torch.tensor(float(init_lambda)))

        self._T_tensor = None

    def set_transition_matrix(self, T):
        if T is None:
            self._T_tensor = None
            return
        if sparse.issparse(T):
            self._T_tensor = torch.from_numpy(T.toarray()).float()
        else:
            self._T_tensor = torch.from_numpy(np.array(T)).float()

    def forward(self, logits, last_visit_codes, transition_T=None):
        T = transition_T if transition_T is not None else self._T_tensor
        if T is None:
            return logits

        if self._T_tensor is None:
            self.set_transition_matrix(T)

        T_device = self._T_tensor.to(logits.device)
        prior_logits = torch.matmul(last_visit_codes.float(), T_device)
        prior_weight = torch.sigmoid(self.prior_lambda)
        return logits + prior_weight * prior_logits


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalConvBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.norm1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.norm2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.final_norm = nn.BatchNorm1d(n_outputs)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        out = out + residual
        out = self.relu(out)
        
        return out


class LightweightTCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, kernel_size=3, dropout=0.1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        layers = []
        num_channels = [hidden_size] * num_layers
        
        for i in range(num_layers):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
        self.output_proj = nn.Conv1d(num_channels[-1], hidden_size, 1)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        
        out = self.network(x)
        out = self.output_proj(out)
        
        return out.transpose(1, 2)




# ============================================================================
# TCRF ¶à×¨¼ÒÈÚºÏ»úÖÆ (TemporalContextRoutingFusion)
# ============================================================================

class TemporalContextRoutingFusion(nn.Module):
 
    
    def __init__(self, dim, num_experts=3, context_dim=16):
        '''
        ³õÊ¼»¯ TCRF ÈÚºÏÄ£¿é
        
        Args:
            dim: ÌØÕ÷Î¬¶È£¨Èç 150£©
            num_experts: ×¨¼ÒÊýÁ¿£¨Ä¬ÈÏ 3£©
            context_dim: ÉÏÏÂÎÄ±àÂëÎ¬¶È£¨Ä¬ÈÏ 16£©
        '''
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        

        self.context_encoder = nn.Sequential(
            nn.Linear(1, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, context_dim)
        )
        
        # ========== 2. ¶¯Ì¬Â·ÓÉÍøÂç ==========
        # ¸ù¾ÝÌØÕ÷ºÍÉÏÏÂÎÄÉú³É×¨¼ÒÈ¨ÖØ
        self.router = nn.Sequential(
            nn.Linear(dim * 2 + context_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, num_experts)
        )
        
        # ========== 3. Èý¸ö×¨¼Ò£¨¶àÁ£¶ÈµÚÒ»²ã£© ==========
        
        # ×¨¼Ò 1£ºÖðÔªËØÈÚºÏ£¨Ï¸Á£¶È£©
        # ²Ù×÷£ºx * y * sigmoid(w)
        # ²ÎÊýÁ¿£ºdim
        # ×÷ÓÃ£º²¶×½Ï¸Á£¶ÈÌØÕ÷½»»¥
        self.expert_elementwise = nn.Parameter(torch.ones(dim))
        
        # ×¨¼Ò 2£ºÖðÍ¨µÀÈÚºÏ£¨ÖÐÁ£¶È£©
        # ²Ù×÷£ºMLP([x, y])
        # ²ÎÊýÁ¿£º2*dim^2 + dim
        # ×÷ÓÃ£º²¶×½Í¨µÀ¼äÒÀÀµ¹ØÏµ
        self.expert_channel = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
        # ×¨¼Ò 3£ºÈ«¾ÖÈÚºÏ£¨´ÖÁ£¶È£©
        # ²Ù×÷£ºalpha * x + (1 - alpha) * y
        # ²ÎÊýÁ¿£º1
        # ×÷ÓÃ£º¼òµ¥ÎÈ½¡µÄÏßÐÔÈÚºÏ
        self.expert_global = nn.Parameter(torch.tensor(0.5))
        
        # ========== 4. ¶àÁ£¶ÈÈÚºÏÈ¨ÖØ£¨¶àÁ£¶ÈµÚ¶þ²ã£© ==========
        # ¿ØÖÆ×¨¼ÒÈÚºÏ½á¹ûÓëÔ­Ê¼Ïà³ËµÄ±ÈÀý
        self.granularity_weights = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.Sigmoid()
        )
    
    def forward(self, x, y, seq_len=None):
        '''
        Ç°Ïò´«²¥
        
        Args:
            x: ÌØÕ÷ 1 [batch, dim]£¨Èç retain_feat£©
            y: ÌØÕ÷ 2 [batch, dim]£¨Èç agg_feat£©
            seq_len: ÐòÁÐ³¤¶È [batch]£¨¿ÉÑ¡£¬ÓÃÓÚÊ±ÐòÉÏÏÂÎÄ¸ÐÖª£©
        
        Returns:
            output: ÈÚºÏºóµÄÌØÕ÷ [batch, dim]
        '''
        batch_size = x.size(0)
        
        # ========== ²½Öè 1£ºÊ±ÐòÉÏÏÂÎÄ¸ÐÖª ==========
        if seq_len is None:
            # Èç¹ûÃ»ÓÐÌá¹©ÐòÁÐ³¤¶È£¬Ê¹ÓÃÄ¬ÈÏÖµ
            seq_len = torch.tensor([x.size(1)] * batch_size if x.dim() > 2 
                                  else [1] * batch_size)
        
        # ½«ÐòÁÐ³¤¶È¹éÒ»»¯²¢±àÂëÎªÉÏÏÂÎÄÏòÁ¿
        # ¹éÒ»»¯£ºseq_len / 100.0£¬±ÜÃâÊýÖµ¹ý´ó
        context_input = seq_len.float().view(-1, 1).to(x.device) / 100.0
        context = self.context_encoder(context_input)  # [batch, context_dim]
        
        # ========== ²½Öè 2£º¶¯Ì¬×¨¼ÒÂ·ÓÉ ==========
        # Æ´½ÓÁ½¸öÌØÕ÷
        xy_concat = torch.cat([x, y], dim=-1)  # [batch, dim*2]
        
        # Æ´½ÓÌØÕ÷ºÍÉÏÏÂÎÄ
        router_input = torch.cat([xy_concat, context], dim=-1)
        # router_input: [batch, dim*2 + context_dim]
        
        # Éú³É×¨¼ÒÈ¨ÖØ£¨softmax ¹éÒ»»¯£©
        expert_weights = torch.softmax(self.router(router_input), dim=-1)

        out1 = x * y * torch.sigmoid(self.expert_elementwise)
        # out1: [batch, dim]
        
        # ×¨¼Ò 2£ºÖðÍ¨µÀÈÚºÏ£¨ÖÐÁ£¶È£©
        # out2 = MLP([x, y])
        # ·ÇÏßÐÔ±ä»»²¶×½Í¨µÀ¼äÒÀÀµ
        out2 = self.expert_channel(xy_concat)
        # out2: [batch, dim]
        
        # ×¨¼Ò 3£ºÈ«¾ÖÈÚºÏ£¨´ÖÁ£¶È£©
        # out3 = alpha * x + (1 - alpha) * y
        # ¼òµ¥µÄÏßÐÔ¼ÓÈ¨Æ½¾ù
        alpha = torch.sigmoid(self.expert_global)
        out3 = alpha * x + (1 - alpha) * y
        # out3: [batch, dim]
        
        # ×¨¼Ò¼ÓÈ¨ÈÚºÏ
        # expert_output = w1*out1 + w2*out2 + w3*out3
        expert_output = (expert_weights[:, 0:1] * out1 + 
                        expert_weights[:, 1:2] * out2 + 
                        expert_weights[:, 2:3] * out3)
        # expert_output: [batch, dim]
        
        # ========== ²½Öè 4£º¶àÁ£¶ÈÈÚºÏ£¨¶àÁ£¶ÈµÚ¶þ²ã£© ==========
        # Æ´½ÓÈý¸öÊä³ö
        granularity_input = torch.cat([out1, out2, expert_output], dim=-1)
        # granularity_input: [batch, dim*3]
        
        # Éú³ÉÈÚºÏÈ¨ÖØ
        granularity_w = self.granularity_weights(granularity_input)
        
        final_output = granularity_w * expert_output + (1 - granularity_w) * (x * y)
        
        return final_output
    
    def get_expert_weights(self, x, y, seq_len=None):
        batch_size = x.size(0)
        
        if seq_len is None:
            if x.dim() > 2:
                seq_len = torch.tensor([x.size(1)] * batch_size)
            else:
                seq_len = torch.tensor([1] * batch_size)
        
        context_input = seq_len.float().view(-1, 1).to(x.device) / 100.0
        context = self.context_encoder(context_input)
        
        xy_concat = torch.cat([x, y], dim=-1)
        router_input = torch.cat([xy_concat, context], dim=-1)
        expert_weights = torch.softmax(self.router(router_input), dim=-1)
        
        return expert_weights


# ============================================================================
# MultiScaleAggregation 多尺度聚合模块 (优化版)
# ============================================================================

class MultiScaleAggregation(nn.Module):
    '''
    多时间尺度聚合模块 (优化版)
    
    改进点:
    1. 向量化计算 - 使用 mask 机制替代 for 循环，支持 GPU 并行
    2. 边界安全 - 自动处理短序列情况
    3. 互斥窗口 - 避免信息冗余重复计算
    4. 多池化融合 - Mean + Max 捕捉不同特征
    5. 权重约束 - Softmax 归一化确保稳定
    6. 增强输出层 - MLP 增加非线性表达能力
    
    窗口设计 (互斥):
    - 短期: 最近 2 次 (急性变化)
    - 中期: 第 3-7 次 (近期趋势)  
    - 长期: 第 8 次以前 (整体状态)
    '''
    
    def __init__(self, hidden_size, code_num, window_sizes=None, dropout=0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.code_num = code_num
        
        if window_sizes is None:
            self.window_sizes = [2, 7, None]
        else:
            self.window_sizes = window_sizes
        
        self.scale_weights = nn.Parameter(torch.ones(len(self.window_sizes)))
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, code_num)
        )
        
        self.eps = 1e-9
    
    def _generate_window_mask(self, lens, max_seq_len, window_start, window_end):
        '''
        生成窗口 mask (向量化)
        
        Args:
            lens: [batch] 每个样本的实际序列长度
            max_seq_len: 最大序列长度
            window_start: 窗口起始位置 (相对于序列末尾, 负数)
            window_end: 窗口结束位置 (相对于序列末尾, 负数或0)
        
        Returns:
            mask: [batch, seq_len] 1表示在窗口内，0表示在窗口外
        '''
        batch_size = lens.size(0)
        device = lens.device
        
        lens = torch.clamp(lens, min=0, max=max_seq_len)
        positions = torch.arange(max_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        valid_mask = positions < lens.unsqueeze(1)
        
        if window_start is None:
            start_pos = torch.zeros_like(lens)
        else:
            start_pos = torch.clamp(lens - window_start, min=0)
        
        if window_end is None:
            end_pos = lens
        else:
            end_pos = torch.clamp(lens - window_end, min=0)
        
        window_mask = (positions >= start_pos.unsqueeze(1)) & (positions < end_pos.unsqueeze(1))
        
        return window_mask & valid_mask
    
    def forward(self, seq_feat, lens=None):
        '''
        前向传播 (向量化)
        
        Args:
            seq_feat: [batch, seq_len, hidden_size]
            lens: [batch] 每个样本的实际序列长度
        
        Returns:
            output: [batch, code_num]
        '''
        batch_size, max_seq_len, _ = seq_feat.shape
        device = seq_feat.device
        
        if lens is None:
            lens = torch.full((batch_size,), max_seq_len, device=device, dtype=torch.long)
        else:
            lens = torch.clamp(lens.to(device), min=0, max=max_seq_len)
        
        aggregated_scales = []
        
        for idx, w_size in enumerate(self.window_sizes):
            if idx == 0:
                mask = self._generate_window_mask(lens, max_seq_len, w_size, 0)
            elif idx == 1:
                prev_size = self.window_sizes[0] if self.window_sizes[0] is not None else 0
                mask = self._generate_window_mask(lens, max_seq_len, w_size, prev_size)
            else:
                prev_size = self.window_sizes[1] if self.window_sizes[1] is not None else 0
                mask = self._generate_window_mask(lens, max_seq_len, None, prev_size)
            
            mask_float = mask.unsqueeze(-1).to(seq_feat.dtype)
            mask_sum = mask_float.sum(dim=1).clamp(min=self.eps)
            
            masked_feat = seq_feat * mask_float
            
            mean_feat = masked_feat.sum(dim=1) / mask_sum
            
            mask_expanded = mask.unsqueeze(-1).expand_as(seq_feat)
            masked_for_max = seq_feat.clone()
            masked_for_max[mask_expanded == 0] = float('-inf')
            max_feat = masked_for_max.max(dim=1)[0]
            max_feat = torch.where(torch.isinf(max_feat), torch.zeros_like(max_feat), max_feat)
            max_feat = torch.where(torch.isnan(max_feat), torch.zeros_like(max_feat), max_feat)
            
            mean_feat = torch.where(torch.isnan(mean_feat), torch.zeros_like(mean_feat), mean_feat)
            
            scale_feat = torch.cat([mean_feat, max_feat], dim=-1)
            aggregated_scales.append(scale_feat)
        
        stacked = torch.stack(aggregated_scales, dim=1)
        
        norm_weights = torch.softmax(self.scale_weights, dim=0).view(1, -1, 1)
        fused = (stacked * norm_weights).sum(dim=1)
        
        output = self.output_layer(fused)
        
        return output
    
    def get_window_weights(self):
        return torch.softmax(self.scale_weights, dim=0).detach().cpu().numpy()


def compute_simple_fusion(x, y, mode='sum_mul'):
    if mode == 'mul':
        return x * y
    if mode == 'sum_mul':
        return x + y + x * y
    raise ValueError(f'Unsupported simple fusion mode: {mode}')


class DualStreamAlignment(nn.Module):
    def __init__(self, code_num, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.x_norm = nn.LayerNorm(code_num)
        self.y_norm = nn.LayerNorm(code_num)
        self.gate_mlp = nn.Sequential(
            nn.Linear(code_num * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, code_num * 2)
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, y, return_extras=False):
        x_norm = self.x_norm(x)
        y_norm = self.y_norm(y)
        gate_logits = self.gate_mlp(torch.cat([x_norm, y_norm], dim=-1))
        x_gate, y_gate = torch.chunk(torch.sigmoid(gate_logits), 2, dim=-1)
        beta = torch.sigmoid(self.residual_scale)

        x_out = x + beta * (x_gate * y_norm)
        y_out = y + beta * (y_gate * x_norm)

        extras = None
        if return_extras:
            extras = {
                'alignment_scale': beta.item(),
                'x_gate_mean': x_gate.mean().item(),
                'y_gate_mean': y_gate.mean().item(),
            }
        return (x_out, y_out, extras) if return_extras else (x_out, y_out)


class ResidualInteractionFusion(nn.Module):
    def __init__(self, code_num, hidden_dim=512, dropout=0.1, simple_fusion_mode='sum_mul'):
        super().__init__()
        self.simple_fusion_mode = simple_fusion_mode
        self.residual_mlp = nn.Sequential(
            nn.Linear(code_num * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, code_num)
        )
        self.output_norm = nn.LayerNorm(code_num)
        self.residual_scale = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x, y, base_fused=None, return_extras=False):
        interaction = torch.cat([x, y, x - y, x * y], dim=-1)
        residual = self.residual_mlp(interaction)
        residual = self.output_norm(residual)
        alpha = torch.sigmoid(self.residual_scale)
        base = base_fused if base_fused is not None else compute_simple_fusion(x, y, mode=self.simple_fusion_mode)
        output = base + alpha * residual

        extras = None
        if return_extras:
            extras = {
                'interaction_alpha': alpha.item(),
                'interaction_residual_norm': residual.norm(dim=-1).mean().item(),
            }
        return (output, extras) if return_extras else output


class GatedResidualInteractionFusion(nn.Module):
    def __init__(self, code_num, hidden_dim=512, dropout=0.1, simple_fusion_mode='sum_mul'):
        super().__init__()
        self.simple_fusion_mode = simple_fusion_mode
        self.gate_network = nn.Sequential(
            nn.Linear(code_num * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, code_num),
            nn.Sigmoid()
        )
        self.residual_network = nn.Sequential(
            nn.Linear(code_num * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, code_num)
        )
        self.output_norm = nn.LayerNorm(code_num)
        self.residual_scale = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x, y, base_fused=None, return_extras=False):
        gate = self.gate_network(torch.cat([x, y], dim=-1))
        interaction = torch.cat([x, y, x - y, x * torch.sigmoid(y)], dim=-1)
        residual = self.residual_network(interaction)
        residual = self.output_norm(gate * residual)
        alpha = torch.sigmoid(self.residual_scale)
        base = base_fused if base_fused is not None else compute_simple_fusion(x, y, mode=self.simple_fusion_mode)
        output = base + alpha * residual

        extras = None
        if return_extras:
            extras = {
                'gated_interaction_alpha': alpha.item(),
                'gated_interaction_gate_mean': gate.mean().item(),
                'gated_interaction_gate_std': gate.std().item(),
            }
        return (output, extras) if return_extras else output


class CrossAttentionFusion(nn.Module):
    def __init__(self, code_num, hidden_dim=256, dropout=0.1, simple_fusion_mode='sum_mul'):
        super().__init__()
        self.simple_fusion_mode = simple_fusion_mode
        self.x_proj = nn.Linear(code_num, hidden_dim)
        self.y_proj = nn.Linear(code_num, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim * 2, code_num)
        self.output_norm = nn.LayerNorm(code_num)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(-1.0))

    def _cross_branch(self, src, ctx):
        q = self.q_proj(src)
        k = self.k_proj(ctx)
        v = self.v_proj(ctx)
        attn = torch.sigmoid((q * k) / np.sqrt(q.size(-1)))
        return attn * v

    def forward(self, x, y, base_fused=None, return_extras=False):
        x_hidden = self.x_proj(x)
        y_hidden = self.y_proj(y)

        xy_context = self._cross_branch(x_hidden, y_hidden)
        yx_context = self._cross_branch(y_hidden, x_hidden)
        fused_context = torch.cat([xy_context, yx_context], dim=-1)

        cross_out = self.output_proj(self.dropout(fused_context))
        cross_out = self.output_norm(cross_out)
        alpha = torch.sigmoid(self.residual_scale)
        base = base_fused if base_fused is not None else compute_simple_fusion(x, y, mode=self.simple_fusion_mode)
        output = base + alpha * cross_out

        extras = None
        if return_extras:
            extras = {
                'cross_attention_alpha': alpha.item(),
                'cross_attention_context_norm': fused_context.norm(dim=-1).mean().item(),
            }
        return (output, extras) if return_extras else output


class LabelWiseGateFusion(nn.Module):
    def __init__(self, code_num, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(code_num * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, code_num)
        )
        self.output_norm = nn.LayerNorm(code_num)
        self.residual_scale = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x, y, base_fused=None, return_extras=False):
        gate = torch.sigmoid(self.gate_network(torch.cat([x, y], dim=-1)))
        gate_fused = gate * x + (1.0 - gate) * y + x * y
        gate_fused = self.output_norm(gate_fused)
        alpha = torch.sigmoid(self.residual_scale)
        base = base_fused if base_fused is not None else x * y
        output = base + alpha * gate_fused

        extras = None
        if return_extras:
            extras = {
                'label_gate_alpha': alpha.item(),
                'label_gate_mean': gate.mean().item(),
                'label_gate_std': gate.std().item(),
            }
        return (output, extras) if return_extras else output


class LabelGraphResidualHead(nn.Module):
    def __init__(self, adj_matrix, dropout=0.1):
        super().__init__()
        if sparse.issparse(adj_matrix):
            adj_tensor = torch.from_numpy(adj_matrix.toarray()).float()
        elif torch.is_tensor(adj_matrix):
            adj_tensor = adj_matrix.detach().float().cpu()
        else:
            adj_tensor = torch.tensor(np.array(adj_matrix), dtype=torch.float32)

        eye = torch.eye(adj_tensor.size(0), dtype=adj_tensor.dtype)
        adj_tensor = adj_tensor + eye
        row_sum = adj_tensor.sum(dim=-1, keepdim=True).clamp(min=1.0)
        normalized_adj = adj_tensor / row_sum

        self.register_buffer('normalized_adj', normalized_adj)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(-1.0))

    def forward(self, logits, return_extras=False):
        propagated = torch.matmul(logits, self.normalized_adj.to(logits.device))
        propagated = self.dropout(propagated)
        alpha = torch.sigmoid(self.residual_scale)
        output = logits + alpha * propagated

        extras = None
        if return_extras:
            extras = {
                'label_graph_alpha': alpha.item(),
                'label_graph_norm': propagated.norm(dim=-1).mean().item(),
            }
        return (output, extras) if return_extras else output


# ============================================================================
# ImprovedTCRF - 改进版时序上下文路由融合
# 核心思想: 简单融合作为主干，TCRF学习残差修正
# ============================================================================

class ImprovedTCRF(nn.Module):
    '''
    改进版时序上下文路由融合模块
    
    核心改进:
    1. 简单融合作为主干 - 保留噪声过滤优势（元素级乘法是天然共识机制）
    2. TCRF作为残差修正 - 只学习简单融合处理不了的模式
    3. 可学习权重 - 模型自己决定简单融合和TCRF的比例
    4. 轻量级设计 - 参数量大幅减少，避免过拟合
    
    参考: 3.md 方案A
    '''
    
    def __init__(self, code_num, embed_dim=256, hidden_dim=512, num_experts=3, dropout=0.3,
                 simple_fusion_mode='sum_mul'):
        super().__init__()
        
        self.code_num = code_num
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.simple_fusion_mode = simple_fusion_mode
        
        self.input_proj = nn.Sequential(
            nn.Linear(code_num, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.time_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Sigmoid()
        )
        
        router_input_dim = embed_dim * 2 + embed_dim
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts)
        )
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim)
            ) for _ in range(num_experts)
        ])
        
        self.output_proj = nn.Linear(embed_dim, code_num)
        
        self.simple_fusion_weight = nn.Parameter(torch.tensor(0.7))
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, y, seq_len=None, return_extras=False):
        '''
        前向传播
        
        Args:
            x: [B, code_num] output1 (时间聚合)
            y: [B, code_num] output2 (注意力)
            seq_len: [B] 序列长度（可选）
            return_extras: 是否返回额外信息
        
        Returns:
            final_output: [B, code_num] 融合后的特征
            extras: dict 额外信息（可选）
        '''
        B = x.shape[0]
        device = x.device
        
        simple_fusion = compute_simple_fusion(x, y, mode=self.simple_fusion_mode)
        
        x_feat = self.input_proj(x)
        y_feat = self.input_proj(y)
        
        if seq_len is not None:
            if seq_len.dim() == 0:
                seq_len = seq_len.unsqueeze(0).expand(B)
            time_emb = self.time_encoder(seq_len.float().unsqueeze(-1))
        else:
            time_emb = torch.zeros(B, self.embed_dim, device=device)
        
        router_input = torch.cat([x_feat, y_feat, time_emb], dim=-1)
        expert_weights = torch.softmax(self.router(router_input), dim=-1)
        
        concat_feat = torch.cat([x_feat, y_feat], dim=-1)
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(concat_feat))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        tcrf_fused = torch.sum(expert_outputs * expert_weights.unsqueeze(-1), dim=1)
        tcrf_fused = self.layer_norm(tcrf_fused)
        tcrf_fused = self.dropout(tcrf_fused)
        tcrf_output = self.output_proj(tcrf_fused)
        
        alpha = torch.sigmoid(self.simple_fusion_weight)
        final_output = alpha * simple_fusion + (1 - alpha) * tcrf_output
        
        extras = None
        if return_extras:
            extras = {
                'expert_weights': expert_weights.detach().cpu().numpy(),
                'simple_fusion_weight': alpha.item(),
                'simple_fusion_mode': self.simple_fusion_mode,
            }
        
        return final_output, extras
    
    def get_fusion_weight(self):
        return torch.sigmoid(self.simple_fusion_weight).item()


# ============================================================================
# GatedSimpleFusion - 门控简单融合（更轻量的方案B）
# ============================================================================

class GatedSimpleFusion(nn.Module):
    '''
    门控简单融合：TCRF只学习每个维度的缩放因子
    
    特点:
    1. 参数量极少（比原TCRF少30%）
    2. 保留简单融合的噪声过滤优势
    3. 自适应调整每个维度的权重
    '''
    
    def __init__(self, code_num, embed_dim=256, dropout=0.3, simple_fusion_mode='sum_mul'):
        super().__init__()
        
        self.code_num = code_num
        self.simple_fusion_mode = simple_fusion_mode
        
        self.gate_network = nn.Sequential(
            nn.Linear(code_num, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, code_num),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, y, seq_len=None, return_extras=False):
        '''
        前向传播
        
        Args:
            x: [B, code_num] output1
            y: [B, code_num] output2
            seq_len: [B] 序列长度（可选，此版本不使用）
        
        Returns:
            output: [B, code_num] 门控融合后的特征
            extras: None
        '''
        simple = compute_simple_fusion(x, y, mode=self.simple_fusion_mode)
        
        gate = self.gate_network(x + y)
        
        output = simple * gate
        
        extras = None
        if return_extras:
            extras = {
                'gate_mean': gate.mean().item(),
                'gate_std': gate.std().item(),
                'simple_fusion_mode': self.simple_fusion_mode,
            }
        
        return output, extras


# ============================================================================
# StableMoEFusion - 基于 StableMoE (2022) 的稳定路由融合
# 核心改进: 路由EMA平滑 + 负载均衡损失 + 简单融合残差
# ============================================================================

class StableMoEFusion(nn.Module):
    '''
    基于 StableMoE (2022) 的稳定路由融合
    
    核心改进:
    1. 路由权重 EMA 平滑 - 避免剧烈波动
    2. 专家多样性损失 - 防止同质化
    3. 保留简单融合残差 - 共识滤波优势
    4. Top-K 稀疏路由 - 提高效率
    5. 瓶颈架构降参 - BASE Layer 思路
    
    参考: 3.md 方案1
    '''
    
    def __init__(self, code_num, embed_dim=512, hidden_dim=512,
                 num_experts=3, top_k=1, dropout=0.01,
                 load_balance_weight=0.001, router_ema_decay=0.99,
                 simple_fusion_mode='sum_mul'):
        super().__init__()
        
        self.code_num = code_num
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.ema_decay = router_ema_decay
        self.load_balance_weight = load_balance_weight
        self.simple_fusion_mode = simple_fusion_mode
        
        self.input_proj = nn.Sequential(
            nn.Linear(code_num, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.time_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Sigmoid()
        )
        
        router_input_dim = embed_dim * 2 + embed_dim
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts)
        )
        
        self.register_buffer('router_ema', torch.ones(num_experts) / num_experts)
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim)
            ) for _ in range(num_experts)
        ])
        
        self.output_proj = nn.Linear(embed_dim, code_num)
        
        self.residual_weight = nn.Parameter(torch.tensor(0.95))# 95，80，85，93，96 ，92
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, y, seq_len=None, return_extras=False):
        '''
        前向传播
        
        Args:
            x: [B, code_num] output1 (时间聚合)
            y: [B, code_num] output2 (注意力)
            seq_len: [B] 序列长度
            return_extras: 是否返回额外信息
        '''
        B = x.shape[0]
        device = x.device
        
        simple_fusion = compute_simple_fusion(x, y, mode=self.simple_fusion_mode)
        
        x_feat = self.input_proj(x)
        y_feat = self.input_proj(y)
        
        if seq_len is not None:
            if seq_len.dim() == 0:
                seq_len = seq_len.unsqueeze(0).expand(B)
            time_emb = self.time_encoder(seq_len.float().unsqueeze(-1))
        else:
            time_emb = torch.zeros(B, self.embed_dim, device=device)
        
        router_input = torch.cat([x_feat, y_feat, time_emb], dim=-1)
        router_logits = self.router(router_input)
        
        top_k_values, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        expert_weights = F.softmax(top_k_values, dim=-1)
        
        sparse_weights = torch.zeros_like(router_logits)
        sparse_weights.scatter_(-1, top_k_indices, expert_weights)
        
        with torch.no_grad():
            batch_mean_weights = sparse_weights.mean(dim=0)
            self.router_ema = self.ema_decay * self.router_ema + (1 - self.ema_decay) * batch_mean_weights
        
        concat_feat = torch.cat([x_feat, y_feat], dim=-1)
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(concat_feat))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        moe_fused = torch.sum(expert_outputs * sparse_weights.unsqueeze(-1), dim=1)
        moe_fused = self.layer_norm(moe_fused)
        moe_fused = self.dropout(moe_fused)
        moe_output = self.output_proj(moe_fused)
        
        alpha = torch.sigmoid(self.residual_weight)
        output = alpha * simple_fusion + (1 - alpha) * moe_output
        
        extras = None
        if return_extras:
            extras = {
                'expert_weights': sparse_weights.detach().cpu().numpy(),
                'residual_weight': alpha.item(),
                'router_ema': self.router_ema.cpu().numpy(),
                'top_k_indices': top_k_indices.detach().cpu().numpy(),
                'simple_fusion_mode': self.simple_fusion_mode,
            }
        
        return output, extras
    
    def compute_load_balance_loss(self, expert_weights):
        '''
        负载均衡损失（防止专家同质化）
        借鉴 Switch Transformer / GShard
        
        Args:
            expert_weights: [B, num_experts] 专家权重
        
        Returns:
            balance_loss: 负载均衡损失
        '''
        if isinstance(expert_weights, np.ndarray):
            expert_weights = torch.from_numpy(expert_weights)
        
        expert_usage = expert_weights.mean(dim=0)
        
        target = torch.ones_like(expert_usage) / self.num_experts
        
        usage_std = torch.std(expert_usage)
        usage_mean = torch.mean(expert_usage) + 1e-8
        cv = usage_std / usage_mean
        
        balance_loss = cv * self.load_balance_weight
        
        return balance_loss
    
    def get_expert_usage(self):
        return self.router_ema.cpu().numpy()
    
    def get_residual_weight(self):
        return torch.sigmoid(self.residual_weight).item()


