import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from models.layers import SimpleTransitionLayer, FusionTransitionLayer, SoftPositionFusionLayer, LightweightTCN, TemporalContextRoutingFusion, MultiScaleAggregation, ImprovedTCRF, GatedSimpleFusion, StableMoEFusion, LogitPriorLayer, DynamicResidualFusionTransitionLayer, DynamicVectorResidualFusionTransitionLayer, SampleAdaptiveFusionTransitionLayer, ConfidenceAwareFusionTransitionLayer, DualStreamAlignment, ResidualInteractionFusion, GatedResidualInteractionFusion, CrossAttentionFusion, LabelWiseGateFusion, LabelGraphResidualHead, EndToEndSemanticTransitionGenerator
from scipy import sparse

device = 'cuda'


class RETAIN(nn.Module):
    def __init__(self, inputDimSize = 2813, embDimSize =100, hiddenDimSize=10, numClass=2813, dropout_rate=0.1):
        super(RETAIN, self).__init__()
        self.inputDimSize = inputDimSize
        self.relu = torch.nn.ReLU()
        self.inputDimSize = inputDimSize
        self.RNN = torch.nn.GRU(input_size=inputDimSize, hidden_size=hiddenDimSize, num_layers=5, batch_first=True,
                                bidirectional=False)
        self.RNN1 = torch.nn.GRU(input_size=inputDimSize, hidden_size=hiddenDimSize, num_layers=5, batch_first=True,
                                 bidirectional=False)
        self.W_emb = nn.Parameter(torch.rand((inputDimSize, embDimSize)).float(), requires_grad=True)
        self.b_ema = nn.Parameter(torch.rand(embDimSize).float(), requires_grad=True)

        self.w_a = nn.Parameter(torch.rand(hiddenDimSize,1).float(), requires_grad=True)
        self.b_a = nn.Parameter(torch.rand(1).float(), requires_grad=True)

        self.w_a1 = nn.Parameter(torch.rand(hiddenDimSize, 1).float(), requires_grad=True)
        self.b_a1 = nn.Parameter(torch.rand(1).float(), requires_grad=True)

        self.dropout_rate = dropout_rate

        self.W_output = nn.Parameter(torch.rand(embDimSize, numClass).float(), requires_grad=True)
        self.b_output = nn.Parameter(torch.zeros(numClass).float(), requires_grad=True)

        # self.classifier = Classifier(100, output_size, dropout_rate, activation)

    # @torchsnooper.snoop()
    def forward(self, x, visit_lens):
        mask = torch.arange(x.shape[1], device=device)[None, :] < visit_lens[:, None]
        mask_attention = (-10e8)*(1-mask.int()).unsqueeze(-1)
        x = x.float()  # å°xè½¬æ¢ä¸ºfloat32
        self.W_emb = self.W_emb.float()  # å°æéè½¬æ¢ä¸ºfloat32
        self.b_ema = self.b_ema.float()  # å°åå·®è½¬æ¢ä¸ºfloat32

        x1 = self.relu(torch.matmul(x, self.W_emb) + self.b_ema)

        x2, _ = self.RNN(x)  # g
        a = nn.Softmax(dim=1)(torch.matmul(x2, self.w_a) + self.b_a + mask_attention) #
        x3, _ = self.RNN1(x) #h
        b = nn.Tanh()(torch.matmul(x3, self.w_a1) + self.b_a1)
        # ä¸å­å¨çå¥é¢è®°å½å¯¹åºçaæéä¸?ï¼å¨è®¡ç®ä¸­è¢«å¿½ç¥
        c = torch.sum(a * b * x , dim=1)
        return c

class Classifier(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.dropout(x)
        output = self.linear(output)
        return output

class SingleHeadAttentionLayer1(nn.Module):
    def __init__(self, query_size, key_size, value_size, attention_size):
        super().__init__()
        self.attention_size = attention_size
        self.dense_q = nn.Linear(query_size, attention_size)
        self.dense_k = nn.Linear(key_size, attention_size)
        self.dense_v = nn.Linear(value_size, value_size)

    def forward(self, q, k, v):
        query = self.dense_q(q.float().cuda())
        key = self.dense_k(k.float().cuda())
        value = self.dense_v(v.float().cuda())
        g = torch.div(torch.matmul(query, key.T), math.sqrt(self.attention_size))
        score = torch.softmax(g, dim=-1)
        output = torch.sum(torch.unsqueeze(score, dim=-1) * value, dim=-2)
        return output

class SingleHeadAttentionLayer2(nn.Module):
    def __init__(self, query_size, key_size, value_size, attention_size):
        super().__init__()
        self.attention_size = attention_size
        self.dense_q = nn.Linear(query_size, attention_size)
        self.dense_k = nn.Linear(key_size, attention_size)
        self.dense_v = nn.Linear(value_size, value_size)

    def forward(self, q, k, v):
        query = self.dense_q(q.float().cuda())
        key = self.dense_k(k.float().cuda())
        value = self.dense_v(v.float().cuda())
        g = torch.div(torch.matmul(query, key.T), math.sqrt(self.attention_size))
        score = torch.softmax(g, dim=-1)
        output = torch.sum(torch.unsqueeze(score, dim=-1) * value, dim=-2)
        return output

class ScaledDotProductAttention(nn.Module):
    def __init__(self, feature_dim, attn_dim):
        super(ScaledDotProductAttention, self).__init__()
        # åå§åæéç©é?        self.W_q = nn.Parameter(torch.randn(feature_dim, attn_dim))
        self.W_k = nn.Parameter(torch.randn(feature_dim, attn_dim))
        self.W_v = nn.Parameter(torch.randn(feature_dim, attn_dim))
        # æå½±å±ï¼ç¨äºå°è¾åºæå½±ååå§çç¹å¾ç»´åº?        self.proj = nn.Parameter(torch.randn(attn_dim, feature_dim))

    def forward(self, Q, K, V):
        # æ å°Q, K, Vå°ä¸­é´ç»´åº?        Q = torch.matmul(Q, self.W_q)
        K = torch.matmul(K, self.W_k)
        V = torch.matmul(V, self.W_v)
        # è®¡ç®ç¼©æ¾ç¹ç§¯æ³¨æå?        scaling_factor = torch.sqrt(torch.tensor(self.W_k.size(1), dtype=torch.float32))
        scores = torch.dot(Q, K) / scaling_factor
        # ç±äºè¿éæ¯åä¸ªæ¥è¯¢åé®å¼å¯¹ï¼æä»¬ç´æ¥å°softmaxåçå¾åä½ä¸ºæé
        attention_weights = F.softmax(torch.tensor([scores]), dim=0)
        # åºç¨æ³¨æåæéå°V
        attn_output = V * attention_weights.to(device='cuda:0')
        # æå½±è¾åºååå§ç¹å¾ç»´åº?        output = torch.matmul(attn_output, self.proj)
        return output


class MyNetwork(nn.Module):
    def __init__(self, N=2428):
        super(MyNetwork, self).__init__()
        self.linear = nn.Linear(N, N)  # Linear layer to transform A

    def forward(self, A, B):
        transformed_A = self.linear(A)  # Transform A
        combined = torch.matmul(transformed_A, B)  # Combine transformed A with B
        return combined

class Attention(nn.Module):
    def __init__(self, N):
        super(Attention, self).__init__()
        self.query = nn.Linear(N, N)
        self.key = nn.Linear(N, N)
        self.value = nn.Linear(N, N)

    def forward(self, A, B):
        query = self.query(A)  # çææ¥è¯¢åé
        key = self.key(B)  # çæé®åé?        value = self.value(B)  # çæå¼åé?
        # è®¡ç®æ³¨æååæ?        attn_scores = F.softmax(torch.matmul(query, key.t()), dim=-1)

        # èç¦å°æç¸å³çåç´?        focused = torch.matmul(attn_scores, value)

        return focused


class GatedUnit(nn.Module):
    def __init__(self, N):
        super(GatedUnit, self).__init__()
        self.gate = nn.Linear(2 * N, N)

    def forward(self, A, B):
        C = torch.matmul(A, B)

        # èç»AåCï¼ç¨äºé¨æ§åå?        combined = torch.cat((A, C), dim=-1)

        # é¨æ§æºå¶
        gate_values = torch.sigmoid(self.gate(combined))

        # åºç¨é¨æ§
        gated_output = gate_values * C

        return gated_output


class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        N, seq_length, embed_size = x.shape

        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        energy = torch.bmm(queries, keys.transpose(1, 2))
        energy = energy / (embed_size ** (1 / 2))
        if mask is not None:
            key_mask = mask.unsqueeze(1)
            energy = energy.masked_fill(~key_mask, float('-inf'))

        attention = torch.softmax(energy, dim=2)
        attention = torch.nan_to_num(attention, nan=0.0)

        if mask is not None:
            query_mask = mask.unsqueeze(-1).to(attention.dtype)
            attention = attention * query_mask

        out = torch.bmm(attention, values)
        out = self.fc_out(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


# class SelfAttention(nn.Module):
#     def __init__(self, N):
#         super(SelfAttention, self).__init__()
#         self.query = nn.Linear(N, N)
#         self.key = nn.Linear(N, N)
#         self.value = nn.Linear(N, N)
#
#     def forward(self, A):
#         query = self.query(A)  # çææ¥è¯¢åé
#         key = self.key(A)  # çæé®åé?#         value = self.value(A)  # çæå¼åé?#         g = torch.div(torch.matmul(query, key.T), math.sqrt(self.attention_size))
#         score = torch.softmax(g, dim=-1)  # æ¯ä¸ªæéçæé?#         output = torch.sum(torch.unsqueeze(score, dim=-1) * value, dim=-2)
#         return output

        # # è®¡ç®æ³¨æååæ?        # attn_scores = F.softmax(torch.matmul(query, key.t()), dim=-1)
        #
        # # èç¦å°æç¸å³çåç´?        # focused = torch.matmul(attn_scores, value)
        #
        # return focused
#
#
# class SelfAttention(nn.Module):
#     def __init__(self, size):
#         super(SelfAttention, self).__init__()
#         # å®ä¹Q,K,Vçåæ¢ç©éµï¼è¿éå®ä»¬çç»´åº¦é½è®¾ç½®ä¸?50
#         self.query = nn.Linear(size, size)
#         self.key = nn.Linear(size, size)
#         self.value = nn.Linear(size, size)
#
#     def forward(self, x):
#         # x is (batch_size, seq_len, size)
#         # å¯¹åº è¾å¥ç»´åº¦ 32 x 69 x 150
#         batch_size, seq_len,size = x.size()
#
#         # çææ¥è¯¢ãé®ãå?#         Q = self.query(x)  # (batch_size, seq_len, size)
#         K = self.key(x)  # (batch_size, seq_len, size)
#         V = self.value(x)  # (batch_size, seq_len, size)
#
#         # è®¡ç®æ¥è¯¢åé®çç¹ç§?é¤ä»¥sizeçå¹³æ¹æ ¹è¿è¡ç¼©æ¾
#         attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / size ** 0.5  # (batch_size, seq_len, seq_len)
#
#         # åºç¨softmaxå½æ°å¾å°çæ¯æ¯ä¸ªåºåçæé?æ³¨æmask
#         attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len, seq_len)
#
#         # è®¡ç®å æåï¼å¾å°æ³¨æååçè¾å?#         output = torch.matmul(attention_weights, V)  # (batch_size, seq_len, size)
#
#         return output

class Model(nn.Module):
    def __init__(self, code_num, code_size,
                 adj, graph_size, hidden_size, t_attention_size, t_output_size,
                 output_size, dropout_rate,
                 transition_T=None, transition_T_llm=None,
                 use_transition=True, use_llm_transition=False, llm_alpha=0.5,
                 fusion_type='fusion', max_seq_len=70,
                 use_tcn=False, tcn_num_layers=3, tcn_kernel_size=3,
                 stream_fusion_type='simple',
                 use_multi_scale_agg=False,
                 tcrf_type='improved',
                 use_packed_sequence=True,
                 use_masked_attention=True,
                 transition_mode='feature_fusion',
                 prior_lambda=0.5,
                 transition_fusion_variant='static',
                 use_stream_feature_norm=True,
                 use_dual_stream_alignment=False,
                 use_residual_interaction_fusion=False,
                 use_label_wise_gate=False,
                 use_label_graph_head=False,
                 use_gated_residual_interaction_fusion=False,
                 use_cross_attention_fusion=False,
                 simple_fusion_mode='sum_mul',
                 context_branch='retain_attention',
                 transition_gate_mode='sample_adaptive',
                 shared_transition_projection=False,
                 use_end_to_end_llm_transition=False,
                 label_embeddings=None,
                 end_to_end_candidate_mask=None,
                 end_to_end_projector_bottleneck_dim=128,
                 end_to_end_projector_hidden_dim=256,
                 end_to_end_projector_dropout=0.1,
                 end_to_end_matrix_chunk_size=64,
                 end_to_end_transition_device=None):
        super().__init__()
        self.stream_fusion_type = stream_fusion_type
        self.use_multi_scale_agg = use_multi_scale_agg
        self.tcrf_type = tcrf_type
        self.use_packed_sequence = use_packed_sequence
        self.use_masked_attention = use_masked_attention
        self.end_to_end_transition_device = end_to_end_transition_device
        self.transition_mode = transition_mode
        self.transition_fusion_variant = transition_fusion_variant
        self.use_stream_feature_norm = use_stream_feature_norm
        self.use_dual_stream_alignment = use_dual_stream_alignment
        self.use_residual_interaction_fusion = use_residual_interaction_fusion
        self.use_label_wise_gate = use_label_wise_gate
        self.use_label_graph_head = use_label_graph_head and output_size == code_num
        self.use_gated_residual_interaction_fusion = use_gated_residual_interaction_fusion
        self.use_cross_attention_fusion = use_cross_attention_fusion
        self.context_branch = context_branch
        self.transition_gate_mode = transition_gate_mode
        self.shared_transition_projection = shared_transition_projection
        self.use_end_to_end_llm_transition = use_end_to_end_llm_transition
        self.lstm_hidden_size = 10
        self.attention1 = SingleHeadAttentionLayer1(query_size=code_num, key_size=code_num, value_size=code_num, attention_size=50)
        if self.context_branch == 'interaction_mlp':
            self.context_interaction = nn.Sequential(
                nn.Linear(code_num * 4, code_num),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(code_num, code_num),
            )
            self.kv_context_interaction = None
            self.kv_context_attention = None
            self.kv_context_query_projection = None
            self.retain_query_projection = None
            self.retain_query_key_projection = None
            self.retain_query_value_projection = None
            self.retain_query_output_projection = None
        elif self.context_branch == 'kv_aware_attention':
            self.context_interaction = None
            self.kv_context_query_projection = nn.Linear(self.lstm_hidden_size, code_num)
            self.kv_context_interaction = nn.Sequential(
                nn.Linear(code_num * 3, code_num),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(code_num, code_num),
            )
            self.kv_context_attention = SingleHeadAttentionLayer1(
                query_size=code_num,
                key_size=code_num,
                value_size=code_num,
                attention_size=50,
            )
            self.retain_query_projection = None
            self.retain_query_key_projection = None
            self.retain_query_value_projection = None
            self.retain_query_output_projection = None
        elif self.context_branch == 'retain_query_attention':
            self.context_interaction = None
            self.kv_context_interaction = None
            self.kv_context_attention = None
            self.kv_context_query_projection = None
            self.retain_query_projection = nn.Linear(code_num, self.lstm_hidden_size)
            self.retain_query_key_projection = nn.Linear(self.lstm_hidden_size, self.lstm_hidden_size)
            self.retain_query_value_projection = nn.Linear(self.lstm_hidden_size, self.lstm_hidden_size)
            self.retain_query_output_projection = nn.Sequential(
                nn.Linear(self.lstm_hidden_size, code_num),
                nn.Dropout(dropout_rate),
            )
        elif self.context_branch == 'retain_attention':
            self.context_interaction = None
            self.kv_context_interaction = None
            self.kv_context_attention = None
            self.kv_context_query_projection = None
            self.retain_query_projection = None
            self.retain_query_key_projection = None
            self.retain_query_value_projection = None
            self.retain_query_output_projection = None
        else:
            raise ValueError(f'Unsupported context_branch: {self.context_branch}')
        self.classifier = Classifier(code_num, output_size, dropout_rate)
        self.time_layer4 = torch.nn.Linear(10, code_num)
        self.retain = RETAIN(inputDimSize=code_num, embDimSize=100, hiddenDimSize=10, numClass=code_num)
        self.selfattention = SelfAttention(10)
        
        if stream_fusion_type == 'tcrf':
            if tcrf_type == 'improved':
                self.tcrf = ImprovedTCRF(code_num=code_num, simple_fusion_mode=simple_fusion_mode)
            elif tcrf_type == 'gated':
                self.tcrf = GatedSimpleFusion(code_num=code_num, simple_fusion_mode=simple_fusion_mode)
            elif tcrf_type == 'stable_moe':
                self.tcrf = StableMoEFusion(code_num=code_num, simple_fusion_mode=simple_fusion_mode)
            else:
                self.tcrf = TemporalContextRoutingFusion(dim=code_num, num_experts=3, context_dim=16)
        else:
            self.tcrf = None
        
        if use_multi_scale_agg:
            self.multi_scale_agg = MultiScaleAggregation(hidden_size=10, code_num=code_num)
        else:
            self.multi_scale_agg = None
            self.window_weights = nn.Parameter(torch.ones(3) / 3)
        
        self.use_tcn = use_tcn
        if use_tcn:
            self.tcn = LightweightTCN(
                input_size=code_num,
                hidden_size=10,
                num_layers=tcn_num_layers,
                kernel_size=tcn_kernel_size,
                dropout=dropout_rate
            )
        else:
            self.lstm = nn.LSTM(code_num, 10, 10, batch_first=True)
        
        self.layernorm1 = nn.LayerNorm(10)
        self.layernorm2 = nn.LayerNorm(10)
        if self.use_stream_feature_norm:
            self.output1_norm = nn.LayerNorm(code_num)
            self.output2_norm = nn.LayerNorm(code_num)
            self.output1_scale = nn.Parameter(torch.tensor(1.0))
            self.output2_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.output1_norm = None
            self.output2_norm = None
            self.output1_scale = None
            self.output2_scale = None

        self.dual_stream_alignment = DualStreamAlignment(code_num=code_num, hidden_dim=max(64, code_num // 4), dropout=dropout_rate) \
            if self.use_dual_stream_alignment else None
        self.residual_interaction_fusion = ResidualInteractionFusion(
            code_num=code_num,
            hidden_dim=max(128, code_num // 2),
            dropout=dropout_rate,
            simple_fusion_mode=simple_fusion_mode
        ) if self.use_residual_interaction_fusion else None
        self.label_wise_gate = LabelWiseGateFusion(
            code_num=code_num,
            hidden_dim=max(64, code_num // 4),
            dropout=dropout_rate
        ) if self.use_label_wise_gate else None
        self.gated_residual_interaction_fusion = GatedResidualInteractionFusion(
            code_num=code_num,
            hidden_dim=max(128, code_num // 2),
            dropout=dropout_rate,
            simple_fusion_mode=simple_fusion_mode
        ) if self.use_gated_residual_interaction_fusion else None
        self.cross_attention_fusion = CrossAttentionFusion(
            code_num=code_num,
            hidden_dim=max(128, code_num // 8),
            dropout=dropout_rate,
            simple_fusion_mode=simple_fusion_mode
        ) if self.use_cross_attention_fusion else None
        self.label_graph_head = LabelGraphResidualHead(adj_matrix=adj, dropout=dropout_rate) \
            if self.use_label_graph_head else None

        self.transition_T = transition_T
        self.use_llm_transition = use_llm_transition and transition_T is not None
        self.end_to_end_transition_generator = None
        if self.use_end_to_end_llm_transition:
            if label_embeddings is None:
                raise ValueError('label_embeddings are required when use_end_to_end_llm_transition=True')
            self.end_to_end_transition_generator = EndToEndSemanticTransitionGenerator(
                label_embeddings=label_embeddings,
                bottleneck_dim=end_to_end_projector_bottleneck_dim,
                hidden_dim=end_to_end_projector_hidden_dim,
                dropout=end_to_end_projector_dropout,
                chunk_size=end_to_end_matrix_chunk_size,
                candidate_mask=end_to_end_candidate_mask,
            )
            if self.end_to_end_transition_device is not None:
                self.end_to_end_transition_generator = self.end_to_end_transition_generator.to(self.end_to_end_transition_device)
        self.fusion_type = fusion_type
        self.logit_prior_layer = None
        self.dynamic_fusion_transition_layer = None
        self.dynamic_vector_fusion_transition_layer = None
        self.sample_adaptive_fusion_transition_layer = None
        self.confidence_aware_fusion_transition_layer = None
        
        if transition_T is not None and transition_mode == 'feature_fusion':
            if fusion_type == 'soft_position':
                self.soft_position_fusion_layer = SoftPositionFusionLayer(
                    code_num, self.lstm_hidden_size, max_seq_len=max_seq_len,
                    base_alpha=llm_alpha, position_factor=0.2,
                    confidence_adjust_range=0.2, use_confidence=True,
                    dropout=dropout_rate
                )
                self.soft_position_fusion_layer.set_transition_matrices(transition_T, transition_T_llm if use_llm_transition else None)
                self.fusion_transition_layer = None
                self.simple_transition_layer = None
            elif use_llm_transition and (transition_T_llm is not None or self.use_end_to_end_llm_transition):
                if transition_fusion_variant == 'dynamic_residual':
                    self.dynamic_fusion_transition_layer = DynamicResidualFusionTransitionLayer(
                        code_num, self.lstm_hidden_size, init_alpha=llm_alpha, init_beta=0.5, dropout=dropout_rate
                    )
                    self.dynamic_fusion_transition_layer.set_transition_matrices(transition_T, transition_T_llm)
                    self.fusion_transition_layer = None
                    self.dynamic_vector_fusion_transition_layer = None
                    self.sample_adaptive_fusion_transition_layer = None
                    self.confidence_aware_fusion_transition_layer = None
                elif transition_fusion_variant == 'dynamic_vector_residual':
                    self.dynamic_vector_fusion_transition_layer = DynamicVectorResidualFusionTransitionLayer(
                        code_num, self.lstm_hidden_size, init_alpha=llm_alpha, init_beta=0.5, dropout=dropout_rate
                    )
                    self.dynamic_vector_fusion_transition_layer.set_transition_matrices(transition_T, transition_T_llm)
                    self.fusion_transition_layer = None
                    self.dynamic_fusion_transition_layer = None
                    self.sample_adaptive_fusion_transition_layer = None
                    self.confidence_aware_fusion_transition_layer = None
                elif transition_fusion_variant == 'sample_adaptive':
                    self.sample_adaptive_fusion_transition_layer = SampleAdaptiveFusionTransitionLayer(
                        code_num,
                        self.lstm_hidden_size,
                        init_alpha=llm_alpha,
                        init_beta=0.5,
                        dropout=dropout_rate,
                        gate_mode=transition_gate_mode,
                        shared_projection=shared_transition_projection,
                    )
                    self.sample_adaptive_fusion_transition_layer.set_transition_matrices(
                        transition_T,
                        None if self.use_end_to_end_llm_transition else transition_T_llm
                    )
                    self.fusion_transition_layer = None
                    self.dynamic_fusion_transition_layer = None
                    self.dynamic_vector_fusion_transition_layer = None
                    self.confidence_aware_fusion_transition_layer = None
                elif transition_fusion_variant == 'confidence_aware':
                    self.confidence_aware_fusion_transition_layer = ConfidenceAwareFusionTransitionLayer(
                        code_num, self.lstm_hidden_size, init_alpha=llm_alpha, init_beta=0.5, dropout=dropout_rate
                    )
                    self.confidence_aware_fusion_transition_layer.set_transition_matrices(transition_T, transition_T_llm)
                    self.fusion_transition_layer = None
                    self.dynamic_fusion_transition_layer = None
                    self.dynamic_vector_fusion_transition_layer = None
                    self.sample_adaptive_fusion_transition_layer = None
                else:
                    self.fusion_transition_layer = FusionTransitionLayer(
                        code_num, self.lstm_hidden_size, init_scale=1.0, 
                        fusion_mode='learnable' if use_llm_transition else 'fixed',
                        init_alpha=llm_alpha,
                        dropout=dropout_rate
                    )
                    self.fusion_transition_layer.set_transition_matrices(transition_T, transition_T_llm)
                    self.dynamic_fusion_transition_layer = None
                    self.dynamic_vector_fusion_transition_layer = None
                    self.sample_adaptive_fusion_transition_layer = None
                    self.confidence_aware_fusion_transition_layer = None
                self.simple_transition_layer = None
                self.soft_position_fusion_layer = None
            else:
                self.simple_transition_layer = SimpleTransitionLayer(code_num, self.lstm_hidden_size, init_scale=1.0, dropout=dropout_rate)
                self.fusion_transition_layer = None
                self.soft_position_fusion_layer = None
                self.dynamic_fusion_transition_layer = None
                self.dynamic_vector_fusion_transition_layer = None
                self.sample_adaptive_fusion_transition_layer = None
                self.confidence_aware_fusion_transition_layer = None
        else:
            self.simple_transition_layer = None
            self.fusion_transition_layer = None
            self.soft_position_fusion_layer = None
            self.dynamic_fusion_transition_layer = None
            self.dynamic_vector_fusion_transition_layer = None
            self.sample_adaptive_fusion_transition_layer = None
            self.confidence_aware_fusion_transition_layer = None

        if transition_T is not None and transition_mode == 'prior_bias':
            self.logit_prior_layer = LogitPriorLayer(init_lambda=prior_lambda, learnable=True)
            if use_llm_transition and transition_T_llm is not None:
                self.logit_prior_layer.set_transition_matrix(transition_T_llm)
            else:
                self.logit_prior_layer.set_transition_matrix(transition_T)
        
        self.use_transition = (
            (transition_T is not None) and
            (use_transition or use_llm_transition or fusion_type == 'soft_position' or transition_mode == 'prior_bias')
        )
        self.window_weights = nn.Parameter(torch.ones(3) / 3)

    def _masked_window_sum(self, seq_output, lens, window_size=None, offset=0):
        batch_size, seq_len, hidden_dim = seq_output.shape
        device = seq_output.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        valid_mask = positions < lens.unsqueeze(1)
        end_pos = torch.clamp(lens - offset, min=0)
        if window_size is None:
            start_pos = torch.zeros_like(end_pos)
        else:
            start_pos = torch.clamp(end_pos - window_size, min=0)

        window_mask = (positions >= start_pos.unsqueeze(1)) & (positions < end_pos.unsqueeze(1))
        final_mask = (valid_mask & window_mask).unsqueeze(-1).to(seq_output.dtype)
        return (seq_output * final_mask).sum(dim=1)

    def _context_branch_from_visits(self, code_x, lens, codes):
        output2 = []
        for code_x_i, len_i, code in zip(code_x, lens, codes):
            if len_i <= 0:
                output_i = torch.zeros(code_x_i.shape[1], device=code_x_i.device)
            else:
                visits = code_x_i[:len_i].float()
                context = code.unsqueeze(0).expand(len_i, -1)
                if self.context_branch == 'interaction_mlp':
                    interaction = torch.cat(
                        [visits, context, visits * context, torch.abs(visits - context)],
                        dim=-1
                    )
                    output_i = self.context_interaction(interaction)
                else:
                    output_i = self.attention1(visits, context, context)
                output_i = output_i.mean(dim=0)
            output2.append(output_i)

        if len(output2) == 0:
            batch_size = code_x.shape[0]
            output2 = torch.zeros(batch_size, code_x.shape[2], device=code_x.device)
        else:
            output2 = torch.vstack(output2)
        return output2

    def _kv_aware_context_branch(self, seq_output, lens, codes):
        output2 = []
        for seq_i, len_i, code in zip(seq_output, lens, codes):
            if len_i <= 0:
                output_i = torch.zeros(self.kv_context_query_projection.out_features, device=seq_output.device)
            else:
                h_label = self.kv_context_query_projection(seq_i[:len_i].float())
                context = code.unsqueeze(0).expand(len_i, -1)
                kv = self.kv_context_interaction(torch.cat([h_label, context, h_label * context], dim=-1))
                output_i = self.kv_context_attention(h_label, kv, kv).mean(dim=0)
            output2.append(output_i)

        if len(output2) == 0:
            batch_size = seq_output.shape[0]
            return torch.zeros(batch_size, self.kv_context_query_projection.out_features, device=seq_output.device)
        return torch.vstack(output2)

    def _retain_query_context_branch(self, seq_output, lens, codes):
        output2 = []
        scale = math.sqrt(float(self.lstm_hidden_size))
        for seq_i, len_i, code in zip(seq_output, lens, codes):
            if len_i <= 0:
                output_i = torch.zeros(self.retain_query_output_projection[0].out_features, device=seq_output.device)
            else:
                valid_seq = seq_i[:len_i].float()
                query = self.retain_query_projection(code.float()).unsqueeze(0)
                key = self.retain_query_key_projection(valid_seq)
                value = self.retain_query_value_projection(valid_seq)
                score = torch.matmul(query, key.transpose(0, 1)) / scale
                weight = torch.softmax(score, dim=-1)
                pooled = torch.matmul(weight, value).squeeze(0)
                output_i = self.retain_query_output_projection(pooled)
            output2.append(output_i)

        if len(output2) == 0:
            batch_size = seq_output.shape[0]
            return torch.zeros(batch_size, self.retain_query_output_projection[0].out_features, device=seq_output.device)
        return torch.vstack(output2)

    def forward(self, code_x, lens):
        codes = self.retain(code_x, lens)
        output2 = None
        if self.context_branch not in ['kv_aware_attention', 'retain_query_attention']:
            output2 = self._context_branch_from_visits(code_x, lens, codes)
            if self.use_stream_feature_norm:
                output2 = self.output2_norm(output2) * self.output2_scale
        code_x = code_x.float()
        seq_len = code_x.size(1)
        valid_mask = torch.arange(seq_len, device=code_x.device).unsqueeze(0) < lens.unsqueeze(1)
        
        if self.use_tcn:
            output = self.tcn(code_x)
            output = output * valid_mask.unsqueeze(-1).to(output.dtype)
        else:
            if self.use_packed_sequence:
                packed_input = nn.utils.rnn.pack_padded_sequence(
                    code_x,
                    lens.detach().cpu(),
                    batch_first=True,
                    enforce_sorted=False
                )
                packed_output, _ = self.lstm(packed_input)
                output, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_output,
                    batch_first=True,
                    total_length=seq_len
                )
            else:
                lstm_out = self.lstm(code_x)
                output = lstm_out[0]
        
        output = self.layernorm1(output)
        if self.use_masked_attention:
            output = output * valid_mask.unsqueeze(-1).to(output.dtype)
            output = self.selfattention(output, mask=valid_mask)
        else:
            output = self.selfattention(output)

        self.last_transition_extras = None
        batch_indices = torch.arange(code_x.size(0), device=code_x.device)
        last_visit_indices = torch.clamp(lens - 1, min=0)
        last_visit_codes = code_x[batch_indices, last_visit_indices]
        dynamic_transition_T_llm = None
        if self.use_end_to_end_llm_transition and self.end_to_end_transition_generator is not None:
            dynamic_transition_T_llm = self.end_to_end_transition_generator()
            dynamic_transition_T_llm = dynamic_transition_T_llm.to(output.device)
        if self.use_transition:
            if self.fusion_type == 'soft_position' and self.soft_position_fusion_layer is not None:
                output = self.soft_position_fusion_layer(output)
            elif self.use_llm_transition and self.confidence_aware_fusion_transition_layer is not None:
                output, self.last_transition_extras = self.confidence_aware_fusion_transition_layer(
                    output, mask=valid_mask, last_visit_codes=last_visit_codes, return_extras=True
                )
            elif self.use_llm_transition and self.sample_adaptive_fusion_transition_layer is not None:
                output, self.last_transition_extras = self.sample_adaptive_fusion_transition_layer(
                    output,
                    mask=valid_mask,
                    transition_T_llm=dynamic_transition_T_llm,
                    return_extras=True
                )
            elif self.use_llm_transition and self.dynamic_vector_fusion_transition_layer is not None:
                output, self.last_transition_extras = self.dynamic_vector_fusion_transition_layer(
                    output, mask=valid_mask, return_extras=True
                )
            elif self.use_llm_transition and self.dynamic_fusion_transition_layer is not None:
                output, self.last_transition_extras = self.dynamic_fusion_transition_layer(
                    output, mask=valid_mask, return_extras=True
                )
            elif self.use_llm_transition and self.fusion_transition_layer is not None:
                output = self.fusion_transition_layer(output)
            elif self.simple_transition_layer is not None:
                output = self.simple_transition_layer(output, self.transition_T)
            output = self.layernorm2(output)
            output = output * valid_mask.unsqueeze(-1).to(output.dtype)

        if self.context_branch == 'kv_aware_attention':
            output2 = self._kv_aware_context_branch(output, lens, codes)
            if self.use_stream_feature_norm:
                output2 = self.output2_norm(output2) * self.output2_scale
        elif self.context_branch == 'retain_query_attention':
            output2 = self._retain_query_context_branch(output, lens, codes)
            if self.use_stream_feature_norm:
                output2 = self.output2_norm(output2) * self.output2_scale

        if self.multi_scale_agg is not None:
            output1 = self.multi_scale_agg(output, lens)
        else:
            output1 = []
            weights = torch.softmax(self.window_weights, dim=0)
            short_sum = self._masked_window_sum(output, lens, window_size=3, offset=0)
            medium_sum = self._masked_window_sum(output, lens, window_size=10, offset=0)
            long_sum = self._masked_window_sum(output, lens, window_size=None, offset=0)
            output1 = weights[0] * short_sum + weights[1] * medium_sum + weights[2] * long_sum
            
            if output1.numel() == 0:
                output1 = torch.zeros(len(output), output.shape[2], device=output.device)
                output1 = self.time_layer4(output1)
            else:
                output1 = self.time_layer4(output1)
        if self.use_stream_feature_norm:
            output1 = self.output1_norm(output1) * self.output1_scale
        self.last_structure_extras = {}
        if self.dual_stream_alignment is not None:
            output1, output2, alignment_extras = self.dual_stream_alignment(output1, output2, return_extras=True)
            self.last_structure_extras['dual_stream_alignment'] = alignment_extras
        output3 = []
        self.last_tcrf_extras = None
        if self.tcrf is not None:
            tcrf_result = self.tcrf(output1, output2, lens)
            if isinstance(tcrf_result, tuple):
                output3, self.last_tcrf_extras = tcrf_result
            else:
                output3 = tcrf_result
        else:
            for i in range(len(output2)):
                output31 = output1[i] * output2[i]
                output3.append(output31)
            
            if len(output3) == 0:
                output3 = torch.zeros(len(output2), output1.shape[1] if len(output1.shape) > 1 else output1.shape[0], device=output1.device)
            else:
                output3 = torch.vstack(output3)
        if self.residual_interaction_fusion is not None:
            output3, interaction_extras = self.residual_interaction_fusion(output1, output2, base_fused=output3, return_extras=True)
            self.last_structure_extras['residual_interaction_fusion'] = interaction_extras
        if self.label_wise_gate is not None:
            output3, gate_extras = self.label_wise_gate(output1, output2, base_fused=output3, return_extras=True)
            self.last_structure_extras['label_wise_gate'] = gate_extras
        if self.gated_residual_interaction_fusion is not None:
            output3, gated_rif_extras = self.gated_residual_interaction_fusion(
                output1, output2, base_fused=output3, return_extras=True
            )
            self.last_structure_extras['gated_residual_interaction_fusion'] = gated_rif_extras
        if self.cross_attention_fusion is not None:
            output3, cross_attention_extras = self.cross_attention_fusion(
                output1, output2, base_fused=output3, return_extras=True
            )
            self.last_structure_extras['cross_attention_fusion'] = cross_attention_extras
        output = self.classifier(output3)
        if self.label_graph_head is not None:
            output, graph_extras = self.label_graph_head(output, return_extras=True)
            self.last_structure_extras['label_graph_head'] = graph_extras
        if self.transition_mode == 'prior_bias' and self.logit_prior_layer is not None:
            output = self.logit_prior_layer(output, last_visit_codes)
        return output

