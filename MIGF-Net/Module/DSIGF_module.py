import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

""" transformer """
device = torch.device("cuda:0")

class DSIGF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder_ifb = IFB(self.config)

    def forward(self, embed1, embed2):

        enc_outputs, enc_self_attn_probs = self.encoder_ifb(embed1, embed2)

        return enc_outputs


""" encoder """
class IFB(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.d_hidn))
        self.dropout = nn.Dropout(self.config.emb_dropout)
        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_layer)])

    def forward(self, embed1, embed2):
        embed1 = embed1.permute((0, 2, 1))
        embed2 = embed2.permute((0, 2, 1))

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=embed1.shape[0])
        cls_embed1 = torch.cat((cls_tokens, embed1), dim=1)
        cls_embed1 = self.dropout(cls_embed1)

        cls_embed2 = torch.cat((cls_tokens, embed2), dim=1)
        cls_embed2 = self.dropout(cls_embed2)

        mask_inputs = torch.ones(embed1.shape[0], self.config.n_seq * self.config.n_seq + 1).to(device)
        attn_mask = get_attn_pad_mask(mask_inputs, mask_inputs, self.config.i_pad)

        attn_probs = []
        for layer in self.layers:

            cls_embed1, attn_prob = layer(cls_embed1, cls_embed2, attn_mask)
            attn_probs.append(attn_prob)

        return cls_embed1, attn_probs


""" encoder layer """
class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)

    def forward(self, cls_embed1, cls_embed2, attn_mask):

        att_outputs, attn_prob = self.self_attn(cls_embed1, cls_embed2, cls_embed2, attn_mask)
        att_outputs = self.layer_norm1(cls_embed1 + att_outputs)

        ffn_outputs = self.pos_ffn(att_outputs)
        ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)

        return ffn_outputs, attn_prob


""" attention pad mask """
def get_attn_pad_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(i_pad)
    pad_attn_mask = pad_attn_mask.unsqueeze(1).expand(batch_size, len_q, len_k)
    return pad_attn_mask


""" multi head attention """
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W_Q = nn.Conv1d(self.config.d_hidn, self.config.n_head * self.config.d_head, kernel_size=1)
        self.W_K = nn.Conv1d(self.config.d_hidn, self.config.n_head * self.config.d_head, kernel_size=1)
        self.W_V = nn.Conv1d(self.config.d_hidn, self.config.n_head * self.config.d_head, kernel_size=1)
        self.scaled_dot_attn = ScaledDotProductAttention(self.config)
        self.linear = nn.Conv1d(self.config.n_head * self.config.d_head, self.config.d_hidn, kernel_size=1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)
        Q = Q.permute(0, 2, 1)
        K = K.permute(0, 2, 1)
        V = V.permute(0, 2, 1)
        q_s = self.W_Q(Q).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.n_head, 1, 1)

        context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.config.n_head * self.config.d_head).permute(0, 2, 1)

        output = self.linear(context).permute(0, 2, 1)
        output = self.dropout(output)

        return output, attn_prob


""" scale dot product attention """
class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1 / (self.config.d_head ** 0.5)

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2))
        scores = scores.mul_(self.scale)
        scores.masked_fill_(attn_mask, -1e9)

        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)

        context = torch.matmul(attn_prob, V)

        return context, attn_prob


""" feed forward """
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(in_channels=self.config.d_hidn, out_channels=self.config.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.config.d_ff, out_channels=self.config.d_hidn, kernel_size=1)
        self.active = F.gelu
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):

        output = self.conv1(inputs.transpose(1, 2))
        output = self.active(output)

        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)

        return output