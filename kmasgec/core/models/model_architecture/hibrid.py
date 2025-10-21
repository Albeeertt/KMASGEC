import torch.nn as nn
import torch
import math
from transformers import LongformerConfig
from transformers.models.longformer.modeling_longformer import LongformerLayer as LongformerEncoderLayer
import torch.nn.functional as F

class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, value_drop):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(value_drop)

    def forward(self, query, key, value,  attention_mask):
        res = query
        query = self.norm(query)
        if attention_mask is not None:
            attn_output, _ = self.attn(query, key, value, key_padding_mask=attention_mask)
        else:
            attn_output, _ = self.attn(query, key, value)
        x = res + self.dropout(attn_output)
        return x



class FeedForward(nn.Module):
    def __init__(self, hidden_dim, feedForward_dim, value_drop):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, feedForward_dim)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(feedForward_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(value_drop)

    def forward(self, x):
        res = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return res + self.dropout(x)

class Transformer_encoder(nn.Module):

    def __init__(self, hidden_dim, ff_dim, num_heads, value_dropout: float = .1):
        super().__init__()
        self.attn = SelfAttentionBlock(hidden_dim, num_heads, value_dropout)
        self.ff = FeedForward(hidden_dim, ff_dim, value_dropout)

    def forward(self, query, key=None, value=None, attention_mask = None):

        if key is None:
            key = value = query

        x = self.attn(query, key, value, attention_mask)
        x = self.ff(x)
        return x


class HibridModel(nn.Module):

    def __init__(self, vocab_size, padding_idx, max_len_seq, num_classes, num_layers, embed_dim, hidden_dim, ff_dim, num_heads, dropout, attn_windows):
        super().__init__()
        config = LongformerConfig(
            attention_window=[attn_windows]*num_layers,
            hidden_size=hidden_dim,
            num_attention_heads=num_heads,
            intermediate_size=ff_dim
        )
        self.lsh_attention = nn.ModuleList([LongformerEncoderLayer(config) for _ in range(num_layers)])
        self.dropout_init = nn.Dropout(dropout)

        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        # self.add_more_embed = nn.Linear(embed_dim, hidden_dim)

        # Positional embeddings
        self.pos_embed = nn.Embedding(max_len_seq, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, num_classes)
        )
        self._init_weights()



    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids (LongTensor): [batch_size, seq_len] token indices.
            attention_mask (BoolTensor): [batch_size, seq_len] where True indicates tokens to attend.
        Returns:
            logits (Tensor): [batch_size, num_classes]
        """
        bsz, seq_len = input_ids.size()
        
        token_emb = self.token_embed(input_ids)  # [B, L, D]
        # token_emb = self.add_more_embed(token_emb)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        pos_emb = self.pos_embed(positions)
        
        x = token_emb + pos_emb

        x = self.dropout_init(x)
        for layer in self.lsh_attention:
            x = layer(x, attention_mask != 0, is_index_masked=(attention_mask == 0), is_index_global_attn=(attention_mask == 2))
            x = x[0]
        cls_repr = x[:, 0, :]  # [B, D]
        cls_repr = self.dropout(cls_repr)
        logits = self.classifier(cls_repr)  # [B, C]
        return logits


    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)




class Fusion(nn.Module):
    """
    Transformer-based classifier for sequence data.

    Args:
        vocab_size (int): size of the token vocabulary.
        embed_dim (int): dimension of token embeddings.
        num_heads (int): number of attention heads.
        num_layers (int): number of Transformer encoder layers.
        dim_feedforward (int): inner dimension of feedforward networks.
        num_classes (int): number of output classes.
        max_seq_len (int): maximum sequence length (for positional embeddings).
        dropout (float): dropout probability.
    """
    def __init__(
        self,
        vocab_size: int,
        padding_idx: int,
        embed_dim: int = 128,
        hidden_size: int = 768,
        num_heads: int = 8,
        num_layers: int = 12,
        dim_feedforward: int = 3072,
        num_classes: int = 2,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout_init = nn.Dropout(dropout)

        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.add_more_embed = nn.Linear(embed_dim, hidden_size)

        # Positional embeddings
        self.pos_embed = nn.Embedding(max_seq_len, hidden_size)

        self.listDecoder = nn.ModuleList()
        for _ in range(num_layers):
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu'
            )
            self.listDecoder.append(decoder_layer)


        # Classification head: takes CLS token representation
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
    
        convFirst = nn.Sequential(
                        nn.Conv1d(hidden_size, hidden_size, 3, padding=1),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Conv1d(hidden_size, hidden_size, 3, padding=1),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Conv1d(hidden_size, hidden_size, 3, padding=1)
                        )

        convSecond = nn.Sequential(
                        nn.Conv1d(hidden_size, hidden_size, 3, stride=2, padding=1),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Conv1d(hidden_size, hidden_size, 3, padding=1),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Conv1d(hidden_size, hidden_size, 3, padding=1)
                        )

        convThird = nn.Sequential(
                        nn.Conv1d(hidden_size, hidden_size, 3, stride=2, padding=1),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Conv1d(hidden_size, hidden_size, 3, padding=1),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Conv1d(hidden_size, hidden_size, 3, padding=1)
                        )

        self.listConv = nn.ModuleList()
        self.listConv.append(convFirst)
        self.listConv.append(convSecond)
        self.listConv.append(convThird)

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.BoolTensor = None) -> torch.Tensor:
        """
        Args:
            input_ids (LongTensor): [batch_size, seq_len] token indices.
            attention_mask (BoolTensor): [batch_size, seq_len] where True indicates tokens to attend.
        Returns:
            logits (Tensor): [batch_size, num_classes]
        """
        bsz, seq_len = input_ids.size()
        # Embedding
        token_emb = self.token_embed(input_ids)  # [B, L, D]
        token_emb = self.add_more_embed(token_emb)
        in_conv = token_emb
        # Positional indices
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        pos_emb = self.pos_embed(positions)
        #pos_emb = self.pos_embed[:, : x.size(1), :]
        # Combine
        x = token_emb + pos_emb  # [B, L, D]
        # Transformer expects [S, B, D]
        x = x.transpose(0, 1)
        # Encoder
        x = self.dropout_init(x)
        mask_memory = attention_mask

        for idx, (decoderLayer, convLayer) in enumerate(zip(self.listDecoder, self.listConv)):
            in_conv = in_conv * (~mask_memory).unsqueeze(-1).float()
            in_conv = in_conv.permute(0, 2, 1)
            in_conv = convLayer(in_conv)
            in_conv = in_conv.permute(2, 0, 1).contiguous()
            if idx != 0:
                mask_memory = downsample_pad_mask_conv(mask_memory, 3, 2, 1, 1)
            x = decoderLayer(tgt = x, memory = in_conv, tgt_key_padding_mask = attention_mask, memory_key_padding_mask = mask_memory)
            in_conv = in_conv.permute(1, 0, 2)

        # Extract CLS representation
        cls_repr = x[0]  # [B, D]
        # Classification
        cls_repr = self.dropout(cls_repr)
        logits = self.classifier(cls_repr)  # [B, C]
        return logits


def downsample_pad_mask_conv(pad_mask_in: torch.BoolTensor,
                             k: int, s: int = 1, p: int = 0, d: int = 1) -> torch.BoolTensor:
    """
    pad_mask_in : [N, L_in] con True = padding
    Devuelve    : [N, L_out] con True = padding tras Conv1d(k,s,p,d)
    Regla: salida es PAD si y solo si la ventana tenía 0 tokens válidos.
    """
    # 1) máscara de validez como flotante para conv
    valid = (~pad_mask_in).float().unsqueeze(1)        # [N, 1, L_in]

    # 2) kernel de unos para contar válidos en cada ventana
    weight = torch.ones(1, 1, k, device=valid.device, dtype=valid.dtype)

    # 3) cuenta de válidos por posición de salida (respeta stride, padding, dilation)
    count = F.conv1d(valid, weight, stride=s, padding=p, dilation=d)  # [N,1,L_out]

    # 4) salida es pad si la cuenta es 0
    pad_mask_out = (count.squeeze(1) == 0)             # [N, L_out], True = PAD
    return pad_mask_out
