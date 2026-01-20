import torch.nn as nn
import torch
import math

# BigBird layer-by-layer imports
from transformers.models.big_bird.modeling_big_bird import (
    BigBirdConfig,
    BigBirdSelfAttention
)

# Reformer layer-by-layer imports
from transformers.models.reformer.modeling_reformer import (
    ReformerConfig,
    ReformerAttention
)


class BigBirdBlock(nn.Module):
    """
    Bloque de BigBird: LayerNorm → Sparse Attention → FF → LayerNorm.
    """
    def __init__(self, config: BigBirdConfig):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = BigBirdSelfAttention(config)
        self.dropout1 = nn.Dropout(config.attention_probs_dropout_prob)
        # self.intermediate = BigBirdIntermediate(config)
        # self.output_dense = BigBirdOutput(config)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.feed_forward_size),
            nn.GELU(),
            nn.Linear(config.feed_forward_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )

    def forward(self, x, attention_mask=None):
        # Pre-norm + attention
        res = x
        x = self.layernorm1(x)
        x = self.attention(
            x,
            attention_mask=attention_mask
        )
        x = self.dropout1(x) + res
        # Pre-norm + FF
        res = x
        x = self.layernorm2(x)
        x = self.feed_forward(x)
        x = self.dropout2(x) + res
        return x


class BigBirdTransformer(nn.Module):
    """
    BigBird completo: embeddings + N BigBirdBlock + optional cabeza.
    """
    def __init__(self, config: BigBirdConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.Linear(4, config.hidden_size)
        self.blocks = nn.ModuleList([BigBirdBlock(config) for _ in range(config.num_hidden_layers)])
        self.classifier = nn.Linear(config.hidden_size, 4)
        self.lastLayer = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask=None):
        x = self.embeddings(input_ids)
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.classifier(x)
        x = self.lastLayer(x)
        return x


class ReformerBlock(nn.Module):
    """
    Bloque de Reformer: LayerNorm → LSH Attention → FF → LayerNorm.
    """
    def __init__(self, config: ReformerConfig):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = ReformerAttention(config)
        self.dropout1 = nn.Dropout(config.lsh_attention_probs_dropout_prob)
        self.dropout2 = nn.Dropout(config.lsh_attention_probs_dropout_prob)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
                # Feed-forward con activación mapeada manualmente

        # TODO: Es un feed forward distinto. No es el usual, a si que revisarlo.
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.feed_forward_size),
            nn.GELU(),
            nn.Linear(config.feed_forward_size, config.hidden_size)
        )

    def forward(self, x, attention_mask=None, head_mask=None):
        # Pre-norm + LSH Attention
        res = x
        x = self.layernorm1(x)
        x = self.attention(x, attention_mask=attention_mask, head_mask=head_mask)[0]
        x = self.dropout1(x) + res
        # Pre-norm + FF
        res = x
        x = self.layernorm2(x)
        x = self.feed_forward(x)
        x = self.dropout2(x) + res
        return x


class ReformerTransformer(nn.Module):
    """
    Reformer completo: embeddings + N ReformerBlock + optional cabeza.
    """
    def __init__(self, config: ReformerConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.num_embeddings, config.embedding_dim)
        self.firstDropout = nn.Dropout(config.lsh_attention_probs_dropout_prob)
        self.blocks = nn.ModuleList([ReformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.lastLinear = nn.Linear(config.hidden_size, config.shape_output)
        self.lastLayer = nn.Softmax(dim=-1)

    def forward(self, input_ids, probs, attention_mask=None, head_mask=None):
        embedding_ids = self.embedding(input_ids)
        x = torch.cat((embedding_ids, probs), dim= -1)
        x = self.firstDropout(x)
        for block in self.blocks:
            x = block(x, attention_mask, head_mask)
        x = self.lastLinear(x)
        x = self.lastLayer(x)
        return x



class TransformerClassifier_twoHeads(nn.Module):
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
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        num_classes: int = 2,
        num_classes2: int = 2,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout_init = nn.Dropout(dropout)
        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        # Positional embeddings
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        # Classification head: takes CLS token representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        self.classifier2 = nn.Linear(embed_dim, num_classes2)

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
        # Positional indices
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        pos_emb = self.pos_embed(positions)
        # Combine
        x = token_emb + pos_emb  # [B, L, D]
        # Prepend CLS token
        cls_token = self.cls_token.expand(bsz, -1, -1)  # [B, 1, D]
        x = torch.cat([cls_token, x], dim=1)  # [B, L+1, D]
        # Transformer expects [S, B, D]
        x = x.transpose(0, 1)
        # Build extended mask for CLS + tokens
        if attention_mask is not None:
            cls_mask = torch.ones((bsz, 1), dtype=torch.bool, device=input_ids.device)
            extended_mask = torch.cat([cls_mask, attention_mask], dim=1)  # [B, L+1]
            # Transformer uses key_padding_mask of shape [B, S]
            key_padding_mask = ~extended_mask
        else:
            key_padding_mask = None
        # Encoder
        x = self.dropout_init(x)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)  # [L+1, B, D]
        # Extract CLS representation
        cls_repr = x[0]  # [B, D]
        # Classification
        cls_repr = self.dropout(cls_repr)
        logits = self.classifier(cls_repr)  # [B, C]
        logits2 = self.classifier2(cls_repr)
        return logits, logits2

class TransformerClassifier_pool(nn.Module):
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
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        num_classes: int = 2,
        # max_seq_len: int = 512,
        dropout: float = 0.1,
        pooling: str = "mean"
    ):
        super().__init__()

        self.convs = nn.Sequential(
                    nn.Conv1d(embed_dim, embed_dim,kernel_size = 3, stride= 1, padding = 1, dilation = 1, bias=False),
                    nn.GELU(),
                    nn.Conv1d(embed_dim, embed_dim,kernel_size = 3, stride= 1, padding = 1, dilation = 1, bias=False),
                    nn.GELU(),
                    nn.Conv1d(embed_dim, embed_dim,kernel_size = 3, stride= 1, padding = 1, dilation = 1, bias=False),
                    nn.GELU(),
                    nn.Conv1d(embed_dim, embed_dim,kernel_size = 3, stride= 1, padding = 1, dilation = 1,  bias=False),
                    nn.GELU()
                    )

        self.pooling = pooling
        self.embed_dim = embed_dim
        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        # Positional embeddings
        # self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        div = torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim)
        self.register_buffer("pos_div_term", torch.exp(div))  # [D/2]

        self.embed_ln = nn.LayerNorm(embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        rep_dim = embed_dim
        if self.pooling == "mean-max":
                rep_dim = 2 * embed_dim 

        self.classifier = nn.Sequential(
                        nn.LayerNorm(rep_dim),
                        nn.Linear(rep_dim, dim_feedforward),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(dim_feedforward, num_classes)
        )

        if pooling == "attn":
            self.attn_score = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.Tanh(),
                nn.Linear(embed_dim // 2, 1)  # (B, L, 1)
            )

        if pooling == "cls_token":
             self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
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
        token_emb = self.convs(token_emb.transpose(1, 2))
        token_emb = token_emb.transpose(1, 2)
        token_emb = token_emb.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
        if self.pooling == "cls_token":
            cls_tok = self.cls_token.expand(bsz, 1, -1)
            token_emb = torch.cat([cls_tok, token_emb], dim=1)
            cls_mask = torch.ones(bsz, 1, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
            seq_len += 1
        # Positional indices
        # positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        # pos_emb = self.pos_embed(positions)
        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.float32).unsqueeze(1)
        pe = torch.zeros(seq_len, token_emb.size(-1), device=input_ids.device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * self.pos_div_term)
        pe[:, 1::2] = torch.cos(positions * self.pos_div_term)
        x = token_emb + pe.unsqueeze(0)
        # x = (token_emb * math.sqrt(self.embed_dim)) + (pos_emb * attention_mask.unsqueeze(-1))
        x = self.embed_ln(x)
        # Transformer expects [S, B, D]
        x = x.transpose(0, 1)
        # Build extended mask for CLS + tokens
        if attention_mask is not None:
            valid = attention_mask
        else:
            valid = None
        key_padding_mask = (~valid)
        # Encoder
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)  # [L+1, B, D]
        if self.pooling == "mean":
            m = valid.transpose(0, 1).unsqueeze(-1).to(dtype=x.dtype)  # m: [L, B, 1]
            sum_x = (x * m).sum(dim=0)               # sumar sobre L (dim=0) -> [B, D]
            len_x = m.sum(dim=0).clamp(min=1e-6)     # [B, 1]
            rep = sum_x / len_x 
        elif self.pooling == "max":
            very_neg = torch.finfo(x.dtype).min
            v = valid.transpose(0, 1)
            x_masked = x.masked_fill(~v.unsqueeze(-1), very_neg)
            rep = x_masked.amax(dim=0)  
        elif self.pooling == "attn":
            v = valid.transpose(0, 1)
            scores = self.attn_score(x).squeeze(-1)
            scores = scores.masked_fill(~v, float("-inf"))
            w = torch.softmax(scores, dim=0).unsqueeze(-1)
            rep = (x * w).sum(dim=0)
        elif self.pooling == "mean-max":
            m = valid.transpose(0, 1).unsqueeze(-1).to(dtype=x.dtype)
            sum_x = (x * m).sum(dim=0)
            len_x = m.sum(dim=0).clamp(min=1e-6)
            mean_x = sum_x / len_x
            
            very_neg = torch.finfo(x.dtype).min
            v = valid.transpose(0, 1)
            x_masked = x.masked_fill(~v.unsqueeze(-1), very_neg)
            mx_x = x_masked.amax(dim=0)

            rep = torch.cat([mean_x, mx_x], dim=-1)
        elif self.pooling == "cls_token":
            rep = x[0]
        logits = self.classifier(rep)  # [B, C]
        return logits

class TransformerClassifier(nn.Module):
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
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        num_classes: int = 2,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.embed_dim = embed_dim
        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        
        # Positional embeddings
        # self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        div = torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim)
        self.register_buffer("pos_div_term", torch.exp(div))  # [D/2]

        # self.embed_ln = nn.LayerNorm(embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        # self.dropout = nn.Dropout(dropout)
        
        self.pre_head_ln = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )


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
        # CLS token
        cls_tok = self.cls_token.expand(bsz, 1, -1)
        token_emb = torch.cat([cls_tok, token_emb], dim=1)
        cls_mask = torch.ones(bsz, 1, device=attention_mask.device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
        seq_len += 1
        # Positional indices
        # positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        # pos_emb = self.pos_embed(positions)
        # x = (token_emb * math.sqrt(self.embed_dim)) + pos_emb
        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.float32).unsqueeze(1)
        pe = torch.zeros(seq_len, token_emb.size(-1), device=input_ids.device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * self.pos_div_term)
        pe[:, 1::2] = torch.cos(positions * self.pos_div_term)
        x = token_emb + pe.unsqueeze(0)
        # x = self.embed_ln(x)
        # Transformer expects [S, B, D]
        x = x.transpose(0, 1)
        # Build extended mask for CLS + tokens
        if attention_mask is not None:
            key_padding_mask = (~attention_mask)
        else:
            key_padding_mask = None
        # Encoder
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)  # [L+1, B, D]
        # Extract CLS representation
        cls_repr = x[0]  # [B, D]
        cls_repr = self.pre_head_ln(cls_repr)
        # Classification
        # cls_repr = self.dropout(cls_repr)
        logits = self.classifier(cls_repr)  # [B, C]
        return logits


class DNAlbert(nn.Module):
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

        # Transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )
        self.num_layers = num_layers

        # Classification head: takes CLS token representation
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        # self._init_weights()

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.BoolTensor = None) -> torch.Tensor:
        """
        Args:
            input_ids (LongTensor): [batch_size, seq_len] token indices.
            attention_mask (BoolTensor): [batch_size, seq_len] where True indicates tokens to attend.
        Returns:
            logits (Tensor): [batch_size, num_classes]
        """
        bsz, seq_len = input_ids.size()



        token_emb = self.token_embed(input_ids)  # [B, L, D]
        token_emb = self.add_more_embed(token_emb)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        pos_emb = self.pos_embed(positions)
        x = token_emb + pos_emb  # [B, L, D]
        # cls_token = self.cls_token.expand(bsz, -1, -1)  # [B, 1, D]
        # x = torch.cat([cls_token, x], dim=1)  # [B, L+1, D]
        x = x.transpose(0, 1)
        # if attention_mask is not None:
        #    cls_mask = torch.ones((bsz, 1), dtype=torch.bool, device=input_ids.device)
        #    extended_mask = torch.cat([cls_mask, attention_mask], dim=1)  # [B, L+1]
        #    key_padding_mask = ~extended_mask
        # else:
        #    key_padding_mask = None
        key_padding_mask = ~attention_mask
        x = self.dropout_init(x)
        for i in range(self.num_layers):
            x = self.encoder_layer(x, src_key_padding_mask=key_padding_mask)
        cls_repr = x[0, :, :]
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


class DNAlbert_2(nn.Module):
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

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head: takes CLS token representation
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        self._init_weights()

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
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        # Extract CLS representation
        cls_repr = x[0]  # [B, D]
        # Classification
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


