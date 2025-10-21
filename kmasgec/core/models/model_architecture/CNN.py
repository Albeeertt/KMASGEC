import torch.nn as nn
import torch
import math




class CNN(nn.Module):

    def __init__(self, in_channels: int, kernel_size: int, num_classes: int, seq_len: int, vocab_size: int, padding_value: int,  dropout: float):

        super().__init__()

        out_first = 64
        out_second = 128
        out_third = 256
        default_stride = 1
        change_stride = 2
        p = kernel_size // 2

        self.emb = nn.Embedding(vocab_size, in_channels, padding_idx= padding_value)
        
        self.layers = nn.Sequential(
                    nn.Conv1d(in_channels, out_first, kernel_size, stride= default_stride, padding = p),
                    nn.GELU(),
                    nn.Dropout(dropout),

                    nn.Conv1d(out_first, out_first, kernel_size, stride= default_stride, padding = p),
                    nn.GELU(),
                    nn.Dropout(dropout),

                    nn.Conv1d(out_first, out_first, kernel_size, stride= default_stride, padding = p),
                    nn.GELU(),
                    nn.Dropout(dropout),
        

                    nn.Conv1d(out_first, out_second, kernel_size, stride= change_stride, padding = p),
                    nn.GELU(),
                    nn.Dropout(dropout),

                    nn.Conv1d(out_second, out_second, kernel_size, stride= default_stride, padding = p),
                    nn.GELU(),
                    nn.Dropout(dropout),

                    nn.Conv1d(out_second, out_second, kernel_size, stride= default_stride, padding = p),
                    nn.GELU(),
                    nn.Dropout(dropout),


                    nn.Conv1d(out_second, out_third, kernel_size, stride= change_stride, padding = p),
                    nn.GELU(),
                    nn.Dropout(dropout),

                    nn.Conv1d(out_third, out_third, kernel_size, stride= default_stride, padding = p),
                    nn.GELU(),
                    nn.Dropout(dropout),

                    nn.Conv1d(out_third, out_third, kernel_size, stride= default_stride, padding = p),
                    nn.GELU(),
                    nn.Dropout(dropout)
                    )

        self.avgPool = nn.AdaptiveAvgPool1d(1)
        self.maxPool = nn.AdaptiveMaxPool1d(1)
        in_dim = 2 * out_third

        self.ffn = nn.Sequential(
                    nn.Linear(in_dim, 1024),
                    nn.GELU(),
                    nn.Dropout(dropout),

                    nn.Linear(1024, 256),
                    nn.GELU(),
                    nn.Dropout(dropout),

                    nn.Linear(256, num_classes)
                    )

    def forward(self, x, attention_mask=None):
        x = self.emb(x)
        x = x.transpose(1, 2)
        x = self.layers(x)
        xa = self.avgPool(x).squeeze(-1)
        xm = self.maxPool(x).squeeze(-1)
        x  = torch.cat([xa, xm], dim=1)
        outputs = self.ffn(x)
        return outputs
