import torch
import torch.nn as nn
from self_attention import MultiAttention


class TransformerModule(nn.Module):
    """Documentation for TransformerModule

    """

    def __init__(self, input_size, num_heads, forward_expansion,
                 dropout_ratio):
        super(TransformerModule, self).__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.forward_expansion = forward_expansion
        self.dropout_ratio = dropout_ratio
        # layers
        self.multi_attention = MultiAttention(self.input_size, self.num_heads)
        self.norm_layer_1 = nn.LayerNorm(self.input_size)
        self.ff_layer = nn.Sequential(
            nn.Linear(self.input_size,
                      self.forward_expansion * self.input_size),  #
            nn.ReLU(),  #
            nn.Linear(self.forward_expansion * self.input_size,
                      self.input_size)  #
        )
        self.norm_layer_2 = nn.LayerNorm(self.input_size)

        self.dropout = nn.Dropout(self.dropout_ratio)
        #
        # self.relu = nn.ReLU()

    def forward(self, keys, values, queries, mask):
        attention = self.multi_attention(keys, values, queries, mask)
        att_out = self.norm_layer_1(attention + queries)
        att_out = self.dropout(att_out)
        ff_out = self.ff_layer(att_out)
        out = self.norm_layer_2(att_out + ff_out)
        out = self.dropout(out)
        return out
