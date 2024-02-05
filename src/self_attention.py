""" MultiAttention Module """
# coding: utf-8
# !/usr/bin/env python

import torch
import torch.nn as nn


class MultiAttention(nn.Module):
    """SelfAttention module

    """

    def __init__(self, input_size, num_heads):
        super(MultiAttention, self).__init__()
        #
        self.input_size = input_size
        self.num_heads = num_heads
        self.head_size = input_size // num_heads
        # check num_heads
        assert_err_msg = (f"input_size ({input_size}) "
                          f"must be an integer multiple "
                          f"of num_heads ({num_heads})")

        assert (num_heads * self.head_size == input_size), assert_err_msg
        # layers
        self.value_layer = nn.Linear(self.head_size,
                                     self.head_size,
                                     bias=False)
        self.key_layer = nn.Linear(self.head_size, self.head_size, bias=False)
        self.query_layer = nn.Linear(self.head_size,
                                     self.head_size,
                                     bias=False)
        self.out_layer = nn.Linear(self.input_size,
                                   self.input_size,
                                   bias=False)
        # F
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax()

    def forward(self, keys, values, queries, mask):

        # query: (batch_size, seq_len, embed_size)
        batch_size = queries.shape[0]
        value_len = values.shape[1]  # key_len = value_len always true
        key_len = keys.shape[1]  # source sentence length
        query_len = queries.shape[1]  # target sentence length
        # split input_size into self.num_heads chunks
        # reshaped dims: (batch_size, value_len, num_heads, head_size)
        values = values.reshape(batch_size, value_len, self.num_heads,
                                self.head_size)
        keys = keys.reshape(batch_size, key_len, self.num_heads,
                            self.head_size)
        queries = queries.reshape(batch_size, query_len, self.num_heads,
                                  self.head_size)

        keys = self.value_layer(keys)
        values = self.value_layer(values)
        queries = self.value_layer(queries)
        # dot prod
        # z dims:      (batch_size, num_heads, query_len, key_len)

        z = torch.einsum("bqnh, bknh -> bnqk", [queries, keys])
        # apply mask, mask is used to avoid seeing into the future
        if mask is not None:
            z = z.masked_fill(mask == 0, float(-1e-20))
        # attention  a(q, k , v)= softmax((Q K.T) / sqrt(d)) V
        sqrt_dim = self.input_size**(1 / 2)
        attention = torch.softmax(z / sqrt_dim,
                                  dim=3)  # att across the src sentence dim=3
        # attention dims: (batch_size, num_heads, query_len, key_len)
        # values dims: (batch_size, value_len, num_heads, head_size)
        # out dims:(batch_size, query_len, num_heads, head_size)
        out = torch.einsum("bnqk, bknh -> bqnh", [attention, values])
        # concatenate multi-head outputs
        out = out.reshape(batch_size, query_len,
                          self.num_heads * self.head_size)
        out = self.out_layer(out)
        return out
