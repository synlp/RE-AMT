import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, negative_slope=0.01, dropout_prob=0.1, bias=True, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.linear_W = nn.Linear(in_features, out_features, bias=bias)
        self.linear_self_W = nn.Linear(out_features*2, 1, bias=bias)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_prob)
        self.concat = concat

        self.reset_linear_parameters(self.linear_W, bias=bias)
        self.reset_linear_parameters(self.linear_self_W, bias=bias)

    def reset_linear_parameters(self, w, bias=True):
        init.xavier_normal_(w.weight)
        if bias is True:
            fan_in, _ = init._calculate_fan_in_and_fan_out(w.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(w.bias, -bound, bound)

    def self_attention(self, wh):
        batch_size, seq_len, hidden_size = wh.size()
        wh_repeated_in_chunks = wh.repeat_interleave(seq_len, dim=1)
        wh_repeated_alternating = wh.repeat(1, seq_len, 1)
        wh_combination_matrix = torch.cat([wh_repeated_in_chunks, wh_repeated_alternating], dim=2)
        wh_combination_matrix = wh_combination_matrix.view(batch_size, seq_len, seq_len,hidden_size*2)
        return self.leakyrelu(self.linear_self_W(wh_combination_matrix).squeeze(-1))

    def masked_attention(self, att, adj_matrix):
        return torch.where(adj_matrix>0, att, -9e15*torch.ones_like(att))

    def forward(self, inputs, adj_matrix):
        wh = self.linear_W(inputs)
        attention_scores = self.self_attention(wh)
        attention_scores = self.masked_attention(attention_scores, adj_matrix)
        attention_scores = self.softmax(attention_scores)
        attention_scores = self.dropout(attention_scores)
        return F.elu(torch.matmul(attention_scores, wh))


class GAT(nn.Module):
    def __init__(self, hidden_size, att_header, negative_slope=0.01, dropout_prob=0.1, bias=True):
        super(GAT, self).__init__()
        gat_hidden_size = hidden_size // att_header
        self.dropout = nn.Dropout(dropout_prob)
        self.GATLayers = nn.ModuleList(GraphAttentionLayer(hidden_size, gat_hidden_size, negative_slope=negative_slope,
                                                           dropout_prob=dropout_prob, bias=bias)
                                       for _ in range(att_header))
        self.out_att = GraphAttentionLayer(gat_hidden_size * att_header, hidden_size, negative_slope=negative_slope,
                                                           dropout_prob=dropout_prob, bias=bias)

    def forward(self, inputs, adj_matrix):
        outputs = torch.cat([gat(inputs, adj_matrix) for gat in self.GATLayers], dim=-1)
        outputs = self.dropout(outputs)
        return self.out_att(outputs, adj_matrix)