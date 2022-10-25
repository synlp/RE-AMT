import torch
from torch import nn

class KVMN(nn.Module):
    def __init__(self, hidden_size, key_size, val_size, b_add=True, b_both_kv=False, b_self_prob=False):
        super(KVMN, self).__init__()
        self.temper = hidden_size ** 0.5
        self.key_embedding = nn.Embedding(key_size, hidden_size)
        self.val_embedding = nn.Embedding(val_size, hidden_size)
        self.b_add = b_add
        self.b_both_kv = b_both_kv
        self.b_self_prob = b_self_prob

    def forward(self, hidden_state, key_seq, value_matrix, key_mask_matrix, output_kvmn_weight=False):
        embedding_key = self.key_embedding(key_seq)
        embedding_val = self.val_embedding(value_matrix)

        if self.b_self_prob is True:
            key_seq_h = hidden_state.permute(0, 2, 1)[:,:,:key_seq.shape[-1]]
            if key_seq_h.shape[-1] < key_seq.shape[-1]:
                key_seq_h = torch.cat([key_seq_h, torch.zeros([key_seq_h.shape[0], key_seq_h.shape[1], (key_seq.shape[-1] - key_seq_h.shape[2])])], dim=-1)
            u = torch.matmul(hidden_state.float(), key_seq_h.float()) / self.temper
        else:
            key_seq_h = embedding_key.permute(0, 2, 1)
            u = torch.matmul(hidden_state.float(), key_seq_h.float()) / self.temper

        tmp_key_mask_matrix = torch.clamp(key_mask_matrix, 0, 1)

        exp_u = torch.exp(u)
        delta_exp_u = torch.mul(exp_u, tmp_key_mask_matrix.float())

        sum_delta_exp_u = torch.stack([torch.sum(delta_exp_u, 2)] * delta_exp_u.shape[2], 2)

        #batch_size, max_seq_len, max_key_len
        p = torch.div(delta_exp_u, sum_delta_exp_u + 1e-10)

        embedding_val = embedding_val.permute(3, 0, 1, 2)
        o = torch.mul(p, embedding_val.float()).type_as(hidden_state)

        o = o.permute(1, 2, 3, 0)#batch_size, max_seq_len, max_key_len, hidden_size
        o = torch.sum(o, 2)

        if self.b_both_kv is True:
            #batch_size, max_key_len, hidden_size
            # embedding_key = embedding_key.permute(0, 2, 1)
            embedding_key_matrix = torch.stack([embedding_key] * p.shape[1], 1).permute(3, 0, 1, 2)
            ko = torch.mul(p, embedding_key_matrix.float()).type_as(hidden_state).permute(1, 2, 3, 0)
            ko = torch.sum(ko, 2)
            o = torch.add(o, ko)

        if self.b_add is True:
            o = torch.add(o, hidden_state)

        if output_kvmn_weight is True:
            return o, p
        return o