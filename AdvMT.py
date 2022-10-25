import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from loss import EntropyLoss

import math

from pytorch_transformers import BertModel,BertConfig

class BiLSTM(nn.Module):
    def __init__(self, in_feature, out_feature, num_layers=1, batch_first = True):
        super(BiLSTM, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=in_feature,
            hidden_size=out_feature,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=True,
            dropout=0.5
        )

    def rand_init_hidden(self, batch_size, device):
        return (torch.zeros(2 * self.num_layers, batch_size, self.out_feature).to(device),
                torch.zeros(2 * self.num_layers, batch_size, self.out_feature).to(device))

    def forward(self, input):
        batch_size, seq_len, hidden_size = input.shape
        hidden = self.rand_init_hidden(batch_size, input.device)
        output, hidden = self.lstm(input, hidden)
        return output.contiguous().view(batch_size, seq_len, self.out_feature * 2)

class Transformer(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=512, dropout=0.2, num_layers=1):
        super(Transformer, self).__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, input):
        output = self.transformer(input.transpose(0, 1))
        return output.transpose(0, 1)

class AdvMT(nn.Module):
    def __init__(self, tag_size=100, ne_tag_size=2, bert_path=None, embedding='embedding', encoder="transformer",
                 num_layers=1, criteria_size=2, multi_criteria=True, adversary=True, adv_coefficient=1, pooling="max_pooling"):
        super(AdvMT, self).__init__()
        self.tag_size = tag_size
        self.criteria_size = criteria_size
        self.num_layers = num_layers
        self.embedding_type = embedding
        self.encoder_type = encoder
        self.multi_criteria = multi_criteria
        self.adversary = adversary
        self.adv_coefficient = adv_coefficient
        bert_config = BertConfig.from_pretrained(bert_path)
        hidden_size = bert_config.hidden_size
        self.config = bert_config
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.pooling = pooling
        self.ne_tag_size = ne_tag_size
        if multi_criteria is False:
            self.criteria_size = 1
        if encoder == "transformer":
            self.bert = BertModel.from_pretrained(bert_path)
            self.private_encoder = Transformer(hidden_size, nhead=8, dim_feedforward=2048, dropout=0.1, num_layers=self.num_layers)
            self.private_encoder_ne = Transformer(hidden_size, nhead=8, dim_feedforward=2048, dropout=0.1, num_layers=self.num_layers)
            
            self.classifier_re = torch.nn.Linear(hidden_size*4, tag_size)
            self.classifier_ne = torch.nn.Linear(hidden_size, ne_tag_size)
            if self.adversary:
                self.classifier_at = torch.nn.Linear(hidden_size, 2)
        elif encoder == "BiLSTM":
            self.bert = BertModel.from_pretrained(bert_path)
            self.private_encoder = BiLSTM(hidden_size, hidden_size, num_layers=self.num_layers)
            self.private_encoder_ne = BiLSTM(hidden_size, hidden_size, num_layers=self.num_layers)
            
            self.classifier_re = torch.nn.Linear(hidden_size*7, tag_size)
            self.classifier_ne = torch.nn.Linear(hidden_size*2, ne_tag_size)
            if self.adversary:
                self.classifier_at = torch.nn.Linear(hidden_size, 2)
        else:
            raise Exception("Invalid encoder")
            
    def get_valid_seq_output(self, sequence_output, valid_ids):
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=sequence_output.dtype, device=sequence_output.device)
        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            valid_output[i][:temp.size(0)] = temp
        return valid_output
    
    def cons_valid_seq_output(self, sequence_output, valid_ids):
        batch_size, max_len = sequence_output.shape
        for i in range(batch_size):
            for j in range(max_len):
                if valid_ids[i][j] == 1:
                    sequence_output[i][j] = 1
        return sequence_output
        
    def extract_entity(self, sequence, e_mask):
        if self.pooling == "max_pooling":
            return self.max_pooling(sequence, e_mask)
        elif self.pooling == "avg_pooling":
            return self.avg_pooling(sequence, e_mask)
            
    def max_pooling(self, sequence, e_mask):
        entity_output = sequence * torch.stack([e_mask] * sequence.shape[-1], 2) + torch.stack(
            [(1.0 - e_mask) * -1000.0] * sequence.shape[-1], 2)
        entity_output = torch.max(entity_output, -2)[0]
        return entity_output.type_as(sequence)

    def avg_pooling(self, sequence, e_mask):
        extended_e_mask = e_mask.unsqueeze(1)
        extended_e_mask = torch.bmm(
            extended_e_mask.float(), sequence.float()).squeeze(1)
        entity_output = extended_e_mask.float() / (e_mask != 0).sum(dim=1).unsqueeze(1)
        return entity_output.type_as(sequence)

    def forward(self, input_ids, criteria_index, token_type_ids=None, attention_mask=None, labels=None, labels_NE=None, e1_mask=None, e2_mask=None,
                valid_ids=None, b_use_valid_filter=False):
        if criteria_index not in [0, 1]:
            raise Exception("criteria_index Invalid")

        sequence_output, pooled_output = self.bert(input_ids, token_type_ids)
        shared_output = sequence_output
        private_output = self.private_encoder(sequence_output)

        if valid_ids is not None:
            shared_output = self.get_valid_seq_output(shared_output, valid_ids)
            private_output = self.get_valid_seq_output(private_output, valid_ids)
        
        tmp_shared_output = shared_output
        
        e1_h_p = self.extract_entity(private_output, e1_mask)
        e2_h_p = self.extract_entity(private_output, e2_mask)
        
        if self.pooling == "max_pooling":
            shared_output = torch.max(shared_output, 1)[0]
            private_output = torch.max(private_output, 1)[0]
        elif self.pooling == "avg_pooling":
            shared_output = torch.mean(shared_output, 1)
            private_output = torch.mean(private_output, 1)                                                  
        private_output = torch.cat([private_output, e1_h_p, e2_h_p], dim=-1)

        sequence_output = torch.cat([shared_output, private_output], dim=-1)
        sequence_output = self.dropout(sequence_output)

        logits_re = self.classifier_re(sequence_output)

        if criteria_index:
            batch_size, max_len = e1_mask.shape

            if self.adversary:
                logits_at = self.classifier_at(tmp_shared_output)
            
            tmp_shared_output = self.private_encoder_ne(tmp_shared_output)
            logits_ne = self.classifier_ne(tmp_shared_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss_re = loss_fct(logits_re.view(-1, self.tag_size), labels.view(-1))
            if criteria_index:
                loss_ne = loss_fct(logits_ne.view(-1, self.ne_tag_size), labels_NE.view(-1))
                ne_results = torch.argmax(F.log_softmax(logits_ne, dim=2), dim=2)
                if self.adversary:
                    comparison_result = (labels_NE == ne_results)
                    comparison_result = comparison_result.long()
                    loss_at = loss_fct(logits_at.view(-1, 2), comparison_result.view(-1))
                    return logits_re, loss_re, loss_ne, loss_at
                return logits_re, loss_re, loss_ne, 1
            return logits_re, loss_re, 0, 0
        else:
            tmp_shared_output = self.private_encoder_ne(tmp_shared_output)
            logits_ne = self.classifier_ne(tmp_shared_output)
            tag_seq_ne = torch.argmax(logits_ne, -1)
            return logits_re, tag_seq_ne
        