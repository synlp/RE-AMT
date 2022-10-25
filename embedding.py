import torch
import torch.nn as nn
from pytorch_transformers import BertModel,BertConfig

class DiaEmbedding(nn.Module):
    def __init__(self, bert_path, embedding_type, hidden_size, bert_config=None, using_position=False, max_position_embeddings=512):
        super(DiaEmbedding, self).__init__()
        self.embedding_type = embedding_type
        self.using_position = using_position
        if bert_config is None:
            bert_config = BertConfig.from_pretrained(bert_path)
        self.bert_config = bert_config
        if embedding_type == "embedding":
            self.embedding = nn.Embedding(bert_config.vocab_size, hidden_size)
            if using_position is True:
                self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
            print("Using embedding from scratch")
        elif embedding_type == "bert_embedding":
            self.bert = BertModel.from_pretrained(bert_path)
            self.embedding = self.bert.embeddings
            print("Using BERT embedding")
        else:
            self.bert = BertModel.from_pretrained(bert_path)
            print("Using BERT")

    def get_embedding(self, input_ids, token_type_ids=None):
        if self.embedding_type ==  "embedding":
            output = self.embedding(input_ids)
            if self.using_position is True:
                seq_length = input_ids.size(1)
                position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
                position_embeddings = self.position_embeddings(position_ids)

                output = output + position_embeddings
        elif self.embedding_type == "bert_embedding":
            output = self.embedding(input_ids, token_type_ids)
        else:
            output, _ = self.bert(input_ids, token_type_ids)

        return output

    def forward(self, input_ids, token_type_ids=None, wordpiece_ids=None, c2w_map=None):
        char_embedding = self.get_embedding(input_ids, token_type_ids)
        if wordpiece_ids is not None and self.embedding_type in ["bert_embedding", "bert"]:
            word_embedding = self.get_embedding(wordpiece_ids, token_type_ids)
            batch_size, max_len, feat_dim = char_embedding.shape
            add_wordpiece_embedding = torch.stack(
                [torch.stack([word_embedding[i]] * max_len)[range(max_len), c2w_map[i]] for i in range(batch_size)])
            return char_embedding + add_wordpiece_embedding
        else:
            return char_embedding