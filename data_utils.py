import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_transformers import BertTokenizer
from get_knowledge import generate_knowledge_api


max_seq = 100


def get_graph(seq_len, feature_data, feature2id):
    ret = [[0] * seq_len for _ in range(seq_len)]
    for i, item in enumerate(feature_data):
        feature = item['dep_text']
        for j, dep_text in enumerate(feature):
            ret[i][j] = feature2id[dep_text]
    return torch.tensor(ret)

def get_simple_graph(seq_len, feature_data):
    ret = [[0] * seq_len for _ in range(seq_len)]
    for i, item in enumerate(feature_data):
        feature = item['range']
        for j, dep in enumerate(feature):
            ret[i][j] = dep
    return torch.tensor(ret)

def pad_graph(seq_len,graph):
    ret = [[0] * seq_len for _ in range(seq_len)]
    for i, item in enumerate(graph):
        for j in range(len(item)):
            ret[i][j] = item[j]
    return torch.tensor(ret)


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)

    def id_to_sequence(self, sequence, reverse=False, padding='post', truncating='post'):
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer, opt):

        if '15' in opt.dataset:
            Spath = './data/semeval15/'
        elif '16' in opt.dataset:
            Spath = './data/semeval16/'
        elif '14' in opt.dataset:
            Spath = './data/semeval14/'
        else:
            Spath = './data/acl-14-short-data//'

        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        all_data = []
        train_feature_data, test_feature_data, feature2count, feature2id, _ = generate_knowledge_api(Spath,
                                                                                                     opt.knowledge_type,
                                                                                                     opt.dataset, opt.direct, opt.tool)

        dataset_name = opt.dataset
        for i in range(9):
            dataset_name = dataset_name.replace(str(i),'')
        if 'train' in fname:
            feature_data = train_feature_data
            # fin_dg = open(Spath + dataset_name + '_train.txt.graph', 'rb')
        elif 'test' in fname:
            feature_data = test_feature_data
            # fin_dg = open(Spath + dataset_name + '_test.txt.graph', 'rb')
        # dependency_graphs = pickle.load(fin_dg)
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()

            polarity = lines[i + 2].strip()
            raw_text = text_left + " " + aspect + " " + text_right
            raw_text_bert = '[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]"
            raw_text_bert_single = '[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP]'
            text_bert_single_indices = tokenizer.text_to_sequence(raw_text_bert_single)
            # bert seg_id and bert index constructing
            text_bert_indices = tokenizer.text_to_sequence(raw_text_bert)
            text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            raw_len = (np.sum(text_raw_indices != 0))
            aspect_bert_indices = tokenizer.text_to_sequence(aspect)
            left_bert_indices = tokenizer.text_to_sequence(text_left)
            aspect_len = np.sum(aspect_bert_indices != 0)
            bert_segments_ids = np.asarray([0] * (raw_len + 2) + [1] * (aspect_len + 1))
            bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)

            raw_text_left_list = text_left.split()
            raw_text_right_list = text_right.split()
            raw_aspect_list = aspect.split()
            valid_indices_left, valid_indices_aspect, valid_indices_right = [], [], []
            aspect_indices_left, aspect_indices_aspect, aspect_indices_right = [], [], []
            weight_indices =  []

            for word in raw_text_left_list:
                token = tokenizer.tokenizer.tokenize(word)
                for m in range(len(token)):
                    if m == 0:
                        valid_indices_left.append(1)
                        aspect_indices_left.append(0)
                    else:
                        valid_indices_left.append(0)
            for word in raw_aspect_list:
                token = tokenizer.tokenizer.tokenize(word)
                for m in range(len(token)):
                    if m == 0:
                        valid_indices_aspect.append(1)
                        aspect_indices_aspect.append(1)
                    else:
                        valid_indices_aspect.append(0)
            for word in raw_text_right_list:
                token = tokenizer.tokenizer.tokenize(word)
                for m in range(len(token)):
                    if m == 0:
                        valid_indices_right.append(1)
                        aspect_indices_right.append(0)
                    else:
                        valid_indices_right.append(0)


            valid_indices = [1] + valid_indices_left + valid_indices_aspect + valid_indices_right + [1] \
                            + valid_indices_aspect + [1]
            valid_indices = tokenizer.id_to_sequence(valid_indices)

            aspect_indices = aspect_indices_left + aspect_indices_aspect + aspect_indices_right

            context_indices = [0] + [1] * (len(aspect_indices)-2)
            aspect_indices = tokenizer.id_to_sequence(aspect_indices)
            context_indices = tokenizer.id_to_sequence(context_indices)

            context_len = len(aspect_indices_left) + len(aspect_indices_aspect) + len(aspect_indices_right)
            for w_index in range(len(aspect_indices_left)):
                weight_indices.append(1 - (len(aspect_indices_left) - w_index) / context_len)
            for w_index in range(len(aspect_indices_aspect)):
                weight_indices.append(0)
            for w_index in range(len(aspect_indices_right)):
                weight_indices.append(1 - (w_index+1) / context_len)
            weight_indices = tokenizer.id_to_sequence(weight_indices)
            simple_graph = get_simple_graph(opt.max_seq_len, feature_data[int(i / 3)])
            graph = get_graph(opt.max_seq_len, feature_data[int(i / 3)], feature2id)


            # polarity and labels
            polarity = int(polarity) + 1
            if polarity not in [0,1,2]:
                print('err!')
                print(polarity)
                exit()

            text_raw_bert_indices = tokenizer.text_to_sequence(
                "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
            aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")
            # dependency_graph = dependency_graphs[i]
            # dependency_graph = pad_graph(opt.max_seq_len, dependency_graph)

            data = {
                'raw_text': raw_text_bert,
                'aspect':aspect,
                'text_bert_indices': text_bert_indices,
                'text_bert_single_indices':text_bert_single_indices,
                'left_bert_indices':left_bert_indices,
                'bert_segments_ids': bert_segments_ids,
                'aspect_indices': aspect_indices,
                'context_indices':context_indices,
                'valid_indices': valid_indices,
                'polarity': polarity,
                'text_raw_bert_indices':text_raw_bert_indices,
                'aspect_bert_indices':aspect_bert_indices,
                "simple_graph":simple_graph,
                "graph": graph,
                # "spacy":dependency_graph,
                'weight_indices':weight_indices,
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
