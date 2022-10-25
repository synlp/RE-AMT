# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import pickle
import argparse
import csv
import logging
import os
import random
import sys
import time
import glob
import json
import datetime
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from AdvMT import AdvMT
from models import BertTokenizer
from models import BertAdam, warmup_linear
from models import LinearWarmUpScheduler
from apex import amp
from util import is_main_process
from util_task import (
    save_zen_model
)
from metrics import (
    compute_metrics,
    compute_micro_f1,
    acc_and_f1,
    semeval_official_eval,
    tacred_official_eval
)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, e1=None, e2=None, adj=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.e1 = e1
        self.e2 = e2
        self.adj = adj


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, e1_mask=None, e2_mask=None,
                 b_use_valid_filter=False,
                 valid_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask
        self.b_use_valid_filter = b_use_valid_filter
        self.valid_ids = valid_ids


def load_examples(args, tokenizer, processor, mode):
    if mode == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif mode == "test":
        examples = processor.get_test_examples(args.data_dir)
    elif mode == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    return processor.build_dataload(examples, tokenizer, args.max_seq_length, args.batch_size, mode, args)


def change_word(word):
    if "-RRB-" in word:
        return word.replace("-RRB-", ")")
    if "-LRB-" in word:
        return word.replace("-LRB-", "(")
    return word


class REDataset(Dataset):
    def __init__(self, features, max_seq_length):
        self.data = features
        self.max_seq_length = max_seq_length

    def __getitem__(self, index):
        input_ids = torch.tensor(self.data[index]["input_ids"], dtype=torch.long)
        input_mask = torch.tensor(self.data[index]["input_mask"], dtype=torch.long)
        valid_ids = torch.tensor(self.data[index]["valid_ids"], dtype=torch.long)
        segment_ids = torch.tensor(self.data[index]["segment_ids"], dtype=torch.long)
        e1_mask_ids = torch.tensor(self.data[index]["e1_mask"], dtype=torch.long)
        e2_mask_ids = torch.tensor(self.data[index]["e2_mask"], dtype=torch.long)
        label_ids = torch.tensor(self.data[index]["label_id"], dtype=torch.long)
        labels_NE_SL = torch.tensor(self.data[index]["label_ne_sl"], dtype=torch.long)
        labels_NE_RL = torch.tensor(self.data[index]["label_ne_rl"], dtype=torch.long)
        labels_NE_TL = torch.tensor(self.data[index]["label_ne_tl"], dtype=torch.long)
        b_use_valid_filter = torch.tensor(self.data[index]["b_use_valid_filter"], dtype=torch.long)

        return input_ids, input_mask, valid_ids, segment_ids, label_ids, labels_NE_SL, labels_NE_RL, labels_NE_TL, e1_mask_ids, e2_mask_ids, \
               b_use_valid_filter

    def __len__(self):
        return len(self.data)


class SemevalProcessor():
    """Processor for the semeval data set."""

    def __init__(self, direct=False, tool="stanford"):
        self.direct = direct
        self.tool = tool
        self.keys_dict = {}
        self.vals_dict = {}
        self.labels_dict = {}

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.get_knowledge_feature(data_dir, flag="train"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.get_knowledge_feature(data_dir, flag="dev"), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.get_knowledge_feature(data_dir, flag="test"), "test")

    def get_knowledge_feature(self, data_dir, flag="train"):
        return self.read_features(data_dir, flag=flag)

    def get_labels(self):
        return ["Component-Whole(e2,e1)", "Instrument-Agency(e2,e1)", "Member-Collection(e1,e2)", "Cause-Effect(e2,e1)",
                "Entity-Destination(e1,e2)", "Content-Container(e1,e2)", "Message-Topic(e1,e2)",
                "Product-Producer(e2,e1)",
                "Member-Collection(e2,e1)", "Entity-Origin(e1,e2)", "Cause-Effect(e1,e2)", "Component-Whole(e1,e2)",
                "Message-Topic(e2,e1)", "Product-Producer(e1,e2)", "Entity-Origin(e2,e1)", "Content-Container(e2,e1)",
                "Instrument-Agency(e1,e2)", "Entity-Destination(e2,e1)", "Other"]

    def label_mapping(self):
        return {"Component-Whole(e2,e1)": ["Whole", "Component"],
                "Instrument-Agency(e2,e1)": ["Agency", "Instrument"],
                "Member-Collection(e1,e2)": ["Member", "Collection"],
                "Cause-Effect(e2,e1)": ["Effect", "Cause"],
                "Entity-Destination(e1,e2)": ["Entity", "Destination"],
                "Content-Container(e1,e2)": ["Content", "Container"],
                "Message-Topic(e1,e2)": ["Message", "Topic"],
                "Product-Producer(e2,e1)": ["Producer", "Product"],
                "Member-Collection(e2,e1)": ["Collection", "Member"],
                "Entity-Origin(e1,e2)": ["Entity", "Origin"],
                "Cause-Effect(e1,e2)": ["Cause", "Effect"],
                "Component-Whole(e1,e2)": ["Component", "Whole"],
                "Message-Topic(e2,e1)": ["Topic", "Message"],
                "Product-Producer(e1,e2)": ["Product", "Producer"],
                "Entity-Origin(e2,e1)": ["Origin", "Entity"],
                "Content-Container(e2,e1)": ["Container", "Content"],
                "Instrument-Agency(e1,e2)": ["Instrument", "Agency"],
                "Entity-Destination(e2,e1)": ["Destination", "Entity"],
                "Other": ["Other", "Other"]
                }

    def get_key_list(self):
        return self.keys_dict.keys()

    def _create_examples(self, features, set_type):
        examples = []
        for i, feature in enumerate(features):
            guid = "%s-%s" % (set_type, i)
            feature["guid"] = guid
            examples.append(feature)
        return examples

    def read_json(self, data_path):
        data = []
        with open(data_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line == '':
                    continue
                data.append(json.loads(line))
        return data

    def prepare_keys_dict(self, data_dir):
        keys_frequency_dict = defaultdict(int)
        for flag in ["train", "test", "dev"]:
            datafile = os.path.join(data_dir, '{}.{}.json'.format(flag, self.tool))
            if os.path.exists(datafile) is False:
                continue
            all_data = self.read_json(datafile)
            for data in all_data:
                for word in data['word']:
                    keys_frequency_dict[change_word(word)] += 1
        keys_dict = {"[UNK]": 0}
        for key, freq in sorted(keys_frequency_dict.items(), key=lambda x: x[1], reverse=True):
            keys_dict[key] = len(keys_dict)
        self.keys_dict = keys_dict

    def prepare_labels_dict(self):
        label_list = self.get_labels()
        labels_dict = {}
        for label in label_list:
            labels_dict[label] = len(labels_dict)
        self.labels_dict = labels_dict

    def read_features(self, data_dir, flag):
        all_data = self.read_json(os.path.join(data_dir, '{}.{}.json'.format(flag, self.tool)))
        all_feature_data = []
        for data in all_data:
            tokens = []
            sentences = data['sentences']
            for sentence in sentences:
                tokens.extend(sentence['tokens'])
            ori_sentence = data["ori_sentence"]
            label = data["label"]
            if label == "other":
                label = "Other"

            e11_p = ori_sentence.index("<e1>")  # the start position of entity1
            e12_p = ori_sentence.index("</e1>")  # the end position of entity1
            e21_p = ori_sentence.index("<e2>")  # the start position of entity2
            e22_p = ori_sentence.index("</e2>")  # the end position of entity2

            if e11_p < e21_p:
                start_range = list(range(e11_p, e12_p - 1))
                end_range = list(range(e21_p - 2, e22_p - 3))
            else:
                start_range = list(range(e11_p - 2, e12_p - 3))
                end_range = list(range(e21_p, e22_p - 1))

            words = []
            TL_label_squence = []
            for token in tokens:
                token['word'] = token['word'].replace('\xa0', '')
                words.append(change_word(token['word']))
                TL_label_squence.append(token['TL_tag'])

            all_feature_data.append({
                "words": words,
                "ori_sentence": ori_sentence,
                "label": label,
                "e1": data["e1"],
                "e2": data["e2"],
                'TL_label_squence': TL_label_squence
            })
        return all_feature_data

    def get_ne_label_size(self, ne_label):
        label_size = {0: 2, 1: 19, 2: 19}
        return label_size[ne_label]

    def convert_examples_to_features(self, examples, tokenizer, max_seq_length):
        """Loads a data file into a list of `InputBatch`s."""

        label_mapping_tl = {'O': 0, 'CARDINAL': 1, 'DATE': 2, 'EVENT': 3, 'FAC': 4, 'GPE': 5, 'LANGUAGE': 6, 'LAW': 7,
                            'LOC': 8, 'MONEY': 9, 'NORP': 10, 'ORDINAL': 11, 'ORG': 12, 'PERCENT': 13, 'PERSON': 14,
                            'PRODUCT': 15, 'QUANTITY': 16, 'TIME': 17, 'WORK_OF_ART': 18}

        label_map = self.labels_dict
        label_map_temp = self.label_mapping()
        label_mapping = dict()
        temp = set()
        for label_ in label_map_temp.keys():
            temp.add(label_map_temp[label_][0])
            temp.add(label_map_temp[label_][1])
        temp = list(temp)
        for i in temp:
            label_mapping[i] = len(label_mapping) + 1
        label_mapping["O"] = 0
        features = []
        b_use_valid_filter = False
        for (ex_index, example) in enumerate(examples):
            tokens = ["[CLS]"]
            valid = [0]
            e1_mask = [0]
            e2_mask = [0]
            e1_mask_val = 0
            e2_mask_val = 0
            SL_val = 0
            RL_val = 0
            TL_val = 0
            labels_NE_SL = [0]
            labels_NE_RL = [0]
            labels_NE_TL = [0]
            for i, word in enumerate(example["ori_sentence"]):
                if len(tokens) >= max_seq_length - 1:
                    break
                if word in ["<e1>", "</e1>", "<e2>", "</e2>"]:
                    tokens.append(word)
                    labels_NE_SL.append(SL_val)
                    labels_NE_RL.append(RL_val)
                    labels_NE_TL.append(TL_val)
                    valid.append(0)
                    e1_mask.append(e1_mask_val)
                    e2_mask.append(e2_mask_val)
                    if word in ["<e1>"]:
                        e1_mask_val = 1
                        SL_val = 1
                        RL_val = label_mapping[label_map_temp[example["label"]][0]]
                        
                        next_tag = example['TL_label_squence'][i + 1]
                        TL_val = label_mapping_tl[next_tag]
                    elif word in ["</e1>"]:
                        e1_mask_val = 0
                        SL_val = 0
                        RL_val = 0
                        TL_val = 0
                        e1_mask[-1] = 0
                        labels_NE_SL[-1] = 0
                        labels_NE_RL[-1] = 0
                        labels_NE_TL[-1] = 0
                    if word in ["<e2>"]:
                        e2_mask_val = 1
                        SL_val = 1
                        RL_val = label_mapping[label_map_temp[example["label"]][1]]
                        
                        next_tag = example['TL_label_squence'][i + 1]
                        TL_val = label_mapping_tl[next_tag]
                    elif word in ["</e2>"]:
                        e2_mask_val = 0
                        SL_val = 0
                        RL_val = 0
                        TL_val = 0
                        e2_mask[-1] = 0
                        labels_NE_SL[-1] = 0
                        labels_NE_RL[-1] = 0
                        labels_NE_TL[-1] = 0
                    continue
                token = tokenizer.tokenize(word)
                if len(tokens) + len(token) > max_seq_length - 1:
                    break
                tokens.extend(token)
                # e1_mask.append(e1_mask_val)
                # e2_mask.append(e2_mask_val)
                for m in range(len(token)):
                    if m == 0:
                        valid.append(1)
                        e1_mask.append(e1_mask_val)
                        e2_mask.append(e2_mask_val)
                        labels_NE_SL.append(SL_val)
                        labels_NE_RL.append(RL_val)
                        labels_NE_TL.append(TL_val)
                    else:
                        valid.append(1)
                        e1_mask.append(e1_mask_val)
                        e2_mask.append(e2_mask_val)
                        labels_NE_SL.append(SL_val)
                        labels_NE_RL.append(RL_val)
                        labels_NE_TL.append(TL_val)
                        b_use_valid_filter = True

            tokens.append("[SEP]")
            valid.append(0)
            e1_mask.append(0)
            e2_mask.append(0)
            labels_NE_SL.append(0)
            labels_NE_RL.append(0)
            labels_NE_TL.append(0)
            segment_ids = [0] * len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            ner_label_sq = ['O'] + example['TL_label_squence']
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            valid += padding
            e1_mask += [0] * (max_seq_length - len(e1_mask))
            e2_mask += [0] * (max_seq_length - len(e2_mask))
            labels_NE_SL += [0] * (max_seq_length - len(labels_NE_SL))
            labels_NE_RL += [0] * (max_seq_length - len(labels_NE_RL))
            labels_NE_TL += [0] * (max_seq_length - len(labels_NE_TL))
            ner_label_sq += ['O'] * (max_seq_length - len(ner_label_sq))

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(valid) == max_seq_length
            assert len(e1_mask) == max_seq_length
            assert len(e2_mask) == max_seq_length
            assert len(labels_NE_SL) == max_seq_length
            assert len(labels_NE_RL) == max_seq_length
            assert len(labels_NE_TL) == max_seq_length
            assert len(ner_label_sq) == max_seq_length

            max_words_num = sum(valid)

            label_id = label_map[example["label"]]

            features.append({
                "input_ids": input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
                "label_id": label_id,
                "label_ne_sl": labels_NE_SL,
                "label_ne_rl": labels_NE_RL,
                "label_ne_tl": labels_NE_TL,
                "valid_ids": valid,
                "e1_mask": e1_mask,
                "e2_mask": e2_mask,
                "b_use_valid_filter": b_use_valid_filter
            })
        return features

    def build_dataset(self, examples, tokenizer, max_seq_length, mode, args):
        features = self.convert_examples_to_features(examples, tokenizer, max_seq_length)
        
        return REDataset(features, max_seq_length)


class Ace05enProcessor(SemevalProcessor):
    def get_labels(self):
        return ["PER-SOC", "PHYS", "PART-WHOLE", "ART", "ORG-AFF", "GEN-AFF", "Other"]

    def label_mapping(self):
        return {"PER-SOC": ["PER", "SOC"],
                "PHYS": ["PHYS_1", "PHYS_2"],
                "PART-WHOLE": ["PART", "WHOLE"],
                "ART": ["ART_1", "ART_2"],
                "ORG-AFF": ["ORG", "AFF"],
                "GEN-AFF": ["GEN", "AFF"],
                "Other": ["Other", "Other"]
                }

    def get_ne_label_size(self, ne_label):
        label_size = {0: 2, 1: 13, 2: 2}
        return label_size[ne_label]


def train_multi_criteria(model, optimizer, scheduler, train_data_loader, global_step, args, ne_label=0):
    tr_loss = 0
    optimizer.zero_grad()
    nb_tr_examples, nb_tr_steps = 0, 0
    criteria_index = 1
    for step, batch in enumerate(tqdm(train_data_loader, desc="Iteration")):
        if args.max_steps > 0 and global_step > args.max_steps:
            break
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, valid_ids, segment_ids, label_ids, labels_NE_SL, labels_NE_RL, labels_NE_TL, e1_mask, e2_mask, \
        b_use_valid_filter = batch
        b_use_valid_filter = b_use_valid_filter.detach().cpu().numpy()[0]
        labels_NE = [labels_NE_SL, labels_NE_RL, labels_NE_TL]

        tag_seq, loss_re, loss_ne, loss_at = model(input_ids, criteria_index, segment_ids, input_mask, label_ids,
                                                   labels_NE=labels_NE[ne_label], e1_mask=e1_mask, e2_mask=e2_mask,
                                                   b_use_valid_filter=b_use_valid_filter, valid_ids=valid_ids)
        loss = loss_re + loss_ne * loss_at
        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if args.fp16:
                # modify learning rate with special warm up for BERT which FusedAdam doesn't do
                scheduler.step()

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

    avg_loss = tr_loss / nb_tr_steps if nb_tr_steps > 0 else 0
    return avg_loss, global_step


def train_single_criteria(model, optimizer, scheduler, train_data_loader, global_step, args):
    tr_loss = 0
    optimizer.zero_grad()
    nb_tr_examples, nb_tr_steps = 0, 0
    criteria_index = 0
    for step, batch in enumerate(tqdm(train_data_loader, desc="Iteration")):
        if args.max_steps > 0 and global_step > args.max_steps:
            break
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, valid_ids, segment_ids, label_ids, labels_NE_SL, labels_NE_RL, labels_NE_TL, e1_mask, e2_mask, \
        b_use_valid_filter = batch
        b_use_valid_filter = b_use_valid_filter.detach().cpu().numpy()[0]
        labels_NE = [labels_NE_SL, labels_NE_RL, labels_NE_TL]

        tag_seq, loss_re, loss_ne, loss_at = model(input_ids, criteria_index, segment_ids, input_mask, label_ids,
                                                   e1_mask=e1_mask, e2_mask=e2_mask,
                                                   b_use_valid_filter=b_use_valid_filter, valid_ids=valid_ids)
        loss = loss_re
        
        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if args.fp16:
                # modify learning rate with special warm up for BERT which FusedAdam doesn't do
                scheduler.step()
            
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

    avg_loss = tr_loss / nb_tr_steps if nb_tr_steps > 0 else 0
    return avg_loss, global_step


def train(args, model, tokenizer, processor, device, n_gpu, results):
    results["best_checkpoint"] = 0
    results["best_acc_score"] = 0
    results["best_f1_score"] = -1
    results["best_dev_f1_score"] = 0
    results["best_mrr_score"] = 0
    results["best_checkpoint_path"] = ""
    results["stage_1_acc_ner"] = 0
    results["stage_1_f1_ner"] = 0
    results["best_acc_stage_1"] = 0
    results["best_f1_stage_1"] = 0

    train_examples = processor.get_train_examples(args.data_dir)
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    num_train_optimization_steps_1 = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_epoch_multi_cri
    num_train_optimization_steps_2 = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * (
                                             args.num_train_epochs - args.num_epoch_multi_cri)

    logger.info("epoch : {}".format(num_train_optimization_steps))
    

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.fp16:
        print("using fp16")
        try:
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False)

        if args.loss_scale == 0:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                              loss_scale="dynamic")
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                              loss_scale=args.loss_scale)

        scheduler = LinearWarmUpScheduler(optimizer, warmup=args.warmup_proportion,
                                          total_steps=num_train_optimization_steps_1)
    else:
        print("using fp32")
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps_1)

        scheduler = None

    print("lr: {} warm: {} total_step: {}".format(args.learning_rate, args.warmup_proportion,
                                                  num_train_optimization_steps))

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    num_of_no_improvement = 0

    train_data = processor.build_dataset(train_examples, tokenizer, args.max_seq_length, "train", args)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    train_dataloader_tool = train_dataloader

    model.train()
    for epoch_num in trange(int(args.num_train_epochs), desc="Epoch"):

        if epoch_num == args.num_epoch_multi_cri:
            logger.info("Loading best checkpoint from stage 1...")
            model_path = os.path.join(args.output_dir, 'best_checkpoint_stage_1.bin')
            model_dict = torch.load(model_path)
            model.load_state_dict(model_dict)

            name_list = ['classifier_at', 'private_encoder_ne', 'classifier_ne']
            for name, value in model.named_parameters():
                if name in name_list:
                    value.requires_grad = False
            params_ = filter(lambda p: p.requires_grad, model.parameters())
            
            if args.fp16:
                optimizer = FusedAdam(params_,
                                      lr=args.learning_rate,
                                      bias_correction=False)

                if args.loss_scale == 0:
                    model, optimizer = amp.initialize(model.float(), optimizer, opt_level="O2",
                                                      keep_batchnorm_fp32=False,
                                                      loss_scale="dynamic")
                else:
                    model, optimizer = amp.initialize(model.float(), optimizer, opt_level="O2",
                                                      keep_batchnorm_fp32=False,
                                                      loss_scale=args.loss_scale)

                scheduler = LinearWarmUpScheduler(optimizer, warmup=args.warmup_proportion,
                                                  total_steps=num_train_optimization_steps_2)
            else:
                optimizer = BertAdam(params_,
                                     lr=args.learning_rate,
                                     warmup=args.warmup_proportion,
                                     t_total=num_train_optimization_steps_2)

                scheduler = None
            model.train()

        if epoch_num < args.num_epoch_multi_cri:
            _loss, global_step = train_multi_criteria(model, optimizer, scheduler, train_dataloader, global_step, args,
                                                      ne_label=args.ne_label)
        else:
            _loss, global_step = train_single_criteria(model, optimizer, scheduler, train_dataloader, global_step, args)

       
        # Save model checkpoint
        output_dir = os.path.join(args.output_dir, "epoch-{}".format(epoch_num))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_zen_model(output_dir, model, args)

        checkpoint = output_dir
        result = evaluate(args, model, tokenizer, processor, device, mode="test", output_dir=output_dir, ne_label=args.ne_label)
        model.train()
        if result["f1"] > results["best_f1_score"]:
            if epoch_num < args.num_epoch_multi_cri:
                results["stage_1_acc_ner"] = result["acc_ner"]
                results["stage_1_f1_ner"] = result["f1_ner"]
                results["best_acc_stage_1"] = result["precision"]
                results["best_f1_stage_1"] = result["f1"]
            results["best_f1_score"] = result["f1"]
            results["best_p_score"] = result["precision"]
            results["best_r_score"] = result["recall"]
            results["best_checkpoint"] = epoch_num
            results["best_checkpoint_path"] = checkpoint
            num_of_no_improvement = 0
            logger.info("Saving models...")
            if epoch_num < args.num_epoch_multi_cri:
                model_to_save = model.module if hasattr(model, 'module') else model
                model_path = os.path.join(args.output_dir, 'best_checkpoint_stage_1.bin')
                torch.save(model_to_save.state_dict(), model_path)
            else:
                model_to_save = model.module if hasattr(model, 'module') else model
                model_path = os.path.join(args.output_dir, 'best_checkpoint_stage_2.bin')
                torch.save(model_to_save.state_dict(), model_path)
        else:
            num_of_no_improvement += 1
        result = {"{}_test_{}".format(epoch_num, k): v for k, v in result.items()}
        results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            writer.write(json.dumps(results, ensure_ascii=False))

    loss = _loss if args.do_train else None
    return loss, global_step


def evaluate(args, model, tokenizer, processor, device, mode="test", output_dir=None, ne_label=1):
    label_map = processor.labels_dict
    id2label_map = {i: label for label, i in processor.labels_dict.items()}

    if mode == "test":
        examples = processor.get_test_examples(args.data_dir)
    elif mode == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    eval_data = processor.build_dataset(examples, tokenizer, args.max_seq_length, mode, args)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    nb_eval_steps, nb_eval_examples = 0, 0
    pred_scores = None
    out_label_ids = None
    eval_start_time = time.time()
    criteria_index = 0
    gold_ne_seq = []
    pred_ne_seq = []
    ori_seq = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)

        input_ids, input_mask, valid_ids, segment_ids, label_ids, labels_NE_SL, labels_NE_RL, labels_NE_TL, e1_mask, e2_mask, \
        b_use_valid_filter = batch
        b_use_valid_filter = b_use_valid_filter.detach().cpu().numpy()[0]
        labels_NE = [labels_NE_SL, labels_NE_RL, labels_NE_TL]

        with torch.no_grad():
            logits, logits_ne = model(input_ids, criteria_index, segment_ids, input_mask, e1_mask=e1_mask, e2_mask=e2_mask,
                           b_use_valid_filter=b_use_valid_filter, valid_ids=valid_ids)

        nb_eval_steps += 1
        gold_ne_seq.extend(labels_NE[ne_label].tolist())
        pred_ne_seq.extend(logits_ne.tolist())
        ori_seq.extend(input_ids.tolist())
        if pred_scores is None:
            pred_scores = logits.detach().cpu().numpy()
            out_label_ids = label_ids.detach().cpu().numpy()
        else:
            pred_scores = np.append(pred_scores, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, label_ids.detach().cpu().numpy(), axis=0)

    preds = np.argmax(pred_scores, axis=1)
    pred_scores = pred_scores[:, 1].reshape(-1).tolist()
    eval_run_time = time.time() - eval_start_time

    if args.task_name == 'semeval':
        result = semeval_official_eval(id2label_map, preds, out_label_ids, output_dir)
    elif "tacred" in args.task_name:
        result = tacred_official_eval(id2label_map, preds, out_label_ids, output_dir)
    else:
        result = compute_metrics(preds, out_label_ids, len(label_map), label_map['Other'])
        result["micro-f1"] = compute_micro_f1(preds, out_label_ids, label_map, ignore_label='Other',
                                              output_dir=output_dir)
        result["f1"] = result["micro-f1"]
    result["eval_run_time"] = eval_run_time
    result["inference_time"] = eval_run_time / len(eval_data)
    for i in range(len(gold_ne_seq)):
        index = ori_seq[i].index(0)
        gold_ne_seq[i] = gold_ne_seq[i][:index]
        pred_ne_seq[i] = pred_ne_seq[i][:index]

    acc_f1 = acc_and_f1(pred_ne_seq, gold_ne_seq, ne_label=args.ne_label)
    result["acc_ner"] = acc_f1["acc"]
    result["f1_ner"] = acc_f1["f1"]
    logging.info(result)

    return result


def get_model_param(model):
    n_trainable_params, n_nontrainable_params = 0, 0
    for p in model.parameters():
        n_params = torch.prod(torch.tensor(p.shape))
        if p.requires_grad:
            n_trainable_params += n_params.item()
        else:
            n_nontrainable_params += n_params.item()
    logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
    logger.info('> training arguments:')
    return {
        "n_trainable_params": n_trainable_params,
        "n_nontrainable_params": n_nontrainable_params,
        "n_params": n_nontrainable_params + n_trainable_params
    }


def get_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        required=True,
                        help="The checkpoint file from pretraining")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--eval_dev",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--eval_all_cuda",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--b_2_kvmn",
                        action='store_true',
                        help="b_2_kvmn.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--google_pretrained",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--max_steps", default=-1.0, type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--direct',
                        action='store_true',
                        help="Whether to consider dependency type derection")
    parser.add_argument('--b_tg',
                        action='store_true',
                        help="using type gcn")
    parser.add_argument('--b_att',
                        action='store_true',
                        help="gcn using attention")
    parser.add_argument('--b_ale',
                        action='store_true',
                        help="gcn using attentive ensemble")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument("--old", action='store_true', help="use old fp16 optimizer")
    parser.add_argument('--vocab_file',
                        type=str, default=None, required=True,
                        help="Vocabulary mapping/file BERT was pretrainined on")
    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")
    parser.add_argument("--rank",
                        type=int,
                        default=0,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--init_method', type=str, default='tcp://127.0.0.1:23456')
    parser.add_argument('--input_fmt', type=str, default='raw')
    parser.add_argument('--model_type', type=str, default='bert_entity')
    parser.add_argument('--max_key_size', type=int, default=128)
    parser.add_argument('--save', action='store_true', help="save model")
    parser.add_argument("--tool", type=str, default="ner")
    parser.add_argument("--cls_method", type=str, default="cls_entity")
    parser.add_argument("--pooling", type=str, default="avg_pooling")

    parser.add_argument('--multi_criteria', action='store_true', help="Using multi criteria")
    parser.add_argument('--adversary', action='store_true', help="Using adversary learning")
    parser.add_argument('--patient', default=5, type=int)
    parser.add_argument('--num_epoch_multi_cri', default=20, type=int)
    parser.add_argument('--ne_label', default=0, type=int)
    parser.add_argument('--encoder', default="transformer", type=str)

    args = parser.parse_args()

    args.task_name = args.task_name.lower()

    return args


def main():
    args = get_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "semeval": SemevalProcessor,
        "ace2005en": Ace05enProcessor,
        "demo": Ace05enProcessor
    }

    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    
    args.device = device
    args.n_gpu = n_gpu
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    str_args = "bts_{}_lr_{}_warmup_{}_seed_{}_pooling_{}".format(
        args.train_batch_size,
        args.learning_rate,
        args.warmup_proportion,
        args.seed,
        args.pooling
    )
    args.output_dir = os.path.join(args.output_dir, 'result-{}-{}-{}'.format(args.task_name, str_args, now_time))
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        print("WARNING: Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir) and is_main_process():
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()
    args.task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name](tool=args.tool)
    processor.prepare_labels_dict()
    label_list = processor.get_labels()
    num_labels = len(label_list)
    num_labels_ne = processor.get_ne_label_size(args.ne_label)

    print("LOAD tokenizer from", args.vocab_file)
    tokenizer = BertTokenizer(args.vocab_file, do_lower_case=args.do_lower_case,
                              max_len=args.max_seq_length)  # for bert large
    tokenizer.add_never_split_tokens(["<e1>", "</e1>", "<e2>", "</e2>"])
    print("LOAD CHECKPOINT from", args.init_checkpoint)

    model = AdvMT(tag_size=num_labels, ne_tag_size=num_labels_ne, bert_path=args.bert_model, embedding="bert",
                    adv_coefficient=0.06, pooling=args.pooling, encoder=args.encoder, adversary=args.adversary)

    model.to(device)

    results = {"init_checkpoint": args.init_checkpoint, "lr": args.learning_rate, "warmup": args.warmup_proportion,
               "train_batch_size": args.train_batch_size * args.gradient_accumulation_steps,
               "fp16": args.fp16, "b_tg": args.b_tg, "b_att": args.b_att, "b_ale": args.b_ale}
    results.update(get_model_param(model))
    results["train_start_runtime"] = time.time()
    loss, global_step = train(args, model, tokenizer, processor, device, n_gpu, results)
    results["train_runtime"] = time.time() - results["train_start_runtime"]
    results["global_step"] = global_step
    results["loss"] = loss
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        writer.write(json.dumps(results, ensure_ascii=False))
    for key in sorted(results.keys()):
        logger.info("{} = {}\n".format(key, str(results[key])))


if __name__ == "__main__":
    main()
