from __future__ import absolute_import, division, print_function

import pickle
import logging
import os
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from models import WEIGHTS_NAME, CONFIG_NAME

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, e1=None, e2=None, e1_entity_relation=None, e2_entity_relation=None):
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
        self.e1_entity_relation = e1_entity_relation
        self.e2_entity_relation = e2_entity_relation

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, e1_mask=None, e2_mask=None,
                 key_seq=None,value_matrix=None, key_mask_matrix=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask
        self.key_seq = key_seq
        self.value_matrix = value_matrix
        self.key_mask_matrix = key_mask_matrix

class SemevalProcessor():
    """Processor for the semeval data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["Component-Whole(e2,e1)","Instrument-Agency(e2,e1)","Member-Collection(e1,e2)","Cause-Effect(e2,e1)",
                "Entity-Destination(e1,e2)","Content-Container(e1,e2)","Message-Topic(e1,e2)","Product-Producer(e2,e1)",
                "Member-Collection(e2,e1)","Entity-Origin(e1,e2)","Cause-Effect(e1,e2)","Component-Whole(e1,e2)",
                "Message-Topic(e2,e1)","Product-Producer(e1,e2)","Entity-Origin(e2,e1)","Content-Container(e2,e1)",
                "Instrument-Agency(e1,e2)","Entity-Destination(e2,e1)","Other"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (e1, e2, label, sentence) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if label == "other":
                label = "Other"
            examples.append(InputExample(guid=guid, text_a=sentence, label=label, e1=e1, e2=e2))
        return examples

    def read_tsv(self, input_file):
        '''
        read file
        return format :
        '''
        data = []
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                splits = line.split('\t')
                if len(splits) < 1:
                    continue
                e1, e2, label, sentence = splits

                e11_p = sentence.index("<e1>")  # the start position of entity1
                e12_p = sentence.index("</e1>")  # the end position of entity1
                e21_p = sentence.index("<e2>")  # the start position of entity2
                e22_p = sentence.index("</e2>")  # the end position of entity2

                if e1 in sentence[e11_p:e12_p] and e2 in sentence[e21_p:e22_p]:
                    data.append(splits)
                elif e2 in sentence[e11_p:e12_p] and e1 in sentence[e21_p:e22_p]:
                    splits[0] = e2
                    splits[1] = e1
                    data.append(splits)
                else:
                    print("data format error: {}".format(line))

        return data

class TacredProcessor(SemevalProcessor):
    def get_labels(self):
        return ["org:founded_by","per:employee_of","org:alternate_names","per:cities_of_residence",
                "per:children","per:title","per:siblings","per:religion","per:age","org:website",
                "per:stateorprovinces_of_residence","org:member_of","org:top_members/employees",
                "per:countries_of_residence","org:city_of_headquarters","org:members","org:country_of_headquarters",
                "per:spouse","org:stateorprovince_of_headquarters","org:number_of_employees/members","org:parents",
                "org:subsidiaries","per:origin","org:political/religious_affiliation","per:other_family",
                "per:stateorprovince_of_birth","org:dissolved","per:date_of_death","org:shareholders",
                "per:alternate_names","per:parents","per:schools_attended","per:cause_of_death","per:city_of_death",
                "per:stateorprovince_of_death","org:founded","per:country_of_birth","per:date_of_birth",
                "per:city_of_birth","per:charges","per:country_of_death","Other"]


class SanwenProcessor(SemevalProcessor):
    def get_labels(self):
        return ["Family","Social","Ownership","Part-Whole","Located","General-Special","Use","Create","Near","Other"]

class Ace05enProcessor(SemevalProcessor):
    def get_labels(self):
        return ["PER-SOC","PHYS","PART-WHOLE","ART","ORG-AFF","GEN-AFF"]

class Ace05cnProcessor(SemevalProcessor):
    def get_labels(self):
        return ["Employment","Org-Location","Near","Geographical","User-Owner-Inventor-Manufacturer","Located",
                "Subsidiary","Business","Citizen-Resident-Religion-Ethnicity","Investor-Shareholder","Family",
                "Membership","Lasting-Personal","Sports-Affiliation","Ownership","Student-Alum","Founder","Artifact",
                "Other"]

def convert_examples_to_features_RAW(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = ["[CLS]"]
        for word in example.text_a.split(" "):
            if word in ["<e1>","</e1>","<e2>","</e2>"]:
                continue
            tokens.extend(tokenizer.tokenize(word))
        if len(tokens) > max_seq_length - 1:
            tokens = tokens[:max_seq_length-1]
        tokens.append("[SEP]")
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logging.info("*** Example ***")
            logging.info("guid: %s" % (example.guid))
            logging.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logging.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              e1_mask=input_mask,
                              e2_mask=input_mask
                              ))
    return features

def convert_examples_to_features_SEG(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = ["[CLS]"]
        segment_ids = [0]
        seg_mask = 0
        for word in tokenizer.tokenize(example.text_a):
            if word in ["<e1>","<e2>"]:
                seg_mask = 1
                continue
            if word in ["</e1>","</e2>"]:
                seg_mask = 0
                continue
            tokens.append(word)
            segment_ids.append(seg_mask)
        if len(tokens) > max_seq_length - 1:
            tokens = tokens[:max_seq_length-1]
            segment_ids = segment_ids[:max_seq_length-1]
        tokens.append("[SEP]")
        segment_ids.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logging.info("*** Example ***")
            logging.info("guid: %s" % (example.guid))
            logging.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logging.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              e1_mask=input_mask,
                              e2_mask=input_mask
                              ))
    return features

def convert_examples_to_features_MARK(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = ["[CLS]"]
        e1_mask = [0]
        e2_mask = [0]
        e1_mask_val = 0
        e2_mask_val = 0
        for word in tokenizer.tokenize(example.text_a):
            if word in ["<e1>","</e1>","<e2>","</e2>"]:
                tokens.append(word)
                e1_mask.append(e1_mask_val)
                e2_mask.append(e2_mask_val)
                if word in ["<e1>"]:
                    e1_mask_val = 1
                elif word in ["</e1>"]:
                    e1_mask_val = 0
                    e1_mask[-1] = 0
                if word in ["<e2>"]:
                    e2_mask_val = 1
                elif word in ["</e2>"]:
                    e2_mask_val = 0
                    e2_mask[-1] = 0
                continue
            tokens.append(word)
            e1_mask.append(e1_mask_val)
            e2_mask.append(e2_mask_val)
        if len(tokens) > max_seq_length - 1:
            tokens = tokens[:max_seq_length-1]
            e1_mask = e1_mask[:max_seq_length-1]
            e2_mask = e2_mask[:max_seq_length-1]
        tokens.append("[SEP]")
        e1_mask.append(0)
        e2_mask.append(0)
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        e1_mask += padding
        e2_mask += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(e1_mask) == max_seq_length
        assert len(e2_mask) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logging.info("*** Example ***")
            logging.info("guid: %s" % (example.guid))
            logging.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logging.info("e1_mask: %s" % " ".join([str(x) for x in e1_mask]))
            logging.info("e2_mask: %s" % " ".join([str(x) for x in e2_mask]))
            logging.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              e1_mask=e1_mask,
                              e2_mask=e2_mask))
    return features

def convert_examples_to_features_MSTART(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = ["[CLS]"]
        e1_mask = [0]
        e2_mask = [0]
        for word in tokenizer.tokenize(example.text_a):
            tokens.append(word)
            e1_mask.append(0)
            e2_mask.append(0)
            if word in ["<e1>"]:
                e1_mask[-1] = 1
            if word in ["<e2>"]:
                e2_mask[-1] = 1
        if len(tokens) > max_seq_length - 1:
            tokens = tokens[:max_seq_length-1]
            e1_mask = e1_mask[:max_seq_length-1]
            e2_mask = e2_mask[:max_seq_length-1]
        tokens.append("[SEP]")
        e1_mask.append(0)
        e2_mask.append(0)
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        e1_mask += padding
        e2_mask += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(e1_mask) == max_seq_length
        assert len(e2_mask) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logging.info("*** Example ***")
            logging.info("guid: %s" % (example.guid))
            logging.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logging.info("e1_mask: %s" % " ".join([str(x) for x in e1_mask]))
            logging.info("e2_mask: %s" % " ".join([str(x) for x in e2_mask]))
            logging.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              e1_mask=e1_mask,
                              e2_mask=e2_mask))
    return features

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, input_fmt):
    if input_fmt == "raw":
        features = convert_examples_to_features_RAW(
            examples, label_list, max_seq_length, tokenizer)
    elif input_fmt == "seg":
        features = convert_examples_to_features_SEG(
            examples, label_list, max_seq_length, tokenizer)
    elif input_fmt == "mark":
        features = convert_examples_to_features_MARK(
            examples, label_list, max_seq_length, tokenizer)
    elif input_fmt == "mstart":
        features = convert_examples_to_features_MSTART(
            examples, label_list, max_seq_length, tokenizer)
    return features

def load_examples(args, tokenizer, processor, label_list, mode, input_fmt="raw"):
    if mode == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif mode == "test":
        examples = processor.get_test_examples(args.data_dir)
    elif mode == "dev":
        examples = processor.get_dev_examples(args.data_dir)

    print("data prep")
    cached_train_features_file = args.data_dir + '_{0}_{1}_{2}_{3}_{4}'.format(
        list(filter(None, args.bert_model.split('/'))).pop(), mode, input_fmt, str(args.max_seq_length), str(args.do_lower_case))
    features = None

    try:
        with open(cached_train_features_file, "rb") as reader:
            features = pickle.load(reader)
    except:
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, input_fmt)
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            logging.info("  Saving features into cached file %s", cached_train_features_file)
            with open(cached_train_features_file, "wb") as writer:
                pickle.dump(features, writer)

    logging.info("***** Running evaluation *****")
    logging.info("mode: {}, input_fmt: {}".format(mode, input_fmt))
    logging.info("  Num examples = %d", len(examples))
    logging.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_e1_mask_ids = torch.tensor([f.e1_mask for f in features], dtype=torch.long)
    all_e2_mask_ids = torch.tensor([f.e2_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_e1_mask_ids, all_e2_mask_ids)


def save_zen_model(save_zen_model_path, model, args):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    # If we save using the predefined names, we can load using `from_pretrained`
    # output_model_file = os.path.join(os.path.join(save_zen_model_path,"../"), WEIGHTS_NAME)
    output_model_file = os.path.join(save_zen_model_path, WEIGHTS_NAME)
    output_config_file = os.path.join(save_zen_model_path, CONFIG_NAME)
    if args.save:
        torch.save(model_to_save.state_dict(), output_model_file)
    with open(output_config_file, "w", encoding='utf-8') as writer:
        writer.write(model_to_save.config.to_json_string())
    output_args_file = os.path.join(save_zen_model_path, 'training_args.bin')
    torch.save(args, output_args_file)
    # final_model_path = os.path.join(save_zen_model_path, "model.pt")
    # torch.save({
    #     "model": model_to_save.state_dict(),
    #     "config": model_to_save.config.to_json_string()
    # }, final_model_path)