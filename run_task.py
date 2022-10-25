#! coding: utf-8

import os
import argparse
import string
import random
from pathlib import Path
#import multiprocessing as mp
import time
import argparse


def run_task():
    cmd = "python train_main.py --fp16 --seed 40 --data_dir dataset/demo --init_checkpoint ../models/bert-wwm-uncased --vocab_file ../models/bert-wwm-uncased/vocab.txt " \
    "--config_file ../models/bert-wwm-uncased/config.json --max_seq_length=128 --do_train --do_eval --do_lower_case --train_batch_size=8 --eval_batch_size=16 " \
    "--num_train_epochs 10 --num_epoch_multi_cri 5 --learning_rate 1e-05  --warmup_proportion 0.06 " \
    "--output_dir results/AdvMT --multi_criteria --bert_model ../models/bert-wwm-uncased --task_name demo " \
    "--pooling avg_pooling --encoder transformer"

    os.system(cmd)

def main():
    run_task()

if __name__ == "__main__":
    main()
