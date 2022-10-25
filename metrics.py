import os
import re
import torch
import numpy as np
from score import score
from sklearn.metrics import f1_score,classification_report

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]


def semeval_official_eval(label_map, preds, labels, outdir):
    proposed_answer = os.path.join(outdir, "proposed_answer.txt")
    answer_key  = os.path.join(outdir, "answer_key.txt")
    with open(proposed_answer, 'w', encoding='utf-8') as f:
        for idx, pred in enumerate(labels):
            f.write("{}\t{}\n".format(idx, label_map[pred]))
    with open(answer_key, 'w', encoding='utf-8') as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(idx, label_map[pred]))

    eval_cmd = "perl ./eval/semeval2010_task8_scorer-v1.2.pl {} {}".format(proposed_answer, answer_key)
    print(eval_cmd)
    p,r,f1 = 0,0,0
    try:
        msg = [s for s in os.popen(eval_cmd).read().split("\n") if len(s) > 0]
        b_official = False
        for i,s in enumerate(msg):
            if "(9+1)-WAY EVALUATION TAKING DIRECTIONALITY INTO ACCOUNT -- OFFICIAL" in s:
                b_official = True
            if b_official is False:
                continue
            if "MACRO-averaged result (excluding Other)" in s and "F1 =" in msg[i+1]:
                p = float(re.findall('P = (.+?)%', msg[i+1])[0])
                r = float(re.findall('R = (.+?)%', msg[i+1])[0])
                f1 = float(re.findall('F1 = (.+?)%', msg[i+1])[0])
                break

    except Exception as e:
        print(str(e))
        f1 = 0
    print("p: {}, r: {}, f1: {}".format(p, r, f1))
    return {
        "precision": p,
        "recall": r,
        "f1": f1
    }

def tacred_official_eval(label_map, preds, golds, output_dir=None):
    pred_labels = [label_map[id] for id in preds]
    gold_labels = [label_map[id] for id in golds]
    pred_labels = ['no_relation' if l.lower()=="other" else l for l in pred_labels]
    gold_labels = ['no_relation' if l.lower()=="other" else l for l in gold_labels]

    if output_dir is not None:
        proposed_answer = os.path.join(output_dir, "proposed_answer.txt")
        answer_key = os.path.join(output_dir, "answer_key.txt")
        with open(proposed_answer, 'w', encoding='utf-8') as f:
            for idx, pred in enumerate(gold_labels):
                f.write("{}\t{}\n".format(idx, pred))
        with open(answer_key, 'w', encoding='utf-8') as f:
            for idx, pred in enumerate(pred_labels):
                f.write("{}\t{}\n".format(idx, pred))

    prec_micro, recall_micro, f1_micro = score(gold_labels, pred_labels)
    return {
        "precision":prec_micro,
        "recall":recall_micro,
        "f1":f1_micro,
    }

def write_prediction(relation_labels, output_file, preds):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(8001 + idx, relation_labels[pred]))

def compute_micro_f1(preds, labels, label_map, ignore_label, output_dir=None):
    if output_dir is not None:
        proposed_answer = os.path.join(output_dir, "proposed_answer.txt")
        answer_key = os.path.join(output_dir, "answer_key.txt")
        with open(proposed_answer, 'w', encoding='utf-8') as f:
            for idx, pred in enumerate(labels):
                f.write("{}\t{}\n".format(idx, pred))
        with open(answer_key, 'w', encoding='utf-8') as f:
            for idx, pred in enumerate(preds):
                f.write("{}\t{}\n".format(idx, pred))

    target_name = []
    target_id = []
    for name,id in label_map.items():
        if name in ignore_label:
            continue
        target_id.append(id)
        target_name.append(name)
    res = classification_report(labels, preds, labels=target_id,target_names=target_name, digits=4)
    print(res)
    return float([item for item in [s for s in res.split('\n') if 'micro' in s][0].split(' ') if len(item)][-2])*100

def compute_metrics(preds, labels, rel_size, ignore_label):
    assert len(preds) == len(labels)
    # return acc_and_f1(preds, labels)
    return measure_statistics(preds, labels, rel_size, ignore_label)


def simple_accuracy(preds, labels):
    return (preds == labels).float().mean()

def _simple_accuracy(preds, labels):
    examples_length = len(labels)
    n_correct = 0
    n_total = 0
    for i in range(examples_length):
        gold_sentence = labels[i]
        gen_sentence = preds[i]
        for j in range(len(gold_sentence)):
            n_correct += (gold_sentence[j] == gen_sentence[j])
            n_total += 1
    if n_total == 0:
        n_total = 1
    return n_correct / n_total

def _micro_f1(preds, labels, classes):
    examples_length = len(labels)
    TP = 0
    FP = 0
    FN = 0
    for i in range(examples_length):
        gold_sentence = np.array(labels[i])
        gen_sentence = np.array(preds[i])
        for j in range(classes):
            class_j_true = (gold_sentence == j)
            class_j_pred = (gen_sentence == j)
            length = len(class_j_true)
            for k in range(length):
                if (class_j_true[k] == class_j_pred[k]) and (class_j_pred[k] == True):
                    TP += 1
                if (class_j_true[k] == True) and (class_j_pred[k] == False):
                    FN += 1    
                if (class_j_true[k] == False) and (class_j_pred[k] == True):
                    FP += 1
    precision = TP / max((TP + FP), 1)
    recall = TP / max((TP + FN), 1)
    if (precision + recall == 0):
        f1_score = 0
    else:
        f1_score = 2 * precision * recall / (precision + recall)
    return f1_score

def _macro_f1(preds, labels, classes):
    examples_length = len(labels)
    F1s = []
    for j in range(classes):
        TP = 0
        FP = 0
        FN = 0
        for i in range(examples_length):
            gold_sentence = np.array(labels[i])
            gen_sentence = np.array(preds[i])
            class_j_true = (gold_sentence == j)
            class_j_pred = (gen_sentence == j)
            length = len(class_j_true)
            for k in range(length):
                if (class_j_true[k] == class_j_pred[k]) and (class_j_pred[k] == True):
                    TP += 1
                if (class_j_true[k] == True) and (class_j_pred[k] == False):
                    FN += 1    
                if (class_j_true[k] == False) and (class_j_pred[k] == True):
                    FP += 1
        precision = TP / max((TP + FP), 1)
        recall = TP / max((TP + FN), 1)
        if (precision + recall == 0):
            f1_score = 0
        else:
            f1_score = 2 * precision * recall / (precision + recall)
        F1s.append(f1_score)
    F1s = np.array(F1s)
    return F1s.mean()


def acc_and_f1(preds, labels, average='micro', ne_label=0):
    label_size = {0: 2, 1: 19, 2: 19}
    acc = _simple_accuracy(preds, labels)
    f1 = _macro_f1(preds, labels, label_size[ne_label])
    return {
        "acc": acc,
        "f1": f1,
    }

def fbeta_score(precision, recall, beta=1.0):
    beta_square = beta * beta
    if (precision != 0.0) and (recall != 0.0):
        res = ((1 + beta_square) * precision * recall / (beta_square * precision + recall))
    else:
        res = 0.0
    return res

def measure_statistics(preds, labels, rel_size, ignore_label):
    """
    Calculate: True Positives (TP), False Positives (FP), False Negatives (FN)
    GPU & CPU code
    """
    y = torch.from_numpy(preds)
    t = torch.from_numpy(labels)

    label_num = torch.as_tensor([rel_size]).long()
    ignore_label = torch.as_tensor([ignore_label]).long()

    mask_t = torch.eq(t, ignore_label)        # true = no_relation
    mask_p = torch.eq(y, ignore_label)        # pred = no_relation

    true = torch.where(mask_t, label_num, t)  # t: ground truth labels (replace ignored with +1)
    pred = torch.where(mask_p, label_num, y)  # y: output of neural network (replace ignored with +1)

    tp_mask = torch.where(torch.eq(pred, true), true, label_num)
    fp_mask = torch.where(torch.ne(pred, true), pred, label_num)  # this includes wrong positive classes as well
    fn_mask = torch.where(torch.ne(pred, true), true, label_num)

    tp = torch.bincount(tp_mask, minlength=rel_size + 1)[:rel_size]
    fp = torch.bincount(fp_mask, minlength=rel_size + 1)[:rel_size]
    fn = torch.bincount(fn_mask, minlength=rel_size + 1)[:rel_size]
    tn = torch.sum(mask_t & mask_p)

    atp = np.sum(tp.numpy())
    afp = np.sum(fp.numpy())
    afn = np.sum(fn.numpy())
    atn = np.sum(tn.numpy())
    micro_p = (1.0 * atp) / (atp + afp) if (atp + afp != 0) else 0.0
    micro_r = (1.0 * atp) / (atp + afn) if (atp + afn != 0) else 0.0
    micro_f = fbeta_score(micro_p, micro_r)

    return {
        "precision": micro_p,
        "recall": micro_r,
        "f1": micro_f,
    }



