import string
import re
import json
import sys
import os
import numpy as np
import argparse
import logging
from collections import Counter
from rouge import rouge_scorer
from transformers import AutoTokenizer

from sklearn.metrics import precision_score, recall_score, f1_score


logger = logging.getLogger(__name__)

class GPTTokenizer:
    gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2", max_length=1e5)

    def tokenize(self, s):
        tokens = self.gpt_tokenizer.tokenize(s)
        # GPT2 uses Byte-level BPE, which will include space as part of the word. 
        # But for the first word of a sentence, there is no space before it. 
        # So, we remove all the added spaces ("Ġ"). 
        tokens = [t.lstrip("Ġ") for t in tokens]
        return tokens

xlingual_tokenizer = GPTTokenizer()


# adapted the flowing from Squad v1.1 evaluation, without removing the articles.
def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def exact_match_score(prediction, ground_truth, xlingual=False):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def rouge1_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = rouge_scorer.RougeScorer(['rouge1'], tokenizer=xlingual_tokenizer)
    else:
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rouge1"].fmeasure


def rougeL_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=xlingual_tokenizer) 
    else:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, xlingual=False):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, xlingual=xlingual)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_metrics(predictions, references, xlingual=False):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    exact_match, rouge1, rougeL = 0, 0, 0
    for pred, gold in zip(predictions, references):
        assert isinstance(gold, list)
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rouge1 += metric_max_over_ground_truths(
            rouge1_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rougeL += metric_max_over_ground_truths(
            rougeL_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
    exact_match = 100.0 * exact_match / len(references)
    rouge1 = 100.0 * rouge1 / len(references)
    rougeL = 100.0 * rougeL / len(references)
    metrics = {"exact_match": exact_match, "rouge1": rouge1, "rougeL": rougeL}
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics

def flat(ori:list):
    tgt = []
    for i in ori:
        if isinstance(i,list):
            tgt.append(i[0])
        elif isinstance(i,str):
            tgt.append(i)
        else:
            raise ValueError("invalid list")
        
    return tgt    

def compute_f_score(gold:list,pred:list):
    # note gold can be multiple choices
    gold = flat(gold)
    assert len(gold) == len(pred)
    if not isinstance(pred,list):
        pred = list(pred)
    # try:
    #     lb2idx = dict([(lb,idx) for idx,lb in enumerate(list(set(gold + pred)))])
    # except TypeError:
    #     print(list(set(gold + pred)))
    #     exit()
    lb2idx = dict([(lb,idx) for idx,lb in enumerate(list(set(gold + pred)))])
    func_ib2idx = lambda x:lb2idx[x]
    gold_idx = list(map(func_ib2idx,gold))
    pred_idx = list(map(func_ib2idx,pred))
    micro = f1_score(gold_idx,pred_idx,average='micro')
    macro = f1_score(gold_idx,pred_idx,average='macro')
    
    return {"micro":micro,"macro":macro}

def compute_metrics_add_f(predictions, references, xlingual=False):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    exact_match, rouge1, rougeL = 0, 0, 0
    for pred, gold in zip(predictions, references):
        assert isinstance(gold, list)
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rouge1 += metric_max_over_ground_truths(
            rouge1_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rougeL += metric_max_over_ground_truths(
            rougeL_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
    exact_match = 100.0 * exact_match / len(references)
    rouge1 = 100.0 * rouge1 / len(references)
    rougeL = 100.0 * rougeL / len(references)
    metrics = {"exact_match": exact_match, "rouge1": rouge1, "rougeL": rougeL}
    f_score = compute_f_score(references,predictions)
    metrics.update(f_score)
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics


def compute_grouped_metrics(predictions, references, groups, xlingual=False):
    assert len(predictions) == len(references) == len(groups)

    examples_by_group = {}
    for pred, gold, group in zip(predictions, references, groups):
        if group not in examples_by_group:
            examples_by_group[group] = []
        examples_by_group[group].append((pred, gold))
    
    results = {}
    for group, group_examples in examples_by_group.items():
        task_predictions, task_references = zip(*group_examples)
        group_metrics = compute_metrics(task_predictions, task_references, xlingual=xlingual)
        for metric, value in group_metrics.items():
            results[f"{metric}_for_{group}"] = value
    return results

def compute_grouped_metrics_add_f(predictions, references, groups, xlingual=False):
    assert len(predictions) == len(references) == len(groups)

    examples_by_group = {}
    for pred, gold, group in zip(predictions, references, groups):
        if group not in examples_by_group:
            examples_by_group[group] = []
        examples_by_group[group].append((pred, gold))
    
    results = {}
    for group, group_examples in examples_by_group.items():
        task_predictions, task_references = zip(*group_examples)
        group_metrics = compute_metrics_add_f(task_predictions, task_references, xlingual=xlingual)
        for metric, value in group_metrics.items():
            results[f"{metric}_for_{group}"] = value
    return results

def compute_grouped_metrics_add_f_v2(predictions, references, groups, xlingual=False):
    '''
    the only difference is that we calculate the avg EM and Rouge for CLS and Gen tasks, respectively. 
    '''
    assert len(predictions) == len(references) == len(groups)
    
    ## metric list
    EM_list = ["TE","CEC","CR","DAR","AC","WA"]
    Rouge_list = ["OE","KT","QR","TG","DT","GEC"]
    ## abv dic
    name_abv = {"answerability_classification":"AC"		
    ,"cause_effect_classification": "CEC"		
    ,"coreference_resolution": "CR"		
    ,"data_to_text": "DT"		
    ,"dialogue_act_recognition": "DAR"		
    ,"grammar_error_correction": "GEC"		
    ,"keyword_tagging": "KT"		
    ,"overlap_extraction": "OE"		
    ,"question_rewriting": "QR"		
    ,"textual_entailment": "TE"		
    ,"title_generation": "TG"		
    ,"word_analogy": "WA"}

    examples_by_group = {}
    for pred, gold, group in zip(predictions, references, groups):
        if group not in examples_by_group:
            examples_by_group[group] = []
        examples_by_group[group].append((pred, gold))
    
    results = {}
    all_em,all_rg = [],[] 
    for group, group_examples in examples_by_group.items():
        task_predictions, task_references = zip(*group_examples)
        group_metrics = compute_metrics_add_f(task_predictions, task_references, xlingual=xlingual)
        for metric, value in group_metrics.items():
            results[f"{metric}_for_{group}"] = value
            # calculate the cls and gen tasks respectively.
            if metric == 'exact_match' and name_abv[group] in EM_list:
                all_em.append(value)
            if metric == 'rougeL' and name_abv[group] in Rouge_list:
                all_rg.append(value)
    
    # avg 
    results['predict_exact_match_avg'] = np.mean(all_em)
    results['predict_rougeL_avg'] = np.mean(all_rg)
    
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to predictions file.")
    parser.add_argument("--track", choices=["default", "xlingual"], default="default", 
        help="default track or xlingual track. For xlingual, we need to use a different tokenizer."
    )
    parser.add_argument("--compute_per_category_metrics", action="store_true", help="Compute metrics on every evaluation category.")
    parser.add_argument("--compute_per_task_metrics", action="store_true", help="Compute metrics on every evaluation task.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.predictions) as fin:
        examples = [json.loads(l) for l in fin]

    predictions = [e["prediction"] for e in examples]
    references = [e["Instance"]["output"] for e in examples]
    tasks = []
    for e in examples:
        if e["Task"] == "task121_atomic_question_rewriting":
            e["Task"] = "task121_zest_question_rewriting"
        tasks.append(e["Task"])

    results = compute_metrics(predictions, references, xlingual=args.track == "xlingual")
    print("======== Overall Metrics ========")
    print("all_rougeL", results["rougeL"])
    print("all_EM", results["exact_match"])
    print()
    
    category_metrics = [
        ("Textual Entailment", "exact_match"),
        ("Cause Effect Classification", "exact_match"),
        ("Coreference Resolution", "exact_match"),
        ("Dialogue Act Recognition", "exact_match"),
        ("Answerability Classification", "exact_match"),
        ("Word Analogy", "exact_match"),
        ("Overlap Extraction", "rougeL"),
        ("Keyword Tagging", "rougeL"),
        ("Question Rewriting", "rougeL"),
        ("Title Generation", "rougeL"),
        ("Data to Text", "rougeL"),
        ("Grammar Error Correction", "rougeL"),
    ]
    category_metrics = {"_".join(category.lower().split()): metric for category, metric in category_metrics}

    if args.compute_per_category_metrics:
        print("======== Metrics per category ========")
        task_category = {}
        for task in set(tasks):
            with open(os.path.join("./data/tasks/", task+".json")) as fin:
                task_data = json.load(fin)
                task_category[task] = "_".join(task_data["Categories"][0].lower().split())
        categories = [task_category[e["Task"]] for e in examples] 
        results.update(compute_grouped_metrics(predictions, references, categories, xlingual=args.track=="xlingual"))
        
        for category, metric in category_metrics.items():
            # category = "_".join(category.lower().split())
            if f"{metric}_for_{category}" in results:
                print(f"{metric}_for_{category}", results[f"{metric}_for_{category}"])
        print()
            
    if args.compute_per_task_metrics:
        print("======== Metrics per task ========")
        results_by_task = compute_grouped_metrics(predictions, references, tasks, xlingual=args.track=="xlingual")
        for task in sorted(list(set(tasks))):
            category = task_category[task]
            metric = category_metrics[category]
            print(task, results_by_task[f"{metric}_for_{task}"])
        print()