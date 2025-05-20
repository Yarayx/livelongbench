import re
import string
import jieba
import difflib
import numpy as np
from fuzzywuzzy import fuzz
from typing import List
from collections import Counter
from rouge import Rouge
import json
import os
from tqdm import tqdm
import random
import uuid
from pathlib import Path
import glob
import io
import logging
import pysrt

from .token_length import token_length

logger = logging.getLogger(__name__)

## cal_metrics
def extract_number(text):
    match = re.search(r'\[\[([0-9]*\.?[0-9]+)\]\]', text)
    if match:
        return float(match.group(1))
    match = re.search(r'\[([0-9]*\.?[0-9]+)\]', text)
    if match:
        return float(match.group(1))
    return None


def failure_prompts(old_evaluate_output_path, tag):
    eval_lines = open(old_evaluate_output_path).readlines()
    gen_lines = open(old_evaluate_output_path).readlines()
    scores = []
    effective_samples = []
    no_effective_samples = []
    for line in eval_lines:
        line = json.loads(line.strip())
        if not extract_number(line[tag]) or line['generate_response'] == "":
            no_effective_samples.append(line['id'])
    for line in gen_lines:
        line = json.loads(line.strip())
        if line['id'] in no_effective_samples:
            effective_samples.append(
                {'id': line['id'], 'prompt': line['prompt'], 'question': line['question'], 'answer': line['answer']})
    return effective_samples


def cal_metric_ori(evaluate_output_path, tag, task_category=None, live_category=None):
    lines = open(evaluate_output_path).readlines()
    scores = []
    effective_samples = []
    no_effective_samples = []
    # import pdb;pdb.set_trace()
    for line in lines:
        line = json.loads(line.strip())

        _task_category = line.get("task_category", None)
        _live_category = line.get("live_category", None)
        if task_category and _task_category and _task_category != task_category:
            continue
        if live_category and _live_category and _live_category != live_category:
            continue

        if extract_number(line[tag]) is not None:
            scores.append(extract_number(line[tag]))
            effective_samples.append(line)
        else:
            no_effective_samples.append(line['id'])

    num_full_marks = sum(1 for x in scores if x == 100)
    metric = (len(effective_samples) / len(lines), np.mean(scores), f"{num_full_marks}/{len(effective_samples)}", num_full_marks / len(effective_samples))

    logger.info(f"task_category: {task_category}, live_category: {live_category}, scoring_success_rate: {metric[0]:.2f} , avg_score: {metric[1]:.2f} , perfect_rate_calculation: {metric[2]} , perfect_rate: {metric[3]:.2f}")
    return metric

import json
import numpy as np
from collections import defaultdict

def cal_metric(evaluate_output_path, tag, filters):
    """Calculate metrics based on provided filters."""
    lines = open(evaluate_output_path).readlines()
    scores = defaultdict(list)
    effective_samples = defaultdict(list)
    no_effective_samples = []

    for line in lines:
        line = json.loads(line.strip())
        
        # Apply filters
        include = True
        for key, value in filters.items():
            if value is not None and line.get(key) != value:
                include = False
                break

        if not include:
            continue

        score = extract_number(line.get(tag))
        if score is not None:
            scores['all'].append(score)
            effective_samples['all'].append(line)
        else:
            no_effective_samples.append(line['id'])

    metrics = {}
    for key, score_list in scores.items():
        if score_list:
            num_full_marks = sum(1 for x in score_list if x == 100)
            metrics[key] = {
                "scoring_success_rate": len(score_list) / len(lines),
                "avg_score": np.mean(score_list),
                "perfect_rate_calculation": f"{num_full_marks}/{len(score_list)}",
                "perfect_rate": num_full_marks / len(score_list),
            }
    
    return metrics, no_effective_samples

def classify_by_ranges(value, ranges):
    """Classify a numeric value into a given range set."""
    for category, (low, high) in ranges.items():
        if low <= value < high:
            return category
    return None

def compute_and_save_metrics(evaluate_output_path, output_json_dir):
    """Compute and save metrics for multiple classifications."""
    length_sets = {
        "Set1": (0, 20_000),
        "Set2": (20_000, 50_000),
        "Set3": (50_000, 100_000),
        "Set4": (100_000, 300_000),
        "Set5": (300_000, 600_000),
    }

    # Placeholder for speech_pace and sentence_length_metric ranges
    speech_pace_sets = {"Set1": (0, 3), "Set2": (3, 4), "Set3": (4, 5), "Set4": (5, 6), "Set5": (6, 7)}
    sentence_length_metric_sets = {"Set1": (0, 5), "Set2": (5, 7), "Set3": (7, 9), "Set4": (9, 11), "Set5": (11, float('inf'))}

    with open(evaluate_output_path, 'r') as infile:
        lines = [json.loads(line.strip()) for line in infile]

    # Initialize results
    results = {
        "task_category": defaultdict(list),
        "live_category": defaultdict(list),
        "sub_task": defaultdict(list),
        "length_sets": defaultdict(list),
        "speech_pace_sets": defaultdict(list),
        "sentence_length_metric_sets": defaultdict(list),
    }

    # Categorize and compute metrics
    for line in lines:
        task_category = line.get("task_category")
        live_category = line.get("live_category")
        sub_task = line.get("sub_task")
        length_category = classify_by_ranges(line.get("length", 0), length_sets)
        speech_pace_category = classify_by_ranges(line.get("speech_pace", 0), speech_pace_sets)
        sentence_length_category = classify_by_ranges(line.get("sentence_length_metric", 0), sentence_length_metric_sets)

        if task_category:
            results["task_category"][task_category].append(line)
        if live_category:
            results["live_category"][live_category].append(line)
        if sub_task:
            results["sub_task"][sub_task].append(line)
        if length_category:
            results["length_sets"][length_category].append(line)
        if speech_pace_category:
            results["speech_pace_sets"][speech_pace_category].append(line)
        if sentence_length_category:
            results["sentence_length_metric_sets"][sentence_length_category].append(line)

    # Save categorized results as JSON
    for key, data in results.items():
        aggregated_metrics = {}
        for category, items in data.items():
            scores = [extract_number(item.get("eval_response")) for item in items if extract_number(item.get("eval_response")) is not None]
            if scores:
                num_full_marks = sum(1 for x in scores if x == 100)
                aggregated_metrics[category] = {
                    "scoring_success_rate": len(scores) / len(lines),
                    "avg_score": np.mean(scores),
                    "perfect_rate_calculation": f"{num_full_marks}/{len(scores)}",
                    "perfect_rate": num_full_marks / len(scores),
                }
        output_path = f"{output_json_dir}/{key}.json"
        with open(output_path, 'w') as outfile:
            json.dump(aggregated_metrics, outfile, indent=4)

    print(f"Metrics saved in {output_json_dir}")


## prompt

file_handle_cache = {}

def close_cached_files():
    for file, handle in file_handle_cache.items():
        if isinstance(handle, io.IOBase):
            handle.close()
    file_handle_cache.clear()

def extract_text_from_srt(srt_file_path):
    if not os.path.exists(srt_file_path):
        logger.info(f"文件不存在: {srt_file_path}")
        return ""

    try:
        subs = pysrt.open(srt_file_path, encoding='utf-8')
        
        text_parts = [sub.text for sub in subs]
        
        full_text = '\n'.join(text_parts)
        
        return full_text
    except Exception as e:
        logger.Error(f"Error processing file {srt_file_path}: {e}")
        return ""

def get_content(args, item, doc_name, idx):
    global file_handle_cache
    live_category, live_sub_category = item['live_category'], item['live_sub_category']

    docPath = Path(args.doc_path) / live_category / live_sub_category

    path = docPath / f"{doc_name}.srt"
    try:
        doc = extract_text_from_srt(path)
    except Exception as e:
        logger.Error(f"Error processing SRT file {path}: {e}")

    return doc

def get_contents(args, item, doc_names):
    contents = []
    for idx, doc_name in enumerate(doc_names):
        content = get_content(args, item, doc_name, idx)
        contents.append(content)
    return contents


def get_doc_str(args, item, prompt_template):
    len_prompt_template = token_length(prompt_template) - token_length("{docs}")
    is_shuffle = item.get("shuffle_doc", True)

    docs = item['doc'] if not args.rag else item["recall_chunks"][:args.rag_num]
    docs_list = []

    if args.rag:
        for doc in docs:
            if len_prompt_template + sum(token_length(s) for s in docs_list) + token_length(doc) > args.max_length:
                continue
            docs_list.append(doc)
    else:
        # read content from given doc names
        contents = get_contents(args, item, docs)

        for content in contents:
            if len_prompt_template + sum(token_length(s) for s in docs_list) + token_length(content) > args.max_length:
                continue
            docs_list.append(content)

    # shuffle
    if is_shuffle:
        random.shuffle(docs_list)
    docs_str = "".join(docs_list)
    return docs_str


def get_evaluate_prompts(output_path, tag):
    prompt = '''[Question]
{}

[Gold Answer]
{}

[The Start of Assistant's Predicted Answer]
{}
[The End of Assistant's Predicted Answer]

[System]
We would like to request your feedback on the performance of the AI assistant in response to the user question displayed above according to the gold answer. Please use the following listed aspects and their descriptions as evaluation criteria:
    - Accuracy and Hallucinations: The assistant's answer is semantically consistent with the gold answer; The numerical value and order need to be accurate, and there should be no hallucinations.
    - Completeness: Referring to the reference answers, the assistant's answer should contain all the key points needed to answer the user's question; further elaboration on these key points can be omitted.
Please rate whether this answer is suitable for the question. Please note that the gold answer can be considered as a correct answer to the question.

The assistant receives an overall score on a scale of 1 to 100, where a higher score indicates better overall performance.
Please note that if the assistant's answer and the gold answer fully meet the above criteria, its overall rating should be the full marks (100).
Please first provide a comprehensive explanation of your evaluation, avoiding any potential bias.
Then, output a line indicating the score of the Assistant.

PLEASE OUTPUT WITH THE FOLLOWING FORMAT, WHERE THE SCORE IS A SCALE OF 1 TO 100 BY STRICTLY FOLLOWING THIS FORMAT: "[[score]]", FOR EXAMPLE "Rating: [[100]]":
<start output>
Evaluation evidence: your evluation explanation here, no more than 100 words
Rating: [[score]]
<end output>

Now, start your evaluation:'''
    prompts = []
    lines = open(output_path).readlines()
    for line in lines:
        line = json.loads(line.strip())
        line.pop('docs', '')
        question, instruction = line['question'], line['instruction']
        prompt_template = line['prompt_template']
       
        prompt_template = prompt_template.replace("{docs}", "")
        question = prompt_template.replace("{question}", question).replace("{instruction}", instruction)
        answer = line['answer']
        predict = line[tag]
        line['prompt'] = prompt.format(question, answer, predict)
        prompts.append(line)
    return prompts

## utils
def count_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return sum(1 for _ in file)


def create_path(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def continue_gen(input_path, gen_data, tag):
    seen_id = dict()
    with open(input_path, 'r') as f:
        for item in f.readlines():
            js = json.loads(item.strip())
            if js[tag]:
                seen_id[js['id'][0]] = js
    rewrite_data, continue_generate_data = [], []
    seen_rewrite = set()
    for item in gen_data:
        _id = item['id'][0]
        if _id in seen_rewrite:
            continue
        if _id not in seen_id:
            continue_generate_data.append(item)
        else:
            rewrite_data.append(seen_id[_id])
        # dedup
        seen_rewrite.add(_id)
    with open(input_path, 'w') as f:
        for item in rewrite_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"continue_gen: input_path={input_path}, rewrite_data_num={len(rewrite_data)}, tag={tag}")
    return continue_generate_data

