import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import datasets
import json
import torch
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from typing import Optional, Dict, List
from functools import partial
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from accelerate import Accelerator
from transformers import HfArgumentParser
# from transformers.utils import logging
from torch.utils.data import DataLoader

from src import ModelArgs, DefaultDataCollator, FileLogger, makedirs, get_pipeline
from tasks.longlivebench_utils import continue_gen, create_path, count_lines, cal_metric_ori, compute_and_save_metrics
from tasks.prompt import get_evaluate_prompts, get_doc_str, close_cached_files
from src.generate import generate as eval_generate
from src.eval_config import load
import multiprocessing
import requests
import numpy as np
from decimal import Decimal
import time
from openai import OpenAI
import logging

@dataclass
class Args(ModelArgs):
    output_dir: str = field(
        default="./data/results/",
        metadata={'help': 'The base directory for saving results and logs.'}
    )
    result_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'The directory relative to output_dir for saving results.'}
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={'help': 'Max input length.'}
    )
    truncate_from_middle: bool = field(
        default=True,
        metadata={'help': 'Truncate inputs from the middle.'}
    )
    load_result: bool = field(
        default=False,
        metadata={'help': 'Load result from saved files?'}
    )
    mini: Optional[int] = field(
        default=None,
        metadata={'help': 'How many samples for each dataset?'}
    )
    models: str = field(
        default="glm4plus.yaml",
        metadata={'help': 'Path to models configuration file'}
    )
    eval_model: str = field(
        default="glm4plus.yaml",
        metadata={'help': 'Path to evaluation model configuration file'}
    )
    debug_num: int = field(
        default=0,
        metadata={'help': 'Control the number of generated items. If <0, it means using all data'}
    )
    shuffle_prompts: bool = field(
        default=False,
        metadata={'help': 'Whether to shuffle prompts'}
    )
    debug_level: str = field(
        default="1,2,3,4",
        metadata={'help': 'Represents the level to be evaluated, eg: 1,2 or 3'}
    )
    debug_set: str = field(
        default="1,2,3,4",
        metadata={'help': 'Represents the set level to be evaluated, eg: 1,2 or 3'}
    )
    process_num_gen: int = field(
        default=10,
        metadata={'help': 'Number of processes for generation'}
    )
    process_num_eval: int = field(
        default=10,
        metadata={'help': 'Number of processes for evaluation'}
    )
    seed: int = field(
        default=100003,
        metadata={'help': 'Random seed'}
    )
    ratio: float = field(
        default=1.0,
        metadata={'help': 'Ratio for data selection'}
    )
    doc_path: str = field(
        default='./data/livedata/', 
        metadata={'help': 'Path to document data'}
    )
    input_path: str = field(
        default='./data/longlivebench.jsonl', 
        metadata={'help': 'Path to input data'}
    )
    output_process_path: str = field(
        default='longlivebench_process.jsonl',
        metadata={'help': 'Path to processed output data'}
    )
    output_path: str = field(
        default='longlivebench_generate.jsonl',
        metadata={'help': 'Path to output data'}
    )
    evaluate_output_path: str = field(
        default='longlivebench_evaluate.jsonl',
        metadata={'help': 'Path to evaluation output data'}
    )
    max_length: int = field(
        default=50000,
        metadata={'help': 'Maximum length for data'}
    )
    add_noise: bool = field(
        default=False,
        metadata={'help': 'Whether to add noise'}
    )
    rag: bool = field(
        default=False,
        metadata={'help': 'Whether to use RAG model'}
    )
    continue_gen: bool = field(
        default=False,
        metadata={'help': 'Whether to continue generation from existing file'}
    )
    pipeline: str = field(
        default='default',
        metadata={'help': 'Pipeline name. {recall-refine, rag, direct}'}
    )
    strategy: str = field(
        default='default',
        metadata={'help': 'strategy name. {minf, selfE, selfEs, minf_selfE, minf_selfEs}'}
    )
def clean_column(data, column_name):
    column_values = [item[column_name] for item in data]
    if all(isinstance(x, list) for x in column_values):
        return column_values
    elif all(not isinstance(x, list) for x in column_values):
        return column_values
    else:
        return [x if isinstance(x, list) else [x] for x in column_values]


def process_longlivebench(data, indices, tokenizer, max_length=3500, truncate_from_middle=True):
    outputs = {
        'task_category': [], 'live_category': [],"length": [], 'sub_task': [], 
        'language': [], 'question': [], 'instruction': [], 
        'prompt_template': [], 'answer': [], 'id': [], 'prompt': [], 'speech_pace':[],'sentence_length_metric':[]
    }
    
    for task_category, live_category, length, sub_task, language, question, instruction, prompt_template, answer, id, prompt,speech_pace,sentence_length_metric in zip(
        data['task_category'], data['live_category'], data['length'], data['sub_task'], data['language'], 
        data['question'], data['instruction'], data['prompt_template'], 
        data['answer'], data['id'], data['prompt'], data['speech_pace'], data['sentence_length_metric']
    ):
        if max_length is not None:
            if truncate_from_middle:
                try:
                    tokenized_prompt = tokenizer.encode(prompt, add_special_tokens=False)
                except:
                    tokenized_prompt = tokenizer.encode(prompt)
                if len(tokenized_prompt) > max_length:
                    half = int(max_length / 2)
                    prompt = tokenizer.decode(tokenized_prompt[:half]) + tokenizer.decode(tokenized_prompt[-half:])
            else:
                tokenized_prompt = tokenizer.encode(prompt)
                prompt = tokenizer.decode(tokenized_prompt[-max_length:])

        length = len(tokenizer.encode(prompt))
        # answer = flatten_answer(answer)

        outputs["prompt"].append(prompt)
        outputs["length"].append(length)
        outputs["task_category"].append(task_category)
        outputs["live_category"].append(live_category)
        outputs["sub_task"].append(sub_task)
        outputs["language"].append(language)
        outputs["question"].append(question)
        outputs["instruction"].append(instruction)
        outputs["prompt_template"].append(prompt_template)
        outputs["answer"].append(answer)
        outputs["id"].append(id)
        outputs["speech_pace"].append(speech_pace)
        outputs["sentence_length_metric"].append(sentence_length_metric)

    return outputs

def flatten_answer(answer):
    """将 answer 字典转换为字符串格式"""
    if isinstance(answer, dict):
        # 将字典中的每个键值对转换为字符串
        return ', '.join([f"{key}: {', '.join(value)}" for key, value in answer.items()])
    return answer  # 如果不是字典，直接返回

from llmlingua import PromptCompressor
def compress_data(data,cp_rate):
    compressor = PromptCompressor(
                    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                    use_llmlingua2=True
    )
    compression_key = "prompt"
    rate=float(1/cp_rate)
    cp_pt_list = []
    cp_pt_len_list = []
    original_prompt=data[compression_key]
    for prompt in tqdm(original_prompt):
        results = compressor.compress_prompt_llmlingua2(
            prompt,
            rate=rate,
            force_tokens=['\n', '.', '!', '?', ','],
            chunk_end_tokens=['.', '\n'],
            return_word_label=True,
            drop_consecutive=True
        )
        cp_pt_list.append(results['compressed_prompt'])
        cp_pt_len_list.append(results['compressed_tokens'])
    data = data.remove_columns(compression_key).add_column(compression_key, cp_pt_list)
    data = data.remove_columns("length").add_column("length", cp_pt_len_list)

    return data

def get_generate_prompt(args, item):
    replace_dict = {"{question}": item['question'], "{instruction}": item['instruction']}
    prompt_template = item['prompt_template']
    for k, v in replace_dict.items():
        prompt_template = prompt_template.replace(k, v)
    doc_str = get_doc_str(args, item, prompt_template)
    ## lingua compress doc_str
    prompt_template = prompt_template.replace("{docs}", doc_str)
    item['docs'] = doc_str
    item['prompt'] = prompt_template
    return item

def get_generate_prompts(args):
    prompts = []
    with open(args.input_path, 'r') as file:
        lines = file.readlines()

        if args.shuffle_prompts:
            random.shuffle(lines) 
        # debug num samples
        if args.debug_num and args.debug_num > 0:
            lines = lines[:args.debug_num]
        if args.ratio != 1:
            random.shuffle(lines)
            lines = lines[int(len(prompts) * args.ratio):]

        for line in tqdm(lines, desc="gen_prompts"):
            line = line.strip()  
            if not line:  
                continue
            try:
                item = json.loads(line)
                prompt = get_generate_prompt(args, item)
                prompts.append(prompt)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e} for line: {line}")

    close_cached_files()
    return prompts

def fetch_result(pipe, prompt_input, output_path, tag, args):
    prompt = prompt_input['prompt']
    response_content = pipe(prompt, conv=args.conv)
    response_content = response_content[0]
    print("response_content:", response_content)
    
    result = prompt_input.copy()
    result[tag] = response_content or ""

    def convert_to_serializable(value):
        if isinstance(value, torch.Tensor):
            return value.item() if value.numel() == 1 else value.tolist()
        return value

    def format_value(value):
        if isinstance(value, list):
            return [format_value(v) for v in value] 
        elif isinstance(value, dict):
            return {k: format_value(v) for k, v in value.items()}  
        elif isinstance(value, str):
            return value.encode('utf-8').decode('utf-8') 
        return value 

    result = {k: format_value(convert_to_serializable(v)) for k, v in result.items()}

    with open(output_path, 'a', encoding='utf-8') as fw:
        fw.write(json.dumps(result, ensure_ascii=False) + '\n')

    return result


def generate(pipe, prompts, output_path, process_num, tag, args):
    results = []
    for i, prompt in enumerate(tqdm(prompts, total=len(prompts))):
        print(f"Processing prompt {i+1}/{len(prompts)} 'prompt'")
        result = fetch_result(pipe, prompt, output_path, tag, args)
        results.append(result)
    return results

@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]
    output_dir = os.path.join(args.output_dir, args.result_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_process_path = os.path.join(args.output_dir, args.output_process_path)
    output_path = os.path.join(args.output_dir, args.result_dir, args.output_path)
    evaluate_output_path = os.path.join(args.output_dir, args.result_dir, args.evaluate_output_path)

    accelerator = Accelerator(cpu=args.cpu)
    pipe = get_pipeline(args, device=accelerator.device)
    tokenizer = pipe.generator.tokenizer

    # 1. load data from longlivebench.jsonl
    # first
    generate_data = get_generate_prompts(args)
    with open(output_process_path, 'w') as f:
        for p in generate_data:
            f.write(json.dumps(p, ensure_ascii=False, separators=(',', ':')) + "\n")
    
    print(f"Path exist: {output_process_path}")
    # # 2. load data from longlivebench_process.jsonl
    # with open(output_process_path, "r") as f:
    #     generate_data = [json.loads(item.strip()) for item in f.readlines()]

    # # PROMPTS data from jsonl & processing


    with accelerator.main_process_first():
        process_fn = partial(
            process_longlivebench, 
            tokenizer=tokenizer,
            max_length=args.max_length,
            truncate_from_middle=args.truncate_from_middle
        )

        df = pd.DataFrame(generate_data)
        df = df.applymap(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)

        raw_dataset = datasets.Dataset.from_pandas(df)

        try:
            dataset = raw_dataset.map(process_fn, batched=True, num_proc=32, with_indices=True, remove_columns=raw_dataset.column_names)
            print("Dataset mapping completed successfully.")
        except Exception as e:
            print(f"Error during dataset mapping: {e}")
            return

    # llmlingua2 compressed
    dataset = compress_data(dataset,cp_rate=2)
    
    data_collator = DefaultDataCollator(padding_side=args.padding_side)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=data_collator,
        pin_memory=not args.cpu,
    )

    generate_data = accelerator.prepare(dataloader)
    print("DataLoader preparation completed successfully.")

    pipe.generation_kwargs["max_new_tokens"]=512

    random.seed(args.seed)
    print("output_path:",output_path)

    ## generate
    tag = "generate_response"
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    a_path = f"./data/results/{args.result_dir}/unprocess.jsonl"
    if not os.path.exists(a_path):
        create_path(a_path)

    if not os.path.exists(a_path):
        create_path(a_path)
        print(f"Output path created: {a_path}")
        generate(pipe, generate_data, a_path, args.process_num_gen, tag=tag,args=args)
    else:
        if args.continue_gen:
            continue_generate_data = continue_gen(a_path, generate_data, tag=tag,args=args)
            generate(pipe, continue_generate_data, a_path, args.process_num_gen, tag=tag,args=args)
        else:
            logging.debug(f"Path exist: {a_path}")

    memory_max_usage = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)  
    torch.cuda.empty_cache()
    with open(a_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            for key in data:
                if isinstance(data[key], list) and data[key]:  
                    data[key] = data[key][0]  
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"Generation completed successfully. Memory_Max_Usage:{memory_max_usage}")

    print("evaluate_output_path:",evaluate_output_path)
    eval_config = load(open(f"../Loong/config/models/{args.eval_model}"))
    evaluate_prompts = get_evaluate_prompts(output_path, tag="generate_response")
    tag = "eval_response"

    if not os.path.exists(evaluate_output_path):
        create_path(evaluate_output_path)
    eval_generate(evaluate_prompts, eval_config, evaluate_output_path, args.process_num_eval, tag=tag)

    ## cal_metric to json
    output_json_dir = os.path.join(args.output_dir, args.result_dir)
    compute_and_save_metrics(evaluate_output_path, output_json_dir)

    ## cal_metric
    cal_metric_ori(evaluate_output_path, tag="eval_response")
    

if __name__ == "__main__":
    main()

    