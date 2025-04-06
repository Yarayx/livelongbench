import json
import random
from pathlib import Path
import glob
from .token_length import token_length
import io
import os
import pysrt
file_handle_cache = {}

def close_cached_files():
    for file, handle in file_handle_cache.items():
        if isinstance(handle, io.IOBase):
            handle.close()
    file_handle_cache.clear()


def extract_text_from_srt(srt_file_path):
    if not os.path.exists(srt_file_path):
        print(f"File does not exist:{srt_file_path}")
        return ""

    try:
        subs = pysrt.open(srt_file_path, encoding='utf-8')
        text_parts = [sub.text for sub in subs]
        full_text = '\n'.join(text_parts)
        
        return full_text
    except Exception as e:
        print(f"Error processing file {srt_file_path}: {e}")
        return ""

def get_content(args, item, doc_name, idx):
    global file_handle_cache
    live_category, live_sub_category = item['live_category'], item['live_sub_category']

    docPath = Path(args.doc_path) / live_category / live_sub_category

    path = docPath / f"{doc_name}.srt"
    try:
        doc = extract_text_from_srt(path)
    except Exception as e:
        print(f"Error processing SRT file {path}: {e}")

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
    # lines = results
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
        line['task_category']
        prompts.append(line)
    return prompts
