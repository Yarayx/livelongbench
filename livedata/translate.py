#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 模型及语言参数
MODEL_NAME = "/mnt/LLMs/nllb-200-distilled-600M"
SOURCE_LANG = "zho_Hans"  # 简体中文
TARGET_LANG = "eng_Latn"   # 英文

def translate_text(text, tokenizer, model, source_lang, target_lang):
    """
    利用翻译模型将文本翻译成目标语言。
    此处按换行符将文本拆分为多个句子，逐句翻译后再合并返回。
    """
    # 设置源语言
    tokenizer.src_lang = source_lang
    sentences = text.split('\n')
    translated_sentences = []
    for sentence in sentences:
        # 对于空行，直接保留空行
        if sentence.strip() == "":
            translated_sentences.append("")
            continue
        inputs = tokenizer(sentence, return_tensors="pt", max_length=1024, truncation=True)
        inputs = inputs.to(model.device)
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
            max_length=512,                # 限制生成最大长度
            num_beams=4,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        translated_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        translated_sentences.append(translated_sentence)
    return "\n".join(translated_sentences)

def translate_srt_file(input_file, output_file, tokenizer, model, source_lang, target_lang):
    """
    读取一个 srt 文件，将其中每个字幕块中的文本内容翻译成英文，
    并保持字幕索引和时间戳不变，最后写入新的 srt 文件。
    """
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # SRT 文件的字幕块通常以空行分隔
    blocks = content.strip().split("\n\n")
    output_blocks = []
    for block in blocks:
        lines = block.splitlines()
        if not lines:
            continue
        # 如果该块不符合常规的 srt 格式（例如少于2行），则原样保留
        if len(lines) < 2:
            output_blocks.append(block)
            continue

        # 第一行为字幕索引，第二行为时间戳
        index_line = lines[0]
        timestamp_line = lines[1]
        # 剩下的为字幕文本，需要翻译
        text_lines = lines[2:] if len(lines) > 2 else []
        if text_lines:
            text = "\n".join(text_lines)
            translated_text = translate_text(text, tokenizer, model, source_lang, target_lang)
            translated_lines = translated_text.split("\n")
        else:
            translated_lines = []
        new_block = "\n".join([index_line, timestamp_line] + translated_lines)
        output_blocks.append(new_block)

    output_content = "\n\n".join(output_blocks)
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_content)

def main():
    # 输入与输出目录配置
    input_root = "/mnt/livedata/text_processed"
    output_root = "/mnt/livedata/text_processed_en"

    # 加载翻译模型和分词器
    print("正在加载翻译模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("翻译模型加载完成。")

    # 遍历输入目录下所有 srt 文件
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(".srt"):
                input_file_path = os.path.join(root, file)
                # 计算相对路径，并构造输出文件路径
                rel_path = os.path.relpath(input_file_path, input_root)
                output_file_path = os.path.join(output_root, rel_path)
                print(f"正在翻译文件: {input_file_path} -> {output_file_path}")
                translate_srt_file(input_file_path, output_file_path, tokenizer, model, SOURCE_LANG, TARGET_LANG)

if __name__ == '__main__':
    main()
