#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 模型及语言参数（请根据需要确认语言代码是否正确）
MODEL_NAME = "/mnt/LLMs/nllb-200-distilled-600M"
SOURCE_LANG = "zho_Hans"  # 简体中文
TARGET_LANG = "eng_Latn"   # 英文

def batch_translate_words(words, tokenizer, model, batch_size=64):
    """
    批量翻译词语列表，每次处理 batch_size 个词语，返回翻译后的结果列表。
    """
    # 设置源语言（部分模型可能要求提前设置）
    tokenizer.src_lang = SOURCE_LANG
    translated = []
    for i in range(0, len(words), batch_size):
        batch = words[i:i + batch_size]
        # 编码时对输入进行 padding（词语较短，max_length 可设置较小的值）
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=32)
        inputs = inputs.to(model.device)
        # 生成翻译结果
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(TARGET_LANG),
            max_length=32,
            num_beams=4,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        batch_translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated.extend(batch_translated)
        print(f"已翻译 {min(i+batch_size, len(words))}/{len(words)} 个词语")
    return translated

def main():
    # 读取已有的词频 CSV 文件
    freq_dict = {}
    input_csv = "/mnt/livedata/word_frequency_150.csv"
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过表头
        for row in reader:
            word = row[0].strip()
            frequency = int(row[1])
            freq_dict[word] = frequency

    # 提取所有需要翻译的词语（去重）
    words = list(freq_dict.keys())

    # 加载翻译模型及分词器
    print("正在加载翻译模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("模型加载完成。")

    # 批量翻译所有词语
    translated_words = batch_translate_words(words, tokenizer, model, batch_size=64)

    # 将翻译结果与原词频合并（如果翻译结果相同，则累计词频）
    translated_freq = {}
    for orig, trans in zip(words, translated_words):
        # 若翻译结果为空则保留原词
        eng_word = trans.strip().lower() if trans.strip() != "" else orig
        count = freq_dict[orig]
        translated_freq[eng_word] = translated_freq.get(eng_word, 0) + count

    # 将英文词频保存到新的 CSV 文件中
    output_csv = "/mnt/livedata/word_frequency_150_en.csv"
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "frequency"])
        # 按词频降序写入
        for word, count in sorted(translated_freq.items(), key=lambda x: x[1], reverse=True):
            writer.writerow([word, count])
    print(f"英文词频已保存到 {output_csv}")

if __name__ == "__main__":
    main()
