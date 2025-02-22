# LiveLongBench: Tackling Long-Context Understanding for Spoken Texts from Live Streams

## Overview
Long-context understanding poses significant challenges in natural language processing, particularly for real-world dialogues characterized by speech-based elements, high redundancy, and uneven information density. Although large language models (LLMs) achieve impressive results on existing benchmarks, these datasets fail to reflect the complexities of such texts, limiting their applicability to practical scenarios.

To bridge this gap, **LiveLongBench** introduces the first spoken long-text dataset derived from live streams, designed to reflect the redundancy-rich and conversational nature of real-world scenarios. The benchmark features tasks across three main categories:

- **Retrieval-dependent tasks**
- **Reasoning-dependent tasks**
- **Hybrid tasks**

We evaluate a variety of popular LLMs and specialized methods on this benchmark, revealing that current approaches struggle to process highly redundant texts effectively. Our findings indicate clear preferences for specific task types, with no single method excelling across all tasks. To address these challenges, we propose a simple yet strong baseline that significantly improves performance in processing spoken long-texts.

LiveLongBench serves as a valuable resource for advancing long-text understanding in real-world applications, particularly in e-commerce and live-streaming scenarios.

## Repository Structure
```
LiveLongBench/
├── data/                    # Dataset files and preprocessing scripts
├── models/                  # Pretrained models 
├── scripts/                 # Evaluation scripts
├── results/                 # Benchmarking results
├── src/                     # Source code for model implementation and evaluation
├── README.md                # Project documentation
```

## Getting Started
### 1. Clone the Repository
```bash
git clone https://github.com/Yarayx/livelongbench.git
cd livelongbench
```

### 2. Install Dependencies
We recommend setting up a virtual environment:
```bash
conda create -n livelongbench python=3.8
conda activate livelongbench
pip install -r requirements.txt
```

### 3. Running Model Evaluations
To evaluate different models on **LiveLongBench**, run the following script:
```bash
cd livelongbench
bash scripts/eval_llama31_full.sh
```
Modify the script or create new ones to evaluate additional models.



