import os
import shutil
import random

src_root = "/mnt/yx/yongxuan/livelongbench0406/data/livedata_all"
dst_root = "/mnt/yx/yongxuan/livelongbench0406/data/livedata"

# 创建目标根目录
os.makedirs(dst_root, exist_ok=True)

def split_srt_blocks(lines):
    """将srt文件按字幕块分割，每个块是一个字幕条目"""
    blocks = []
    block = []
    for line in lines:
        if line.strip() == "":
            if block:
                blocks.append(block)
                block = []
        else:
            block.append(line)
    if block:
        blocks.append(block)
    return blocks

def save_random_10_srt_blocks(src_file, dst_file):
    """读取srt文件，随机保留中间的10个字幕块，写入新文件"""
    with open(src_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    blocks = split_srt_blocks(lines)

    if len(blocks) <= 10:
        selected_blocks = blocks  # 不够10条就全保留
    else:
        start_idx = random.randint(0, len(blocks) - 10)
        selected_blocks = blocks[start_idx:start_idx + 10]

    # 写入目标文件
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    with open(dst_file, 'w', encoding='utf-8') as f:
        for i, block in enumerate(selected_blocks, 1):
            f.write(f"{i}\n")  # 重新编号
            f.writelines(block[1:])  # 保留时间戳和内容
            f.write("\n")

def copy_with_processing(src_path, dst_path):
    """复制文件，若为srt则特殊处理"""
    if src_path.endswith(".srt"):
        save_random_10_srt_blocks(src_path, dst_path)
    else:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)

# 遍历源目录
for dirpath, _, filenames in os.walk(src_root):
    for filename in filenames:
        src_path = os.path.join(dirpath, filename)
        relative_path = os.path.relpath(src_path, src_root)
        dst_path = os.path.join(dst_root, relative_path)
        copy_with_processing(src_path, dst_path)

print("✅ 所有文件已复制完成，SRT文件已随机截取中间10行。")
