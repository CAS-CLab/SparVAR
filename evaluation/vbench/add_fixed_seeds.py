import json
import random
import os

def add_seeds_to_json(input_path, output_path, global_seed=42):
    """
    读取 VBench prompt json，为每个条目添加一个固定的、不重复的随机 seed。
    """
    
    # 1. 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return

    print(f"Reading from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_prompts = len(data)
    print(f"Total prompts found: {total_prompts}")

    # 2. 设置全局随机种子，确保每次运行结果一致 (Determinism)
    random.seed(global_seed)

    # 3. 生成不重复的随机 Seed 列表 (Uniqueness)
    # 我们从一个足够大的范围内 (0 到 1亿) 抽取 total_prompts 个不重复的数字
    # random.sample 保证了抽取的数字在列表中是唯一的
    seed_pool_size = 10000
    if total_prompts > seed_pool_size:
        raise ValueError("Prompt count exceeds seed pool size!")
        
    unique_seeds = random.sample(range(seed_pool_size), total_prompts)

    # 4. 将 Seed 分配给每个 Prompt
    for idx, item in enumerate(data):
        item['seed'] = unique_seeds[idx]

    # 5. 保存结果
    print(f"Writing to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        # indent=2 保持格式美观，ensure_ascii=False 防止中文乱码
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("Done! All prompts now have fixed seeds.")
    print(f"Global Seed used: {global_seed}")

if __name__ == "__main__":
    import argparse
    # Defaults reproduce the original T2V behavior; pass the VBench-I2V manifest for the I2V suite:
    #   python evaluation/vbench/add_fixed_seeds.py \
    #       --input_json evaluation/vbench/VBench_I2V_full_info.json \
    #       --output_json evaluation/vbench/VBench_I2V_full_info_fixed_seed.json
    parser = argparse.ArgumentParser(description='Add a fixed, unique per-prompt seed to a VBench prompt json')
    parser.add_argument('--input_json', type=str, default="evaluation/vbench/VBench_rewrited_prompt.json")
    parser.add_argument('--output_json', type=str, default="evaluation/vbench/VBench_rewrited_prompt_fixed_seed.json")
    # 设置全局 Seed (只要这个数字不变，生成的 json 文件内容永远不变)
    parser.add_argument('--global_seed', type=int, default=2025)
    cli = parser.parse_args()

    # 确保目录存在
    os.makedirs(os.path.dirname(cli.output_json), exist_ok=True)

    add_seeds_to_json(cli.input_json, cli.output_json, cli.global_seed)
