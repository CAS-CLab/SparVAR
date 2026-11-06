import argparse
import json
import os
import os.path as osp
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = ('.bmp', '.jpeg', '.jpg', '.png', '.webp')


def parse_args():
    parser = argparse.ArgumentParser(description="DPG-Bench evaluation.")
    parser.add_argument(
        "--image-root-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--csv",
        type=str,
        default='evaluation/dpg_bench/dpg_bench.csv',
    )
    parser.add_argument(
        "--res-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pic-num",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--vqa-model",
        type=str,
        default='mplug',
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of independent evaluation shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Index of the independent evaluation shard.",
    )
    parser.add_argument(
        "--result-json",
        type=str,
        default=None,
        help="Path used to save raw scores for this evaluation shard.",
    )
    parser.add_argument(
        "--merge-result-jsons",
        type=str,
        nargs='+',
        default=None,
        help="Merge raw score files without running VQA inference.",
    )

    args = parser.parse_args()
    return args


class MPLUG(torch.nn.Module):
    def __init__(self, ckpt='damo/mplug_visual-question-answering_coco_large_en', device='gpu'):
        super().__init__()
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        self.pipeline_vqa = pipeline(Tasks.visual_question_answering, model=ckpt, device=device)

    def vqa(self, image, question):
        input_vqa = {'image': image, 'question': question}
        result = self.pipeline_vqa(input_vqa)
        return result['text']

def prepare_dpg_data(args):
    previous_id = ''
    current_id = ''
    question_dict = dict()
    category_count = defaultdict(int)
    # 'item_id', 'text', 'keywords', 'proposition_id', 'dependency', 'category_broad', 'category_detailed', 'tuple', 'question_natural_language'
    data = pd.read_csv(args.csv)
    # read_csv already consumes the header; row 0 is a real question.
    for _, line in data.iterrows():
        current_id = line.item_id
        qid = int(line.proposition_id)
        dependency_list_str = line.dependency.split(',')
        dependency_list_int = []
        for d in dependency_list_str:
            d_int = int(d.strip())
            dependency_list_int.append(d_int)

        if current_id == previous_id:
            question_dict[current_id]['qid2tuple'][qid] = line.tuple
            question_dict[current_id]['qid2dependency'][qid] = dependency_list_int
            question_dict[current_id]['qid2question'][qid] = line.question_natural_language
        else:
            question_dict[current_id] = dict(
                qid2tuple={qid: line.tuple},
                qid2dependency={qid: dependency_list_int},
                qid2question={qid: line.question_natural_language})
        
        category = line.question_natural_language.split('(')[0].strip()
        category_count[category] += 1
        
        previous_id = current_id

    return question_dict

def crop_image(input_image, crop_tuple=None):
    if crop_tuple is None:
        return input_image

    cropped_image = input_image.crop((crop_tuple[0], crop_tuple[1], crop_tuple[2], crop_tuple[3]))

    return cropped_image

def compute_dpg_one_sample(args, question_dict, image_path, vqa_model, resolution):
    generated_image = Image.open(image_path)
    crop_tuples_list = [
        (0,0,resolution,resolution),
        (resolution, 0, resolution*2, resolution),
        (0, resolution, resolution, resolution*2),
        (resolution, resolution, resolution*2, resolution*2),
    ]

    crop_tuples = crop_tuples_list[:args.pic_num]
    key = osp.basename(image_path).split('.')[0]
    value = question_dict.get(key, None)
    qid2tuple = value['qid2tuple']
    qid2question = value['qid2question']
    qid2dependency = value['qid2dependency']

    qid2answer = dict()
    qid2scores = dict()
    qid2validity = dict()

    scores = []
    for crop_tuple in crop_tuples:
        cropped_image = crop_image(generated_image, crop_tuple)
        for id, question in qid2question.items():
            answer = vqa_model.vqa(cropped_image, question)
            qid2answer[id] = answer
            qid2scores[id] = float(answer == 'yes')
            with open(args.res_path.replace('.txt', '_detail.txt'), 'a') as f:
                f.write(image_path + ', ' + str(crop_tuple) + ', ' + question + ', ' + answer + '\n')
        qid2scores_orig = qid2scores.copy()

        for id, parent_ids in qid2dependency.items():
            # zero-out scores if parent questions are answered 'no'
            any_parent_answered_no = False
            for parent_id in parent_ids:
                if parent_id == 0:
                    continue
                if qid2scores[parent_id] == 0:
                    any_parent_answered_no = True
                    break
            if any_parent_answered_no:
                qid2scores[id] = 0
                qid2validity[id] = False
            else:
                qid2validity[id] = True

        score = sum(qid2scores.values()) / len(qid2scores)
        scores.append(score)
    average_score = sum(scores) / len(scores)
    with open(args.res_path, 'a') as f:
        f.write(image_path + ', ' + ', '.join(str(i) for i in scores) + ', ' + str(average_score) + '\n')
  
    return average_score, qid2tuple, qid2scores_orig


def build_dpg_output(model_id, image_root_path, res_path, dpg_scores, category2scores):
    if not dpg_scores:
        raise ValueError("No valid DPG-Bench scores were collected.")

    category2scores_l1 = defaultdict(list)
    for category, scores in category2scores.items():
        l1_category = category.split('-')[0].strip()
        category2scores_l1[l1_category].extend(scores)

    output = f'Model: {model_id}\n'
    output += 'L1 category scores:\n'
    for l1_category in sorted(category2scores_l1):
        output += f'\t{l1_category}: {np.mean(category2scores_l1[l1_category]) * 100}\n'

    output += 'L2 category scores:\n'
    for category in sorted(category2scores):
        output += f'\t{category}: {np.mean(category2scores[category]) * 100}\n'

    output += f'Image path: {image_root_path}\n'
    output += f'Save results to: {res_path}\n'
    output += f'DPG-Bench score: {np.mean(dpg_scores) * 100}'
    return output


def save_result_json(args, model_id, dpg_scores, category2scores):
    if args.result_json is None:
        return

    result_dir = osp.dirname(args.result_json)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    payload = {
        'model_id': model_id,
        'image_root_path': osp.abspath(args.image_root_path),
        'num_shards': args.num_shards,
        'shard_index': args.shard_index,
        'dpg_scores': [float(score) for score in dpg_scores],
        'category2scores': {
            category: [float(score) for score in scores]
            for category, scores in category2scores.items()
        },
    }
    with open(args.result_json, 'w') as f:
        json.dump(payload, f)


def merge_result_jsons(args):
    if args.res_path is None:
        raise ValueError("--res-path is required when merging result JSON files.")

    dpg_scores = []
    category2scores = defaultdict(list)
    shard_indices = set()
    expected_num_shards = None
    model_id = None
    image_root_path = None

    for result_json in args.merge_result_jsons:
        with open(result_json, 'r') as f:
            payload = json.load(f)

        if expected_num_shards is None:
            expected_num_shards = payload['num_shards']
            model_id = payload['model_id']
            image_root_path = payload['image_root_path']
        elif payload['num_shards'] != expected_num_shards:
            raise ValueError("Result JSON files use different numbers of shards.")
        elif payload['image_root_path'] != image_root_path:
            raise ValueError("Result JSON files use different image roots.")

        shard_index = payload['shard_index']
        if shard_index in shard_indices:
            raise ValueError(f"Duplicate shard index: {shard_index}")
        shard_indices.add(shard_index)
        dpg_scores.extend(payload['dpg_scores'])
        for category, scores in payload['category2scores'].items():
            category2scores[category].extend(scores)

    expected_shard_indices = set(range(expected_num_shards))
    if shard_indices != expected_shard_indices:
        missing = sorted(expected_shard_indices - shard_indices)
        raise ValueError(f"Missing result shards: {missing}")

    result_dir = osp.dirname(args.res_path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    output = build_dpg_output(
        model_id=model_id,
        image_root_path=image_root_path,
        res_path=args.res_path,
        dpg_scores=dpg_scores,
        category2scores=category2scores,
    )
    with open(args.res_path, 'w') as f:
        f.write(output + '\n')
    print(output)


def main():
    args = parse_args()
    if args.merge_result_jsons is not None:
        merge_result_jsons(args)
        return

    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards).")

    from accelerate import Accelerator
    from accelerate.utils import gather_object

    accelerator = Accelerator()

    question_dict = prepare_dpg_data(args)

    timestamp = time.time()
    time_array = time.localtime(timestamp)
    time_style = time.strftime("%Y%m%d-%H%M%S", time_array)
    if args.res_path is None:
        args.res_path = osp.join(args.image_root_path, f'dpg-bench_{time_style}_results.txt')
    if accelerator.is_main_process:
        with open(args.res_path, 'w') as f:
            pass
        with open(args.res_path.replace('.txt', '_detail.txt'), 'w') as f:
            pass
    accelerator.wait_for_everyone()

    device = str(accelerator.device)
    if args.vqa_model == 'mplug':
        # DPG_MPLUG_MODEL may point at a local snapshot dir (e.g. the MODELSCOPE_CACHE copy) so the
        # scorer works offline; the default model id needs a modelscope hub round-trip
        vqa_model = MPLUG(ckpt=os.environ.get('DPG_MPLUG_MODEL', 'damo/mplug_visual-question-answering_coco_large_en'), device=device)
    else:
        raise NotImplementedError
    vqa_model = accelerator.prepare(vqa_model)
    vqa_model = getattr(vqa_model, 'module', vqa_model) 

    filename_list = sorted(
        fn for fn in os.listdir(args.image_root_path)
        if fn.lower().endswith(IMAGE_EXTENSIONS)
        and osp.isfile(osp.join(args.image_root_path, fn))
    )
    filename_list = filename_list[args.shard_index::args.num_shards]
    num_each_rank = len(filename_list) / accelerator.num_processes
    local_rank = accelerator.process_index
    local_filename_list = filename_list[round(local_rank * num_each_rank) : round((local_rank + 1) * num_each_rank)]

    local_scores = []
    local_category2scores = defaultdict(list)
    model_id = osp.basename(args.image_root_path.rstrip(os.sep))
    print(
        f'Start to conduct evaluation of {model_id}, '
        f'shard {args.shard_index + 1}/{args.num_shards}'
    )
    for fn in tqdm(local_filename_list):
        image_path = osp.join(args.image_root_path, fn)
        try:
            # compute score of one sample
            score, qid2tuple, qid2scores = compute_dpg_one_sample(
                args=args, question_dict=question_dict, image_path=image_path, vqa_model=vqa_model, resolution=args.resolution)
            local_scores.append(score)
            
            # summarize scores by categoris
            for qid in qid2tuple.keys():
                category = qid2tuple[qid].split('(')[0].strip()
                qid_score = qid2scores[qid]
                local_category2scores[category].append(qid_score)

        except Exception as e:
            print('Failed filename:', fn, e)
            continue
    
    accelerator.wait_for_everyone()
    global_dpg_scores = gather_object(local_scores)
    gathered_category2scores = gather_object([dict(local_category2scores)])
    global_category2scores = defaultdict(list)
    for category2scores in gathered_category2scores:
        for category, scores in category2scores.items():
            global_category2scores[category].extend(scores)

    if accelerator.is_main_process:
        save_result_json(
            args=args,
            model_id=model_id,
            dpg_scores=global_dpg_scores,
            category2scores=global_category2scores,
        )
        output = build_dpg_output(
            model_id=model_id,
            image_root_path=args.image_root_path,
            res_path=args.res_path,
            dpg_scores=global_dpg_scores,
            category2scores=global_category2scores,
        )
        with open(args.res_path, 'a') as f:
            f.write(output + '\n')
        print(output)


if __name__ == "__main__":
    main()
