"""
References: VBench script
- https://github.com/Vchitect/VBench/blob/master/scripts/cal_final_score.py
- https://github.com/Vchitect/VBench/blob/master/scripts/constant.py
"""
import io
import os
import json
import zipfile
import argparse
import pprint

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


TASK_INFO = [
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "dynamic degree",
    "aesthetic quality",
    "imaging quality",
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency"
    ]

TASK_INFO_I2V = [
    "Video-Text Camera Motion",
    "Video-Image Subject Consistency",
    "Video-Image Background Consistency",
    "Subject Consistency",
    "Background Consistency",
    "Motion Smoothness",
    "Dynamic Degree",
    "Aesthetic Quality",
    "Imaging Quality"
    ]

DIM_WEIGHT = {
    "subject consistency":1,
    "background consistency":1,
    "temporal flickering":1,
    "motion smoothness":1,
    "aesthetic quality":1,
    "imaging quality":1,
    "dynamic degree":0.5,
    "object class":1,
    "multiple objects":1,
    "human action":1,
    "color":1,
    "spatial relationship":1,
    "scene":1,
    "appearance style":1,
    "temporal style":1,
    "overall consistency":1
}

DIM_WEIGHT_I2V = {
    "Video-Text Camera Motion": 0.1,
    "Video-Image Subject Consistency": 1,
    "Video-Image Background Consistency": 1,
    "Subject Consistency": 1,
    "Background Consistency": 1,
    "Motion Smoothness": 1,
    "Dynamic Degree": 0.5,
    "Aesthetic Quality": 1,
    "Imaging Quality": 1,
}


NORMALIZE_DIC = {
    "subject consistency": {"Min": 0.1462, "Max": 1.0},
    "background consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal flickering": {"Min": 0.6293, "Max": 1.0},
    "motion smoothness": {"Min": 0.706, "Max": 0.9975},
    "dynamic degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic quality": {"Min": 0.0, "Max": 1.0},
    "imaging quality": {"Min": 0.0, "Max": 1.0},
    "object class": {"Min": 0.0, "Max": 1.0},
    "multiple objects": {"Min": 0.0, "Max": 1.0},
    "human action": {"Min": 0.0, "Max": 1.0},
    "color": {"Min": 0.0, "Max": 1.0},
    "spatial relationship": {"Min": 0.0, "Max": 1.0},
    "scene": {"Min": 0.0, "Max": 0.8222},
    "appearance style": {"Min": 0.0009, "Max": 0.2855},
    "temporal style": {"Min": 0.0, "Max": 0.364},
    "overall consistency": {"Min": 0.0, "Max": 0.364}
}

NORMALIZE_DIC_I2V = {
    "Video-Text Camera Motion" :{"Min": 0.0, "Max":1.0 },
    "Video-Image Subject Consistency":{"Min": 0.1462, "Max": 1.0},
    "Video-Image Background Consistency":{"Min": 0.2615, "Max":1.0 },
    "Subject Consistency":{"Min": 0.1462, "Max": 1.0},
    "Background Consistency":{"Min": 0.2615, "Max": 1.0 },
    "Motion Smoothness":{"Min": 0.7060, "Max": 0.9975},
    "Dynamic Degree":{"Min": 0.0, "Max": 1.0},
    "Aesthetic Quality":{"Min": 0.0, "Max": 1.0},
    "Imaging Quality":{"Min": 0.0, "Max": 1.0},
}

SEMANTIC_WEIGHT = 1
QUALITY_WEIGHT = 4
I2V_WEIGHT = 1.0
I2V_QUALITY_WEIGHT = 1.0

QUALITY_LIST = [ 
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "aesthetic quality",
    "imaging quality",
    "dynamic degree",]

SEMANTIC_LIST = [
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency"
]

I2V_LIST = [
    "Video-Text Camera Motion",
    "Video-Image Subject Consistency",
    "Video-Image Background Consistency",
]

I2V_QUALITY_LIST = [
    "Subject Consistency",
    "Background Consistency",
    "Motion Smoothness",
    "Dynamic Degree",
    "Aesthetic Quality",
    "Imaging Quality",
]

I2VKEY={
    "camera_motion":"Video-Text Camera Motion",
    "i2v_subject":"Video-Image Subject Consistency",
    "i2v_background":"Video-Image Background Consistency",
    "subject_consistency":"Subject Consistency",
    "background_consistency":"Background Consistency",
    "motion_smoothness":"Motion Smoothness",
    "dynamic_degree":"Dynamic Degree",
    "aesthetic_quality":"Aesthetic Quality",
    "imaging_quality":"Imaging Quality",
    }


def _iter_result_jsons(json_files_path):
    """Yield every `*_eval_results.json`-style dict under json_files_path (one level of subdirs)."""
    for file in os.listdir(json_files_path):
        if file.startswith('.') or file.startswith('__'):
            print(f"Skip the file: {file}")
            continue
        cur_file = os.path.join(json_files_path, file)
        if os.path.isdir(cur_file):
            for subfile in os.listdir(cur_file):
                if subfile.endswith(".json"):
                    with open(os.path.join(cur_file, subfile)) as ff:
                        cur_json = json.load(ff)
                        if isinstance(cur_json, dict):
                            yield cur_json
        elif cur_file.endswith('json'):
            with open(cur_file) as ff:
                cur_json = json.load(ff)
                if isinstance(cur_json, dict):
                    yield cur_json


def submission_i2v(json_files_path):
    """VBench-I2V counterpart of submission(): result keys are mapped through I2VKEY
    (e.g. 'i2v_subject' -> 'Video-Image Subject Consistency'), as in the official
    scripts/cal_i2v_final_score.py. Missing leaderboard dims default to 0."""
    upload_data = {}
    for cur_json in _iter_result_jsons(json_files_path):
        for key in cur_json:
            if key not in I2VKEY:
                print(f"Skip non-I2V dimension: {key}")
                continue
            upload_data[I2VKEY[key]] = cur_json[key][0]
    for key in TASK_INFO_I2V:
        if key not in upload_data:
            upload_data[key] = 0
    return upload_data


def get_nomalized_score_i2v(upload_data):
    normalized_score = {}
    for key in TASK_INFO_I2V:
        min_val = NORMALIZE_DIC_I2V[key]['Min']
        max_val = NORMALIZE_DIC_I2V[key]['Max']
        normalized_score[key] = (upload_data[key] - min_val) / (max_val - min_val)
        normalized_score[key] = normalized_score[key] * DIM_WEIGHT_I2V[key]
    return normalized_score


def get_i2v_quality_score(normalized_score):
    quality_score = [normalized_score[key] for key in I2V_QUALITY_LIST]
    return sum(quality_score) / sum([DIM_WEIGHT_I2V[i] for i in I2V_QUALITY_LIST])


def get_i2v_score(normalized_score):
    i2v_score = [normalized_score[key] for key in I2V_LIST]
    return sum(i2v_score) / sum([DIM_WEIGHT_I2V[i] for i in I2V_LIST])


def get_i2v_final_score(quality_score, i2v_score):
    return (quality_score * I2V_QUALITY_WEIGHT + i2v_score * I2V_WEIGHT) / (I2V_QUALITY_WEIGHT + I2V_WEIGHT)


def submission(json_files_path):
    upload_data = {}
    # load your score
    for file in os.listdir(json_files_path):
        if file.startswith('.') or file.startswith('__'):
            print(f"Skip the file: {file}")
            continue
        cur_file = os.path.join(json_files_path, file)
        if os.path.isdir(cur_file):
            for subfile in os.listdir(cur_file):
                if subfile.endswith(".json"):
                    with open(os.path.join(cur_file, subfile)) as ff:
                        cur_json = json.load(ff)
                        if isinstance(cur_json, dict):
                            for key in cur_json:
                                upload_data[key.replace('_',' ')] = cur_json[key][0]
        elif cur_file.endswith('json'):
            # print(f'{cur_file=}')
            with open(cur_file) as ff:
                cur_json = json.load(ff)
                if isinstance(cur_json, dict):
                    for key in cur_json:
                        upload_data[key.replace('_',' ')] = cur_json[key][0]
        
        for key in TASK_INFO:
            if key not in upload_data:
                upload_data[key] = 0
    return upload_data


def get_nomalized_score(upload_data):
    # get the normalize score
    normalized_score = {}
    for key in TASK_INFO:
        min_val = NORMALIZE_DIC[key]['Min']
        max_val = NORMALIZE_DIC[key]['Max']
        normalized_score[key] = (upload_data[key] - min_val) / (max_val - min_val)
        normalized_score[key] = normalized_score[key] * DIM_WEIGHT[key]
    return normalized_score


def get_quality_score(normalized_score):
    quality_score = []
    for key in QUALITY_LIST:
        quality_score.append(normalized_score[key])
    quality_score = sum(quality_score)/sum([DIM_WEIGHT[i] for i in QUALITY_LIST])
    return quality_score


def get_semantic_score(normalized_score):
    semantic_score = []
    for key in SEMANTIC_LIST:
        semantic_score.append(normalized_score[key])
    semantic_score  = sum(semantic_score)/sum([DIM_WEIGHT[i] for i in SEMANTIC_LIST ])
    return semantic_score


def get_final_score(quality_score,semantic_score):
    return (quality_score * QUALITY_WEIGHT + semantic_score * SEMANTIC_WEIGHT) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Aggregate official VBench dimension scores.')
    parser.add_argument(
        '--input', '--json_files_path', dest='json_files_path', required=True,
        help='Directory containing the evaluation result JSON files.')
    parser.add_argument(
        '--output', default=None,
        help='Optional output JSON file for the aggregate and dimension scores.')
    parser.add_argument(
        '--mode', type=str, default='t2v', choices=['t2v', 'i2v'],
        help="'t2v': VBench (quality + semantic); 'i2v': VBench-I2V (quality + i2v).")
    args = parser.parse_args(argv)
    if not os.path.isdir(args.json_files_path):
        parser.error('--input must be an existing directory of evaluation result JSON files')

    if args.mode == 'i2v':
        upload_dict = submission_i2v(args.json_files_path)
        print(f"your submission info: \n{upload_dict} \n")
        pprint.pprint(upload_dict)
        normalized_score = get_nomalized_score_i2v(upload_dict)
        quality_score = get_i2v_quality_score(normalized_score)
        i2v_score = get_i2v_score(normalized_score)
        final_score = get_i2v_final_score(quality_score, i2v_score)
        print('+------------------|------------------+')
        print(f'|     Quality Score|{quality_score}|')
        print(f'|         I2V Score|{i2v_score}|')
        print(f'|       Total Score|{final_score}|')
        print('+------------------|------------------+')
        category_scores = {'quality_score': quality_score, 'i2v_score': i2v_score}
    else:
        upload_dict = submission(args.json_files_path)
        print(f"your submission info: \n{upload_dict} \n")
        pprint.pprint(upload_dict)
        normalized_score = get_nomalized_score(upload_dict)
        quality_score = get_quality_score(normalized_score)
        semantic_score = get_semantic_score(normalized_score)
        final_score = get_final_score(quality_score, semantic_score)
        print('+------------------|------------------+')
        print(f'|     quality score|{quality_score}|')
        print(f'|    semantic score|{semantic_score}|')
        print(f'|       total score|{final_score}|')
        print('+------------------|------------------+')
        category_scores = {'quality_score': quality_score, 'semantic_score': semantic_score}

    result = {
        'mode': args.mode,
        'dimensions': upload_dict,
        'normalized_dimensions': normalized_score,
        **category_scores,
        'total_score': final_score,
    }
    if args.output:
        output_path = os.path.abspath(os.path.expanduser(args.output))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as stream:
            json.dump(result, stream, indent=2)
        print(f'Saved aggregate scores to {output_path}')
    return result


if __name__ == "__main__":
    main()
