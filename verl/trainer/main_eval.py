# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm
import os
import json

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local
from verl.utils.reward_score import default_compute_score



# def save_json(x, save_path):
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     with open(save_path, 'w') as f:
#         json.dump(x, f, indent=4, ensure_ascii=False)
#     print('saved to: ', save_path)

def save_json(x, save_path):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(convert(x), f, indent=4, ensure_ascii=False)
    print('saved to: ', save_path)


 
@ray.remote
def process_item(reward_fn, data_source, response_lst, reward_data):
    ground_truth = reward_data["ground_truth"]
    score_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]
    return data_source, np.mean(score_lst)


 
@ray.remote
def process_item_2(reward_fn, data_source, response_lst, reward_data):
    ground_truth = reward_data["ground_truth"]
    score_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]
    return data_source, score_lst




@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path, use_shm=config.data.get("use_shm", False))
    dataset = pd.read_parquet(local_path)
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    total = len(dataset)
    print('total: ', total)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=config.ray_init.num_cpus)

    # evaluate test_score based on data source
    data_source_reward = defaultdict(list)

    # compute_score = get_custom_reward_fn(config)     
    compute_score = default_compute_score

    # Create remote tasks   
    # remote_tasks = [
    #     process_item.remote(compute_score, data_sources[i], responses[i], reward_model_data[i]) for i in range(total)
    # ]

    remote_tasks = [
        process_item_2.remote(compute_score, data_sources[i], responses[i], reward_model_data[i]) for i in range(total)
    ]

    # Process results as they come in
    eval_scores = [] # 新增：用于记录每个数据点的 correctness
    with tqdm(total=total) as pbar:
        while len(remote_tasks) > 0:
            # Use ray.wait to get completed tasks
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                # data_source, score = ray.get(result_id)           
                # data_source_reward[data_source].append(score)     
                data_source, score_lst = ray.get(result_id)
                data_source_reward[data_source].append(score_lst)
                eval_scores.append(score_lst) 
                pbar.update(1)

     
    # metric_dict = {}
    # for data_source, rewards in data_source_reward.items():
    #     metric_dict[f"test_score/{data_source}"] = np.mean(rewards)


    ### Save correctness for each data point to eval_results.json
    eval_results = []
    for i in range(total):
        data_item = dataset.iloc[i].to_dict()
        scores = eval_scores[i]
        data_item['scores'] = scores
        data_item['pass@1'] = np.mean(scores)
        k = len(scores)
        if k > 1:
            data_item[f'pass@{k}'] = np.max(scores)
        eval_results.append(data_item)
        
    eval_results_path = os.path.join(os.path.dirname(config.data.path), 'eval_results.json')
    save_json(eval_results, eval_results_path)


    ### Get metrics for each source separately
    metric_dict = {}
    for data_source, rewards in data_source_reward.items():
        num_data = len(rewards)
        k = len(rewards[0])   # 
        print(f'{data_source}: {num_data} data, each data has {k} generations')
        
        if k == 1:
            pass_1_list = []
            for score_lst in rewards:
                pass_1 = np.mean(score_lst)
                pass_1_list.append(pass_1)
            avg_pass_1 = np.mean(pass_1_list)
            metric_dict[data_source] = {
                'pass@1': avg_pass_1,
            }
        elif k > 1:
            pass_1_list = []
            pass_k_list = []
            for score_lst in rewards:
                pass_1 = np.mean(score_lst)
                pass_1_list.append(pass_1)

                pass_k = np.max(score_lst)
                if pass_k > 0:
                    pass_k = 1.0
                pass_k_list.append(pass_k)
            avg_pass_1 = np.mean(pass_1_list)
            avg_pass_k = np.mean(pass_k_list)
            metric_dict[data_source] = {
                'pass@1': avg_pass_1,
                f"pass@{k}": avg_pass_k,
            }
        else:
            raise ValueError
    

    # save
    save_path = os.path.join(os.path.dirname(config.data.path), 'metric.json')
    save_json(metric_dict, save_path)
    
    # print
    for k, v in metric_dict.items():
        print(f'{k}: {v}')

    # print(metric_dict)     


if __name__ == "__main__":
    main()

