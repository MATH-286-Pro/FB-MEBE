import dataclasses
import typing as tp
import numpy as np
from pathlib import Path
import torch
from collections import defaultdict


class TRAIN_METRICS_CLASS:
    def __init__(self):
        self._metrics = defaultdict(float)
        self._counts = defaultdict(int)

    def update(self, metrics: dict, key_prefix: str = ""):
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            full_key = key_prefix + key
            self._metrics[full_key] += value
            self._counts[full_key] += 1

    def clear(self):
        self._metrics.clear()
        self._counts.clear()

    @property
    def mean(self) -> dict:
        return {key: value / self._counts[key] for key, value in self._metrics.items() if self._counts[key] > 0}
    
class EVAL_METRICS_CLASS:
    def __init__(self) -> None:
        """
        self.task_dict = {
            "[1.0,0.0,0.0]": 250,
            "[0.5,0.0,0.0]": 200,
            "[0.0,0.0,0.0]": 250
        }
        """
        self.task_dict        = defaultdict(float)                         # 一级字典 One level Dictionary
        self.task_detail_dict = defaultdict(lambda: defaultdict(float))    # 二级字典 Two level Dictionary
        self.task_reg_dict    = defaultdict(lambda: defaultdict(float))    # 二级字典 Two level Dictionary
        
        self.task_counts        = defaultdict(int)                         # 一级计数
        self.task_detail_counts = defaultdict(lambda: defaultdict(int))    # 二级计数
        self.task_reg_counts    = defaultdict(lambda: defaultdict(int))    # 二级计数

    def clear(self):
        self.task_dict.clear()
        self.task_detail_dict.clear()
        self.task_reg_dict.clear()
        self.task_counts.clear()
        self.task_detail_counts.clear()
        self.task_reg_counts.clear()
    
    def update_rew(self, task_name: str, reward: torch.Tensor, reward_dict: tp.Dict[str, float]):

        # update prodected task reward
        self.task_dict[task_name] += reward.mean().item()
        self.task_counts[task_name] += 1

        # update reward details
        for key, value in reward_dict.items():
            self.task_detail_dict[task_name][key] += value
            self.task_detail_counts[task_name][key] += 1
    
    def update_reg(self, task_name: str, reg_value: tp.Dict[str, float]):
        for key, value in reg_value.items():
            self.task_reg_dict[task_name][key] += value
            self.task_reg_counts[task_name][key] += 1
    
    #############################################################################################
    
    # 获取一个 task 的 episode reward
    def get_task_episode_reward(self, task_name: str) -> float:
        count = self.task_counts[task_name]
        return self.task_dict[task_name]

    # 获取一个 task 的所有指标
    def get_task_metrics(self, task_name: str) -> tp.Dict[str, float]:
        dict = {"episode_reward": self.get_task_episode_reward(task_name)}
        
        # 计算 detail 指标的平均值 (reward 和 rew 相关的键不平均，直接返回累加值)
        for key, value in self.task_detail_dict[task_name].items():
            if "reward" in key.lower() or "rew" in key.lower():
                dict[key] = value  # 不做平均，返回累加值
            else:
                count = self.task_detail_counts[task_name][key]
                dict[key] = value / count if count > 0 else 0.0
        
        # 计算 reg 指标的平均值 (reward 和 rew 相关的键不平均，直接返回累加值)
        for key, value in self.task_reg_dict[task_name].items():
            if "reward" in key.lower() or "rew" in key.lower():
                dict[key] = value  # 不做平均，返回累加值
            else:
                count = self.task_reg_counts[task_name][key]
                dict[key] = value / count if count > 0 else 0.0
        
        return dict

    # 获取所有 task 的平均 episode reward
    def get_tasks_episode_reward(self) -> float:
        mean_reward = 0.0
        for task, reward in self.task_dict.items():
            mean_reward += reward
        mean_reward = mean_reward / len(self.task_dict)
        return mean_reward

    # 获取所有 task 的所有指标 + 平均 episode reward
    def get_tasks_metrics(self) -> tp.Dict[str, float]:
        dict = {}
        
        for task in self.task_dict.keys():
            task_metrics = self.get_task_metrics(task)
            for key, value in task_metrics.items():
                dict[f"{task}/{key}"] = value
                if key == "episode_reward":
                    dict[f"eval/{task}_episode_reward"] = value

        dict["eval/mean_episode_reward"] = self.get_tasks_episode_reward()
        
        return dict
