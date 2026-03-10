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
        self.task_dict          = defaultdict(lambda: defaultdict(float))
        self.task_detail_dict   = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.task_reg_dict      = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        self.task_counts        = defaultdict(lambda: defaultdict(int))
        self.task_detail_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.task_reg_counts    = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    def clear(self):
        self.task_dict.clear()
        self.task_detail_dict.clear()
        self.task_reg_dict.clear()
        self.task_counts.clear()
        self.task_detail_counts.clear()
        self.task_reg_counts.clear()

    @staticmethod
    def _to_env_tensor(value: tp.Any, num_envs: int) -> torch.Tensor | None:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().float().reshape(-1)
            if tensor.numel() == num_envs:
                return tensor
            if tensor.numel() == 1:
                return tensor.repeat(num_envs)
            return None
        if isinstance(value, (float, int)):
            return torch.full((num_envs,), float(value), dtype=torch.float32)
        return None

    #############################################################################################
    def _update_field_rew(self, task, sub_task: str, reward: float, reward_dict: tp.Dict[str, tp.Any]):
        self.task_dict[task][sub_task] += float(reward)
        self.task_counts[task][sub_task] += 1

        for key, value in reward_dict.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().float().mean().item()
            self.task_detail_dict[task][sub_task][key] += float(value)
            self.task_detail_counts[task][sub_task][key] += 1

    def _update_field_reg(self, task, sub_task: str, reg_value: tp.Dict[str, tp.Any]):
        for key, value in reg_value.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().float().mean().item()
            self.task_reg_dict[task][sub_task][key] += float(value)
            self.task_reg_counts[task][sub_task][key] += 1

    #############################################################################################
    def update(
        self,
        task: str,
        mode: str,
        log_index: dict[str, list[int]],
        reward: torch.Tensor,
        rew_dict: tp.Dict[str, tp.Any],
        reg_dict: tp.Dict[str, tp.Any],
    ):
        reward_1d = reward.detach().float().reshape(-1)
        num_envs = reward_1d.numel()

        # 统一处理 RANDOM 和 LIST
        for sub_task, indices in log_index.items():

            reward_extract = reward_1d[indices].mean().item()
            rew_dict_extract = {key: self._to_env_tensor(value, num_envs)[indices] for key, value in rew_dict.items()}
            reg_dict_extract = {key: self._to_env_tensor(value, num_envs)[indices] for key, value in reg_dict.items()}

            self._update_field_rew(task, sub_task, reward_extract, rew_dict_extract)
            self._update_field_reg(task, sub_task, reg_dict_extract)
    
    #############################################################################################
    def _get_sub_task_metrics(self, task_name: str, sub_task: str) -> tp.Dict[str, float]:
        count      = self.task_counts[task_name][sub_task]
        reward_sum = self.task_dict[task_name][sub_task]

        # episode return 不计算平均
        metrics: dict[str, float] = {"episode_return": reward_sum}

        for key, value in self.task_detail_dict[task_name][sub_task].items():
            if "reward" in key.lower() or "rew" in key.lower():
                metrics[key] = value
            else:
                detail_count = self.task_detail_counts[task_name][sub_task][key]
                metrics[key] = value / detail_count if detail_count > 0 else 0.0

        for key, value in self.task_reg_dict[task_name][sub_task].items():
            if "reward" in key.lower() or "rew" in key.lower():
                metrics[key] = value
            else:
                reg_count = self.task_reg_counts[task_name][sub_task][key]
                metrics[key] = value / reg_count if reg_count > 0 else 0.0

        return metrics


    # 获取所有 task 的所有指标 + 平均 episode reward
    def get_tasks_metrics(self) -> tp.Dict[str, float]:
        metrics = {}
        
        for task, sub_task_dict in self.task_dict.items():
            
            # 记录 task 中所有 list 评估下的 (mean (sub_task mean))
            task_list_mean = 0.0
            list_count = 0
            reg_list = ['action_rate', 'feet_slip_penalty']

            for sub_task in sorted(sub_task_dict.keys()):
                sub_task_metrics = self._get_sub_task_metrics(task, sub_task)

                # 记录 task(list+random) 的 sub_task 平均奖励
                episode_return = sub_task_metrics.get("episode_return", 0.0)
                metrics[f"{task}/{sub_task}_episode_return"] = episode_return

                # 记录 task(list) 的平均 regularization metrics
                for reg_key in reg_list:
                    metrics[f"{task}/list_{reg_key}"] = sub_task_metrics.get(reg_key, 0.0)

                if 'random' not in sub_task.lower():
                    task_list_mean += episode_return
                    list_count += 1

            metrics[f"{task}/list_mean_episode_return"] = task_list_mean / list_count

        return metrics
