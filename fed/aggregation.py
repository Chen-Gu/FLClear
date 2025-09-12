# -*- coding: UTF-8 -*-
import copy
from collections import OrderedDict

import torch
from typing import List, Dict
import torch.nn as nn


def FedAvg(models, num_samples):
    total_samples = sum(num_samples)
    model_avg = copy.deepcopy(models[0])
    model_avg = {k: v for k, v in model_avg.items()}
    models = [{k: v for k, v in model.items()} for model in models]
    # 加权平均
    for key in model_avg:
        if model_avg[key].dtype == torch.long:  # 处理 torch.long 类型（如 num_batches_tracked）
            weighted_sum = 0.0
            for i in range(len(models)):
                weighted_sum += models[i][key].item() * (num_samples[i] / total_samples)
            model_avg[key] = torch.tensor(round(weighted_sum), dtype=torch.long)
        else:  # 处理浮点类型（weight, bias, running_mean, running_var）
            model_avg[key] = torch.zeros_like(model_avg[key])
            for i in range(len(models)):
                model_avg[key] += models[i][key] * (num_samples[i] / total_samples)
    return model_avg

def FedProximal(model, global_model, mu = 0.01):
    prox_term = 0.0
    if global_model is None:
        print(f'global_model is None!')
    else:
        for param, global_param in zip(model.parameters(), global_model.parameters()):
            prox_term += torch.norm(param - global_param) ** 2
    return (mu / 2) * prox_term


def quantize_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """对单个张量进行均匀量化"""
    if tensor.dtype == torch.long:
        return tensor  # 不量化 torch.long 类型（如 num_batches_tracked）
    t_min, t_max = tensor.min(), tensor.max()
    scale = (t_max - t_min) / (2 ** bits - 1) if t_max != t_min else 1.0
    zero_point = t_min
    quantized = torch.round((tensor - zero_point) / scale).clamp(0, 2 ** bits - 1)
    return zero_point + quantized * scale

def quantize_client(local_model_state: dict, global_model_state: dict, quantization_bits: int = 8) -> dict:
    """
    计算客户端模型与全局模型的权重差，量化权重差后加上原始全局模型权重，生成完整的量化模型权重。

    参数：
        local_model_state: 客户端本地模型的state_dict
        global_model_state: 全局模型的state_dict
        quantization_bits: 量化位数（默认8位）

    返回：
        原始全局模型权重加上量化后的权重差，组成的完整量化模型权重（state_dict格式）
    """
    # 计算权重差
    weight_diff = copy.deepcopy(local_model_state)
    for key in weight_diff:
        weight_diff[key] = local_model_state[key] - global_model_state[key]

    # 量化权重差
    quantized_diff = copy.deepcopy(weight_diff)
    for key in quantized_diff:
        quantized_diff[key] = quantize_tensor(weight_diff[key], quantization_bits)

    # 将量化后的权重差加到原始全局模型权重上
    quantized_model_state = copy.deepcopy(global_model_state)
    for key in quantized_model_state:
        quantized_model_state[key] += quantized_diff[key]

    return quantized_model_state


def fedadam_aggregate(global_model_dict, local_models, local_nums, server_state, eta=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    FedAdam server-side aggregation.
    参数:
        global_model_dict: 全局模型的 state_dict()
        local_models: 各客户端上传的模型参数列表（state_dict）
        local_nums: 每个客户端的样本数列表
        server_state: 服务器端保存的 m_t, v_t 状态
        eta: 学习率
        beta1: 一阶动量衰减因子
        beta2: 二阶动量衰减因子
        tau: 防止除零的小常数

    返回:
        updated_model: 聚合后的全局模型 state_dict
        server_state: 更新后的服务器状态
    """
    # 初始化 m_t, v_t
    if 'm_t' not in server_state:
        server_state['m_t'] = {
            k: torch.zeros_like(v, dtype=torch.float32)
            for k, v in global_model_dict.items()
            if torch.is_floating_point(v)
        }
    if 'v_t' not in server_state:
        server_state['v_t'] = {
            k: torch.zeros_like(v, dtype=torch.float32)
            for k, v in global_model_dict.items()
            if torch.is_floating_point(v)
        }
    if 't' not in server_state:
        server_state['t'] = 0  # 初始化时间步

        # 增量时间步
    server_state['t'] += 1
    t = server_state['t']
    non_trainable_keys = {'running_mean', 'running_var', 'num_batches_tracked'}
    # 先做平均聚合
    avg_model_dict = FedAvg(local_models, local_nums)

    updated_state_dict = {}

    for k in global_model_dict:
        global_param = global_model_dict[k]  # 聚合前的全局模型
        avg_param = avg_model_dict[k]

        if torch.is_floating_point(global_param) and not any(non_key in k for non_key in non_trainable_keys):
            #  可学习参数  使用 FedAdam 聚合
            delta = avg_param - global_param  #求平均更新梯度
            m = beta1 * server_state['m_t'][k] + (1 - beta1) * delta
            v = beta2 * server_state['v_t'][k] + (1 - beta2) * (delta ** 2)
            # v = server_state['v_t'][k] + (delta ** 2)
            # m_hat = m / (1 - beta1 ** t)
            # v_hat = v / (1 - beta2 ** t)

            update = eta * m / (torch.sqrt(v) + eps)
            updated_state_dict[k] = global_param + update
            server_state['m_t'][k] = m
            server_state['v_t'][k] = v

        else:
            # 非可训练参数：直接用 FedAvg 覆盖（如 BatchNorm running_mean）
            updated_state_dict[k] = avg_param


    return updated_state_dict, server_state


def FedAvgWithTrimming(
    models: List[Dict[str, torch.Tensor]],
    init_state_dict: Dict[str, torch.Tensor],
    num_samples: List[int],
    k: float = 0.4,
    device: str = 'cuda:4'
) -> Dict[str, torch.Tensor]:
    """
    FedAvg with Trimming：修剪任务向量，加权聚合，特别处理 torch.long 类型。

    参数：
        models: 客户端状态字典列表。
        init_state_dict: 初始全局模型状态字典。
        num_samples: 每个客户端的样本数。
        k: 修剪比例，保留 top-k% 参数（默认 0.4）。
        device: 计算设备（默认：'cuda:4'）。

    返回：
        合并后的状态字典。
    """
    # 验证输入
    if len(models) != len(num_samples):
        raise ValueError("models 和 num_samples 长度必须一致")
    total_samples = sum(num_samples)
    if total_samples == 0:
        raise ValueError("num_samples 总和必须大于 0")
    if not (0 < k <= 1):
        raise ValueError("k 必须在 (0, 1] 范围内")

    # 初始化合并状态字典
    merged_state_dict = copy.deepcopy(init_state_dict)
    merged_state_dict = {k: v.to(device) for k, v in merged_state_dict.items()}
    models = [{k: v.to(device) for k, v in model.items()} for model in models]

    # 计算任务向量
    task_vectors = []
    for model in models:
        task_vector = {}
        for key in model:
            task_vector[key] = model[key] - init_state_dict[key]
        task_vectors.append(task_vector)

    # 修剪任务向量
    trimmed_task_vectors = []
    for task_vector in task_vectors:
        trimmed_vector = copy.deepcopy(task_vector)
        for key in trimmed_vector:
            if trimmed_vector[key].requires_grad:  # 修剪学习参数
                flat_params = trimmed_vector[key].view(-1)
                abs_params = torch.abs(flat_params)
                num_params = flat_params.numel()
                num_to_keep = int(k * num_params)
                if num_to_keep == 0:
                    trimmed_vector[key] = torch.zeros_like(flat_params).reshape_as(trimmed_vector[key])
                    continue
                threshold = torch.kthvalue(abs_params, num_params - num_to_keep + 1).values
                mask = abs_params >= threshold
                trimmed_vector[key] = flat_params * mask.float()
                trimmed_vector[key] = trimmed_vector[key].reshape_as(task_vector[key])
            else:
                trimmed_vector[key] = task_vector[key]  # 非学习参数不修剪
        trimmed_task_vectors.append(trimmed_vector)

    # 加权聚合
    for key in merged_state_dict:
        if merged_state_dict[key].dtype == torch.long:  # 处理 num_batches_tracked
            weighted_sum = 0.0
            for i in range(len(trimmed_task_vectors)):
                weighted_sum += trimmed_task_vectors[i][key].item() * (num_samples[i] / total_samples)
            merged_state_dict[key] = torch.tensor(round(weighted_sum), dtype=torch.long, device=device)
        else:  # 处理浮点参数
            merged_state_dict[key] = torch.zeros_like(merged_state_dict[key])
            for i in range(len(trimmed_task_vectors)):
                merged_state_dict[key] += trimmed_task_vectors[i][key] * (num_samples[i] / total_samples)
            merged_state_dict[key] += init_state_dict[key]

    return merged_state_dict
def ties_merging(
        state_dicts,
        init_state_dict: Dict[str, torch.Tensor],
        k: float = 0.4,
        lambda_scale: float = 1.0,
        num_samples = None,
        device: str = 'cuda:4'
) -> Dict[str, torch.Tensor]:
    """
    使用 TIES-MERGING 方法合并多个微调模型，基于初始权重。

    参数：
        models: 微调后的 PyTorch 模型列表（nn.Module 实例）。
        init_state_dict: 预训练模型的状态字典。
        k: 保留的最大参数比例（默认：0.2，即保留 top 20%）。
        lambda_scale: 合并任务向量的缩放因子（默认：1.0）。
        device: 计算设备（默认：'cpu'，可设为 'cuda'）。

    返回：
        合并后的模型状态字典。
    """
    # 验证输入
    total_samples = sum(num_samples)
    if total_samples == 0:
        raise ValueError("num_samples 总和必须大于 0")

    # 步骤 0：提取模型状态字典并移到指定设备
    # state_dicts = [model.state_dict() for model in models]
    init_state_dict = {k: v.to(device) for k, v in init_state_dict.items()}

    # 步骤 1：初始化合并状态字典并计算任务向量（仅针对学习参数）
    merged_state_dict = {key: init_state_dict[key].clone() for key in init_state_dict}
    task_vectors = []

    for state_dict in state_dicts:
        task_vector = {}
        for key in state_dict:
            # 仅对学习参数（requires_grad=True）计算任务向量
            if init_state_dict[key].requires_grad:
                task_vector[key] = state_dict[key].to(device) - init_state_dict[key]
        task_vectors.append(task_vector)

    # 步骤 2：修剪冗余参数
    trimmed_task_vectors = []
    for task_vector in task_vectors:
        trimmed_vector = {}
        for key in task_vector:
            tensor = task_vector[key]
            # 计算绝对值并确定 top-k% 的阈值
            abs_tensor = torch.abs(tensor)
            threshold = torch.quantile(abs_tensor.view(-1).float(), 1 - k)
            # 保留绝对值大于阈值的参数，其余设为 0
            mask = abs_tensor >= threshold
            trimmed_vector[key] = tensor * mask.float()
        trimmed_task_vectors.append(trimmed_vector)

    # 步骤 3：选择符号（Elect Sign，仅针对学习参数）
    elected_signs = {}
    for key in task_vectors[0]:  # 只遍历学习参数的键
        # 计算所有任务向量在该参数上的总和
        sum_task_vector = torch.zeros_like(init_state_dict[key])
        for trimmed_vector in trimmed_task_vectors:
            sum_task_vector += trimmed_vector[key]
        signs, counts = torch.sign(sum_task_vector).unique(return_counts=True)
        print(f"Key: {key}, Sign counts: {dict(zip(signs.tolist(), counts.tolist()))}")
        # 确定最终符号向量
        elected_signs[key] = torch.sign(sum_task_vector)

    # 步骤 4：分离合并（Disjoint Merge，仅针对学习参数）
    merged_task_vector = {}
    for key in task_vectors[0]:  # 只遍历学习参数的键
        merged_task_vector[key] = torch.zeros_like(init_state_dict[key])
        for p in range(merged_task_vector[key].numel()):
            # 获取所有任务向量在该参数位置的值和符号
            values = []
            signs = []
            for trimmed_vector in trimmed_task_vectors:
                value = trimmed_vector[key].view(-1)[p]
                if value != 0:  # 只考虑非零值
                    values.append(value)
                    signs.append(torch.sign(value).item())

            # 如果有值，计算与最终符号一致的平均值
            if values:
                elected_sign = elected_signs[key].view(-1)[p].item()
                aligned_values = [v for v, s in zip(values, signs) if s == elected_sign]
                if aligned_values:
                    merged_task_vector[key].view(-1)[p] = sum(aligned_values) / len(aligned_values)

    # 步骤 5：合并学习参数
    for key in merged_state_dict:
        if key in merged_task_vector:
            merged_state_dict[key] += lambda_scale * merged_task_vector[key]

    # 步骤 6：合并非学习参数（缓冲区，如 BatchNorm 的 running_mean 和 running_var）
    for key in init_state_dict:
        if not init_state_dict[key].requires_grad:  # 非学习参数
            if init_state_dict[key].dtype == torch.long:
                # 对于 Long 类型（如 num_batches_tracked），取加权平均并四舍五入
                weighted_sum = 0.0
                for state_dict, samples in zip(state_dicts, num_samples):
                    weighted_sum += state_dict[key].item() * (samples / total_samples)
                merged_state_dict[key] = torch.tensor(round(weighted_sum), dtype=torch.long, device=device)
            else:
                # 对于浮点类型（如 running_mean, running_var），直接加权平均
                weighted_sum = torch.zeros_like(init_state_dict[key])
                for state_dict, samples in zip(state_dicts, num_samples):
                    weighted_sum += state_dict[key].to(device) * (samples / total_samples)
                merged_state_dict[key] = weighted_sum

    return merged_state_dict

def FedMax(models):
    model_max = copy.deepcopy(models[0])
    for k in model_max.keys():
        # 初始化一个张量来存储所有模型在该层的权重
        weights = [model[k] for model in models]
        # 沿第0维度（模型维度）取最大值
        model_max[k] = torch.max(torch.stack(weights), dim=0)[0]
    return model_max
'''def FedAvg(models, nums):
    model_avg = copy.deepcopy(models[0])
    total = sum(nums)
    for k in model_avg.keys():
        model_avg[k] = torch.div(model_avg[k], total / nums[0])
        for i in range(1, len(models)):
            model_avg[k] += torch.div(models[i][k], total / nums[i])
    return model_avg'''
