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

    for key in model_avg:
        if model_avg[key].dtype == torch.long:
            weighted_sum = 0.0
            for i in range(len(models)):
                weighted_sum += models[i][key].item() * (num_samples[i] / total_samples)
            model_avg[key] = torch.tensor(round(weighted_sum), dtype=torch.long)
        else:
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

    if tensor.dtype == torch.long:
        return tensor
    t_min, t_max = tensor.min(), tensor.max()
    scale = (t_max - t_min) / (2 ** bits - 1) if t_max != t_min else 1.0
    zero_point = t_min
    quantized = torch.round((tensor - zero_point) / scale).clamp(0, 2 ** bits - 1)
    return zero_point + quantized * scale

def quantize_client(local_model_state: dict, global_model_state: dict, quantization_bits: int = 8) -> dict:

    weight_diff = copy.deepcopy(local_model_state)
    for key in weight_diff:
        weight_diff[key] = local_model_state[key] - global_model_state[key]

    quantized_diff = copy.deepcopy(weight_diff)
    for key in quantized_diff:
        quantized_diff[key] = quantize_tensor(weight_diff[key], quantization_bits)

    quantized_model_state = copy.deepcopy(global_model_state)
    for key in quantized_model_state:
        quantized_model_state[key] += quantized_diff[key]

    return quantized_model_state

def fedadam_aggregate(global_model_dict, local_models, local_nums, server_state, eta=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
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
        server_state['t'] = 0

    server_state['t'] += 1
    t = server_state['t']
    non_trainable_keys = {'running_mean', 'running_var', 'num_batches_tracked'}

    avg_model_dict = FedAvg(local_models, local_nums)

    updated_state_dict = {}

    for k in global_model_dict:
        global_param = global_model_dict[k]
        avg_param = avg_model_dict[k]

        if torch.is_floating_point(global_param) and not any(non_key in k for non_key in non_trainable_keys):
            delta = avg_param - global_param
            m = beta1 * server_state['m_t'][k] + (1 - beta1) * delta
            v = beta2 * server_state['v_t'][k] + (1 - beta2) * (delta ** 2)

            update = eta * m / (torch.sqrt(v) + eps)
            updated_state_dict[k] = global_param + update
            server_state['m_t'][k] = m
            server_state['v_t'][k] = v

        else:
            updated_state_dict[k] = avg_param

    return updated_state_dict, server_state
