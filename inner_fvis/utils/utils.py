# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Pavlo Molchanov and Hongxu Yin
# --------------------------------------------------------

import torch
import os
from torch import distributed, nn
import numpy as np

from typing import Literal

from utils.lrp_models import *
from utils.lrp_layers import *
from typing import Optional, Union
import warnings

SkipConnectionPropType = Literal["simple", "flows_skip", "flows_skip_simple", "latest"]
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

def module_generator(model: nn.Module, reverse: bool = False):
    """
    Generator for nested Module, can handle nested Sequential in one layer
    Note that you cannot get layers by index

    Args:
        model (nn.Module): Model
        reverse(bool)    : Whether to reverse

    Yields:
        Each layer of the model
    """
    modules = list(model.children())
    if reverse:
        modules = modules[::-1]

    for module in modules:
        if list(module.children()):
            yield from module_generator(module, reverse)
            continue
        yield module


def reverse_normalize(
    x: np.ndarray,
):
    """
    Restore normalization

    Args:
        x(ndarray) : Matrix that has been normalized
        mean(Tuple): Mean specified at the time of normalization
        std(Tuple) : Standard deviation specified at the time of normalization
    """
    if x.shape[0] == 1:
        x = x * imagenet_std + imagenet_mean
        return x
    x[0, :, :] = x[0, :, :] * imagenet_std[0] + imagenet_mean[0]
    x[1, :, :] = x[1, :, :] * imagenet_std[1] + imagenet_mean[1]
    x[2, :, :] = x[2, :, :] * imagenet_std[2] + imagenet_mean[2]

    return x

def apply_heat_quantization(attention, q_level: int = 8):
    max_ = attention.max()
    min_ = attention.min()

    # quantization
    bin = np.linspace(min_, max_, q_level)

    # apply quantization
    for i in range(q_level - 1):
        attention[(attention >= bin[i]) & (attention < bin[i + 1])] = bin[i]

    return attention

def _cumulative_sum_threshold(values: np.ndarray, percentile: Union[int, float]):
    # given values should be non-negative
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]

def _normalize_scale(attr: np.ndarray, scale_factor: float):
    assert scale_factor != 0, "Cannot normalize by scale factor = 0"
    if abs(scale_factor) < 1e-5:
        warnings.warn(
            "Attempting to normalize by value approximately 0, visualized results"
            "may be misleading. This likely means that attribution values are all"
            "close to 0."
        )
    attr_norm = attr / scale_factor
    return np.clip(attr_norm, -1, 1)

def normalize_attr(
    attr: np.ndarray,
    sign: str,
    outlier_perc: Union[int, float] = 2,
    reduction_axis: Optional[int] = None,
):
    attr_combined = attr
    if reduction_axis is not None:
        attr_combined = np.sum(attr, axis=reduction_axis)

    # Choose appropriate signed values and rescale, removing given outlier percentage.
    if sign == "all":
        threshold = _cumulative_sum_threshold(np.abs(attr_combined), 100 - outlier_perc)
    elif sign == "positive":
        attr_combined = (attr_combined > 0) * attr_combined
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    elif sign == "negative":
        attr_combined = (attr_combined < 0) * attr_combined
        threshold = -1 * _cumulative_sum_threshold(
            np.abs(attr_combined), 100 - outlier_perc
        )
    elif sign == "absolute_value":
        attr_combined = np.abs(attr_combined)
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    else:
        raise AssertionError("Visualize Sign type is not valid.")
    return _normalize_scale(attr_combined, threshold)


def layers_lookup(version: SkipConnectionPropType = "latest") -> dict:
    """Lookup table to map network layer to associated LRP operation.

    Returns:
        Dictionary holding class mappings.
    """

    # For the purpose of the ablation study on relevance propagation for skip connections
    if version == "simple":
        return layers_lookup_simple()
    elif version == "flows_skip":
        return layers_lookup_flows_pure_skip()
    elif version == "flows_skip_simple":
        return layers_lookup_simple_flows_pure_skip()
    elif version == "latest":
        return layers_lookup_latest()
    else:
        raise ValueError("Invalid version was specified.")


def layers_lookup_latest() -> dict:
    lookup_table = {
        torch.nn.modules.linear.Linear: RelevancePropagationLinear,
        torch.nn.modules.conv.Conv2d: RelevancePropagationConv2d,
        torch.nn.modules.activation.ReLU: RelevancePropagationReLU,
        torch.nn.modules.dropout.Dropout: RelevancePropagationDropout,
        torch.nn.modules.flatten.Flatten: RelevancePropagationFlatten,
        torch.nn.modules.pooling.AvgPool2d: RelevancePropagationAvgPool2d,
        torch.nn.modules.pooling.MaxPool2d: RelevancePropagationMaxPool2d,
        torch.nn.modules.pooling.AdaptiveAvgPool2d: RelevancePropagationAdaptiveAvgPool2d,
        AdaptiveAvgPool2dWithActivation: RelevancePropagationAdaptiveAvgPool2d,
        torch.nn.BatchNorm2d: RelevancePropagationBatchNorm2d,
        BatchNorm2dWithActivation: RelevancePropagationBatchNorm2d,
        BasicBlock: RelevancePropagationBasicBlock,
        Bottleneck: RelevancePropagationBottleneck,
    }
    return lookup_table


def layers_lookup_simple() -> dict:
    lookup_table = {
        torch.nn.modules.linear.Linear: RelevancePropagationLinear,
        torch.nn.modules.conv.Conv2d: RelevancePropagationConv2d,
        torch.nn.modules.activation.ReLU: RelevancePropagationReLU,
        torch.nn.modules.dropout.Dropout: RelevancePropagationDropout,
        torch.nn.modules.flatten.Flatten: RelevancePropagationFlatten,
        torch.nn.modules.pooling.AvgPool2d: RelevancePropagationAvgPool2d,
        torch.nn.modules.pooling.MaxPool2d: RelevancePropagationMaxPool2d,
        torch.nn.modules.pooling.AdaptiveAvgPool2d: RelevancePropagationAdaptiveAvgPool2d,
        AdaptiveAvgPool2dWithActivation: RelevancePropagationAdaptiveAvgPool2d,
        torch.nn.BatchNorm2d: RelevancePropagationBatchNorm2d,
        BatchNorm2dWithActivation: RelevancePropagationBatchNorm2d,
        BasicBlock: RelevancePropagationBasicBlockSimple,
        Bottleneck: RelevancePropagationBottleneckSimple,
    }
    return lookup_table


def layers_lookup_flows_pure_skip() -> dict:
    lookup_table = {
        torch.nn.modules.linear.Linear: RelevancePropagationLinear,
        torch.nn.modules.conv.Conv2d: RelevancePropagationConv2d,
        torch.nn.modules.activation.ReLU: RelevancePropagationReLU,
        torch.nn.modules.dropout.Dropout: RelevancePropagationDropout,
        torch.nn.modules.flatten.Flatten: RelevancePropagationFlatten,
        torch.nn.modules.pooling.AvgPool2d: RelevancePropagationAvgPool2d,
        torch.nn.modules.pooling.MaxPool2d: RelevancePropagationMaxPool2d,
        torch.nn.modules.pooling.AdaptiveAvgPool2d: RelevancePropagationAdaptiveAvgPool2d,
        AdaptiveAvgPool2dWithActivation: RelevancePropagationAdaptiveAvgPool2d,
        torch.nn.BatchNorm2d: RelevancePropagationBatchNorm2d,
        BatchNorm2dWithActivation: RelevancePropagationBatchNorm2d,
        BasicBlock: RelevancePropagationBasicBlockFlowsPureSkip,
        Bottleneck: RelevancePropagationBottleneckFlowsPureSkip,
    }
    return lookup_table


def layers_lookup_simple_flows_pure_skip() -> dict:
    lookup_table = {
        torch.nn.modules.linear.Linear: RelevancePropagationLinear,
        torch.nn.modules.conv.Conv2d: RelevancePropagationConv2d,
        torch.nn.modules.activation.ReLU: RelevancePropagationReLU,
        torch.nn.modules.dropout.Dropout: RelevancePropagationDropout,
        torch.nn.modules.flatten.Flatten: RelevancePropagationFlatten,
        torch.nn.modules.pooling.AvgPool2d: RelevancePropagationAvgPool2d,
        torch.nn.modules.pooling.MaxPool2d: RelevancePropagationMaxPool2d,
        torch.nn.modules.pooling.AdaptiveAvgPool2d: RelevancePropagationAdaptiveAvgPool2d,
        AdaptiveAvgPool2dWithActivation: RelevancePropagationAdaptiveAvgPool2d,
        torch.nn.BatchNorm2d: RelevancePropagationBatchNorm2d,
        BatchNorm2dWithActivation: RelevancePropagationBatchNorm2d,
        BasicBlock: RelevancePropagationBasicBlockSimpleFlowsPureSkip,
        Bottleneck: RelevancePropagationBottleneckSimpleFlowsPureSkip,
    }
    return lookup_table

def load_model_pytorch(model, load_model, gpu_n=0):
    print("=> loading checkpoint '{}'".format(load_model))

    checkpoint = torch.load(load_model, map_location = lambda storage, loc: storage.cuda(gpu_n))

    if 'state_dict' in checkpoint.keys():
        load_from = checkpoint['state_dict']
    else:
        load_from = checkpoint

    if 1:
        if 'module.' in list(model.state_dict().keys())[0]:
            if 'module.' not in list(load_from.keys())[0]:
                from collections import OrderedDict

                load_from = OrderedDict([("module.{}".format(k), v) for k, v in load_from.items()])

        if 'module.' not in list(model.state_dict().keys())[0]:
            if 'module.' in list(load_from.keys())[0]:
                from collections import OrderedDict

                load_from = OrderedDict([(k.replace("module.", ""), v) for k, v in load_from.items()])

    if 1:
        if list(load_from.items())[0][0][:2] == "1." and list(model.state_dict().items())[0][0][:2] != "1.":
            load_from = OrderedDict([(k[2:], v) for k, v in load_from.items()])

        load_from = OrderedDict([(k, v) for k, v in load_from.items() if "gate" not in k])

    model.load_state_dict(load_from, strict=True)

    epoch_from = -1
    if 'epoch' in checkpoint.keys():
        epoch_from = checkpoint['epoch']
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(load_model, epoch_from))


def create_folder(directory):
    # from https://stackoverflow.com/a/273227
    if not os.path.exists(directory):
        os.makedirs(directory)

def distributed_is_initialized():
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def beta_policy(mom_fn):
    def _alr(optimizer, iteration, epoch, param, indx):
        mom = mom_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group[param][indx] = mom

    return _alr


def mom_cosine_policy(base_beta, warmup_length, epochs):
    def _beta_fn(iteration, epoch):
        if epoch < warmup_length:
            beta = base_beta * (epoch + 1) / warmup_length
        else:
            beta = base_beta
        return beta

    return beta_policy(_beta_fn)


def clip(image_tensor, use_fp16=False):
    '''
    adjust the input based on mean and variance
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def denormalize(image_tensor, use_fp16=False):
    '''
    convert floats back to input
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor