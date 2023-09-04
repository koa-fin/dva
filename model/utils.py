# -*-Encoding: utf-8 -*-
import logging
import os
import shutil
import time
from datetime import timedelta
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist

import torch.nn.functional as F


def get_stride_for_cell_type(cell_type):
    if cell_type.startswith('normal') or cell_type.startswith('combiner'):
        stride = 1
    elif cell_type.startswith('down'):
        stride = 2
    elif cell_type.startswith('up'):
        stride = -1
    else:
        raise NotImplementedError(cell_type)

    return stride


def get_input_size(dataset):
    if dataset in {'mnist', 'omniglot'}:
        return 32
    elif dataset == 'cifar10':
        return 32
    elif dataset.startswith('celeba') or dataset.startswith('imagenet') or dataset.startswith('lsun'):
        size = int(dataset.split('_')[-1])
        return size
    elif dataset == 'ffhq':
        return 256
    else:
        raise NotImplementedError


def get_arch_cells(arch_type):
    if arch_type == 'res_elu':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_elu', 'res_elu']
        arch_cells['down_enc'] = ['res_elu', 'res_elu']
        arch_cells['normal_dec'] = ['res_elu', 'res_elu']
        arch_cells['up_dec'] = ['res_elu', 'res_elu']
        arch_cells['normal_pre'] = ['res_elu', 'res_elu']
        arch_cells['down_pre'] = ['res_elu', 'res_elu']
        arch_cells['normal_post'] = ['res_elu', 'res_elu']
        arch_cells['up_post'] = ['res_elu', 'res_elu']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_bnelu':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnelu', 'res_bnelu']
        arch_cells['down_enc'] = ['res_bnelu', 'res_bnelu']
        arch_cells['normal_dec'] = ['res_bnelu', 'res_bnelu']
        arch_cells['up_dec'] = ['res_bnelu', 'res_bnelu']
        arch_cells['normal_pre'] = ['res_bnelu', 'res_bnelu']
        arch_cells['down_pre'] = ['res_bnelu', 'res_bnelu']
        arch_cells['normal_post'] = ['res_bnelu', 'res_bnelu']
        arch_cells['up_post'] = ['res_bnelu', 'res_bnelu']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_bnswish':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_dec'] = ['res_bnswish', 'res_bnswish']
        arch_cells['up_dec'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_post'] = ['res_bnswish', 'res_bnswish']
        arch_cells['up_post'] = ['res_bnswish', 'res_bnswish']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'mbconv_sep':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['mconv_e6k5g0']
        arch_cells['down_enc'] = ['mconv_e6k5g0']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['mconv_e3k5g0']
        arch_cells['down_pre'] = ['mconv_e3k5g0']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'mbconv_sep11':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['mconv_e6k11g0']
        arch_cells['down_enc'] = ['mconv_e6k11g0']
        arch_cells['normal_dec'] = ['mconv_e6k11g0']
        arch_cells['up_dec'] = ['mconv_e6k11g0']
        arch_cells['normal_pre'] = ['mconv_e3k5g0']
        arch_cells['down_pre'] = ['mconv_e3k5g0']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_mbconv':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res53_mbconv':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish5', 'res_bnswish']
        arch_cells['down_enc'] = ['res_bnswish5', 'res_bnswish']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['res_bnswish5', 'res_bnswish']
        arch_cells['down_pre'] = ['res_bnswish5', 'res_bnswish']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res35_mbconv':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish', 'res_bnswish5']
        arch_cells['down_enc'] = ['res_bnswish', 'res_bnswish5']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['res_bnswish', 'res_bnswish5']
        arch_cells['down_pre'] = ['res_bnswish', 'res_bnswish5']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res55_mbconv':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish5', 'res_bnswish5']
        arch_cells['down_enc'] = ['res_bnswish5', 'res_bnswish5']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['res_bnswish5', 'res_bnswish5']
        arch_cells['down_pre'] = ['res_bnswish5', 'res_bnswish5']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_mbconv9':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_dec'] = ['mconv_e6k9g0']
        arch_cells['up_dec'] = ['mconv_e6k9g0']
        arch_cells['normal_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_post'] = ['mconv_e3k9g0']
        arch_cells['up_post'] = ['mconv_e3k9g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'mbconv_res':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['mconv_e6k5g0']
        arch_cells['down_enc'] = ['mconv_e6k5g0']
        arch_cells['normal_dec'] = ['res_bnswish', 'res_bnswish']
        arch_cells['up_dec'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_pre'] = ['mconv_e3k5g0']
        arch_cells['down_pre'] = ['mconv_e3k5g0']
        arch_cells['normal_post'] = ['res_bnswish', 'res_bnswish']
        arch_cells['up_post'] = ['res_bnswish', 'res_bnswish']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'mbconv_den':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['mconv_e6k5g0']
        arch_cells['down_enc'] = ['mconv_e6k5g0']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['mconv_e3k5g8']
        arch_cells['down_pre'] = ['mconv_e3k5g8']
        arch_cells['normal_post'] = ['mconv_e3k5g8']
        arch_cells['up_post'] = ['mconv_e3k5g8']
        arch_cells['ar_nn'] = ['']
    else:
        raise NotImplementedError
    return arch_cells


def groups_per_scale(num_scales, num_groups_per_scale, is_adaptive, divider=2, minimum_groups=1):
    g = []
    n = num_groups_per_scale
    for s in range(num_scales):
        assert n >= 1
        g.append(n)
        if is_adaptive:
            n = n // divider
            n = max(minimum_groups, n)
    return g
