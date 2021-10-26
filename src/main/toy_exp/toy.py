# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2021-10-25 17:30:31
# @Last Modified by:   Your name
# @Last Modified time: 2021-10-25 17:38:02
#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qi.chen.1@ulaval.ca
# Created Time : Thu May 13 19:49:28 2021
# File Name: toy.py
# Description:
"""

from __future__ import absolute_import,division, print_function

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from src.utils.common import  set_random_seed

matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def truncated_resample(r, size=100, x_a=-12, y_a=-12, x_b=4, y_b=4):
    for i in range(size):
        x, y = r[i]
        #print(x, y)
        while x < x_a or x > x_b or y < y_a or y > y_b:
            tmp = np.random.multivariate_normal(mean=[-4,-4], cov=np.diag([5,5]), size=1)
            r[i] = tmp[0]
            x, y = r[i]
    return r

def gen_toy_data(seed=2, n_dim=2, n_tr=10, n_te=1, m_tr=200, m_va=20, hyper_prior='gaussian', prior='gaussian',unit_var=True, truncated=True):
    # set random seed
    if not seed == 0:
        set_random_seed(seed)
    # init task env
    if hyper_prior=='gaussian':
        if not unit_var:
            tr_task_means = np.random.multivariate_normal(mean=[-4,-4], cov=np.diag([5,5]), size=n_tr)
            tr_task_vars = np.random.multivariate_normal(mean=[0.1,0.1], cov=np.diag([0.5,0.5]), size=n_tr)
            te_task_means = np.random.multivariate_normal(mean=[-4,-4], cov=np.diag([5,5]), size=n_te)
            te_task_vars = np.random.multivariate_normal(mean=[0.1,0.1], cov=np.diag([0.5,0.5]), size=n_te)
        else:
            tr_task_means = np.random.multivariate_normal(mean=[-4,-4], cov=np.diag([5,5]), size=n_tr)
            te_task_means = np.random.multivariate_normal(mean=[-4,-4], cov=np.diag([5,5]), size=n_te)
            if truncated:
                tr_task_means = truncated_resample(tr_task_means, size=n_tr)
                te_task_means = truncated_resample(te_task_means, size=n_te)

            tr_task_vars = [[0.1, 0.1] for i in range(n_tr)]
            te_task_vars = [[0.1, 0.1] for i in range(n_te)]
    else:
        raise ValueError('undefined distribution')
    print(tr_task_means[0], tr_task_vars[0])
    # generate data set
    S_tr = []
    S_va = []
    S = []
    T_tr = []
    T_va = []
    T = []
    if prior=='gaussian':
        for i in range(n_tr):
            train = np.random.multivariate_normal(
                mean=tr_task_means[i],
                cov=np.diag(tr_task_vars[i]),
                size=m_tr).astype(np.float32)
            valid = np.random.multivariate_normal(
                mean=tr_task_means[i],
                cov=np.diag(tr_task_vars[i]),
                size=m_va).astype(np.float32)
            #print(valid)
            S_tr.append(train)
            S_va.append(valid)
            S.append(np.concatenate((train, valid)))
        for i in range(n_te):
            train = np.random.multivariate_normal(
                mean=te_task_means[i],
                cov=np.diag(te_task_vars[i]),
                size=m_tr).astype(np.float32)
            valid = np.random.multivariate_normal(
                mean=te_task_means[i],
                cov=np.diag(te_task_vars[i]),
                size=m_va).astype(np.float32)
            T_tr.append(train)
            T_va.append(valid)
            T.append(np.concatenate((train, valid)))
    else:
        raise ValueError('undefined distribution')
    return (S, T, S_tr, T_tr, S_va, T_va)

# -------------------------------------------------------------------------------------------
#  Learning
# -------------------------------------------------------------------------------------------
learning_type = 'Meta_SGLD'

(S, T, S_tr, T_tr, S_va, T_va) = gen_toy_data(seed=2, n_dim=2, n_tr=20000, n_te=5, m_tr=50, m_va=50, \
        hyper_prior='gaussian', prior='gaussian', unit_var=True, truncated=True)

data_set=S
print(np.shape(S))
print(np.shape(S_tr[0]))
print(np.mean(S_tr[0][:,0]), np.mean(S_tr[0][:,1]),np.var(S_tr[0][:,0]), np.var(S_tr[0][:,1]))

#PAC-Bayes
if learning_type == 'PAC_Bayes':
    from src.main.toy_exp import toy_amit
    complexity_type = 'PAC_Bayes'  # 'Variational_Bayes' / 'PAC_Bayes' /
    toy_amit.learn(data_set, complexity_type)
#Meta-SGLD
if learning_type == 'Meta_SGLD':
    from src.main.toy_exp import toy_maml
    tr = [S_tr, S_va]
    te = [T_tr, T_va]
    print(len(tr))
    toy_maml.learn(tr, te)
#Meta-SGLD

plt.savefig('ToyFig1.pdf', format='pdf', bbox_inches='tight')

plt.show()
