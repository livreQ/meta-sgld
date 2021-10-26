# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2021-10-25 14:59:25
# @Last Modified by:   Your name
# @Last Modified time: 2021-10-25 17:24:46
#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qi.chen.1@ulaval.ca
# Created Time : Sat May 15 16:54:57 2021
# File Name: toy_maml.py
# Description:
"""
from __future__ import absolute_import,division, print_function
import  torch, os
import  numpy as np
import  argparse
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from  src.algo.meta_sgld import Meta

def arg_config():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=400)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=8)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=8)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=5)
    argparser.add_argument('--sample_num', type=int, help='sample batch size, namely sample num', default=50)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=4)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--loss', type=str, help='specify loss function', default='mse')
    argparser.add_argument('--add_noise', type=int, help='add noise for each update', default=1)
    argparser.add_argument('--sample_stocha', type=int, help='random sample for subtasks', default=0)
    argparser.add_argument('--eva_bound', type=int, help='evaluate bound or not', default=1)
    argparser.add_argument('--mc_times', type=int, help='mc times', default=5)
    argparser.add_argument('--task', type=str, help='task type', default='regression')
    argparser.add_argument('--temp', type=int, help='sgld temparature', default=10000)
    argparser.add_argument('--inner_decay_step', type=int, help='inner decay', default=200)
    argparser.add_argument('--outer_decay_step', type=int, help='outer decay', default=200)
    argparser.add_argument('--env_task_num', type=int, help='all tasks avaible', default=20000)
    args = argparser.parse_args()
    return args


def learn(train_data, test_data):

    args = arg_config()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(train_data) 0 s_tr, 1 s_va
    print(np.shape(train_data[0]))# n_tr*m_tr*2
    print(np.shape(test_data[0]))#n_te*m_tr*2
    m_tr = len(train_data[0][0])#m_tr
    m_va = len(train_data[1][0])#m_va
    n_tasks = args.task_num
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    n_epochs = 200
    print(m_tr, m_va)
    args.env_task_num =  len(train_data[0])
    args.meta_lr = 0.2
    args.update_lr = 0.4
    print(args)

    config = [
        ('pd', [2, 2]),# 2-dim distance function
    ]

  
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)
  

    for step in range(n_epochs):
        # decrease meta learning rate with time
        maml.last_meta_lr = maml.meta_lr

        # Sample data batch:
        #b_task = range(n_tasks)

        b_task = np.random.choice(args.env_task_num, n_tasks) # sample a random task index
        print(b_task)

        s_inds = np.random.choice(m_tr, args.k_spt, replace=False)
        q_inds = np.random.choice(m_va, args.k_qry, replace=False)
        #print(type(batch_inds), type(b_task))
        #print(np.shape(train_data[0]), np.shape(train_data[1]))
        x_spt = np.array([train_data[0][i][s_inds] for i in b_task])
        x_qry = np.array([train_data[1][i][q_inds] for i in b_task])
        #print(x_spt, x_qry)
        #x_spt, x_qry = train_data
        y_spt, y_qry = np.zeros(np.shape(x_spt)).astype(np.float32), np.zeros(np.shape(x_qry)).astype(np.float32)
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), \
                                             torch.from_numpy(y_spt).to(device), \
                                                torch.from_numpy(x_qry).to(device),\
                                                    torch.from_numpy(y_qry).to(device)
        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        predicts, train_loss, norm_bound, inco_bound, lip_bound = maml.forward(x_spt, y_spt, x_qry, y_qry)
        u_mean = maml.net.parameters()[0].data


        if step % 20 == 0:
            dis = []

            # test
            x_spt =  np.array(test_data[0])
            x_qry =  np.array(test_data[1])
            y_spt, y_qry = np.zeros(np.shape(x_spt)).astype(np.float32), np.zeros(np.shape(x_qry)).astype(np.float32)
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), \
                                             torch.from_numpy(y_spt).to(device), \
                                                torch.from_numpy(x_qry).to(device),\
                                                    torch.from_numpy(y_qry).to(device)

            # split to single task each time
            test_loss = []
            for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                predict, loss = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                test_loss.append(loss)
                #print("Test\tloss:", loss)
            print("Test\tmeta loss:", np.mean(test_loss))
            print("epoch:{},Train-Test loss gap:{:.4}, norm_bound:{:.4}, inco_bound:{:.4}, lip_bound:{:.4}".format(step, np.abs(train_loss - np.mean(test_loss)), norm_bound, inco_bound, lip_bound))
        print("max norm:",maml.max_u_g_norm, maml.max_w_g_norm)
        #print("estmimated prior")
        #print(u_mean)
    U = u_mean.detach().numpy()
    #  Plots:
    fig1 = plt.figure()
    ax = plt.subplot(111, aspect='equal')
    ax.set(xlim=(-10, 4), ylim=(-10, 4))
    # plot prior:
    plt.plot(U[0], U[1], 'o', label='Predicted U mean')
    ell = Ellipse(xy=(U[0], U[1]),
                  width=0.1, height=0.1,
                  angle=0, color='blue')
    ell.set_facecolor('none')
    ax.add_artist(ell)
    print(np.shape(predicts))
    for (i_task, i) in zip(b_task, range(len(b_task))):
        # plot task data points:
        plt.plot(train_data[0][i_task][:, 0], train_data[0][i_task][:, 1], '.')
        # plot posterior:
        plt.plot(predicts[i][0], predicts[i][1], 'o', label='W mean Task {0}'.format(i_task))
        print("var:", np.var(train_data[0][i_task][:, 0]), np.var(train_data[0][i_task][:, 1]))

    plt.plot(-4, -4, 'x', label='True U mean')

    plt.legend()

