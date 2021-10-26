
"""
# Author: qi.chen.1@ulaval.ca
# Created Time : Sat May 15 16:54:57 2021
# File Name: meta_sgld.py
# Description:
"""
from __future__ import absolute_import,division, print_function
import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
from    src.algo.learner import Learner
from    copy import deepcopy
import logging 
from torchviz import make_dot

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr#\beta_kt inital
        self.meta_lr = args.meta_lr#\eta_t
        self.last_meta_lr = self.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt#m_tr
        self.k_qry = args.k_qry#m_va
        self.task_num = args.task_num
        self.sample_num = args.sample_num
        self.update_step = args.update_step#K
        self.update_step_test = args.update_step_test
        # SGLD setting
        self.eva_bound = args.eva_bound
        self.sample_stocha =  args.sample_stocha
        self.add_noise = args.add_noise
        self.mc_times =  args.mc_times
        self.loss_type = args.loss
        self.task_type = args.task #'regression'#'classification'
        # define model 
        self.net = Learner(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        # initial param
        self.spt_siz = None
        self.qrt_size = None
        self.inner_g_bound = 0
        self.inner_g_inco_bound = 0
        self.outer_g_bound = 0
        self.outer_g_inco_bound = 0
        self.epoch = 0
        self.temp = args.temp
        self.inner_decay_step = args.inner_decay_step
        self.outer_decay_step = args.outer_decay_step
        self.env_task_num = args.env_task_num
        self.max_u_g_norm = 0.8645#0.94018#0.9150
        self.max_w_g_norm = 74.46068#74.0875#72.11578
        self.inner_lip_bound = 0
        self.outer_lip_bound = 0

    def _loss(self, input_, target_):
        if self.loss_type == 'cross_entropy':
            return F.cross_entropy(input_, target_, reduce='mean')
        elif self.loss_type == 'mse':
            return F.mse_loss(input_, target_)
        else:
            raise NotImplementedError


    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def cal_grad_norm(self, grad):
        para_norm = 0
        for g in grad:
            para_norm += g.data.norm(2).item() ** 2
            #print(para_norm)
        return para_norm


    def cal_grad_incohence(self, grad1, grad2):
        g_incohen = 0
        for (g1, g2) in zip(grad1, grad2):
            g_incohen += (g1.data - g2.data).norm(2).item() ** 2
            #print(g_incohen)
        return g_incohen


    def get_stochastic_sample_idx(self):
        if self.sample_stocha: # True SGLD, False GLD
            s_inds = np.random.choice(self.spt_size, self.sample_num, replace=False)
            q_inds = np.random.choice(self.qrt_size, self.sample_num, replace=False)
        else:
            s_inds = range(self.spt_size)
            q_inds = range(self.qrt_size)
        return (s_inds, q_inds)


    def inner_mc(self, x_spt, y_spt, x_qry, y_qry, update_lr, init_weights=None):
        # m_tr * feature_size / m_va * feature_size
        if init_weights is None:
            init_weights = self.net.parameters()
            noise_var = np.sqrt(self.last_meta_lr/self.temp)
        else:
            noise_var = np.sqrt(update_lr/self.temp)
        
        g_norm = []
        g_incohen = []
        
        for i in range(self.mc_times):
            # sample initial parameter
            fast_weights = [p + torch.normal(torch.zeros_like(p), noise_var) for p in init_weights]
            #print(fast_weights)
            for j in range(self.mc_times):
                # sample data
                s_inds, q_inds = self.get_stochastic_sample_idx()
                # calcule gradient
                out = self.net(x_spt[s_inds], fast_weights, bn_training=True)
                loss1 = self._loss(out, y_spt[s_inds])
                #print(loss)
                #print(fast_weights)
                g1 = torch.autograd.grad(loss1, fast_weights)
                g_norm.append(self.cal_grad_norm(g1))
           
                out = self.net(torch.cat((x_spt[s_inds], x_qry[q_inds]),0), vars=fast_weights, bn_training=True)
                loss2 = self._loss(out, torch.cat((y_spt[s_inds], y_qry[q_inds]),0))
                g2 = torch.autograd.grad(loss2, fast_weights)  
                g_incohen.append(self.cal_grad_incohence(g1,g2))
                #print(g1, g2)
                #print(loss1.cpu().detach().numpy(),loss2.cpu().detach().numpy())
                if not self.sample_stocha:
                    break

        g_norm_mean, g_norm_var, g_inco_mean, g_inco_var = np.mean(g_norm, axis=0), np.var(g_norm, axis=0), \
            np.mean(g_incohen, axis=0), np.var(g_incohen, axis=0)
        if g_norm_mean > self.max_w_g_norm:
            self.max_w_g_norm = g_norm_mean
        self.inner_g_bound += g_norm_mean * update_lr * self.temp/ 2
        self.inner_g_inco_bound += g_inco_mean * update_lr * self.temp/ 2
        self.inner_lip_bound += self.max_w_g_norm * update_lr * self.temp/ 2
        return g_norm_mean, g_norm_var, g_inco_mean, g_inco_var


    def outer_mc(self, x_spt, y_spt, x_qry, y_qry):
        task_num = x_spt.size(0)
        self.spt_size = x_spt.size(1)
        self.qrt_size = x_qry.size(1)
        noise_var = np.sqrt(self.last_meta_lr/self.temp)
        g_norm = []
        g_incohen = []
        
        for i in range(self.mc_times):
            meta_loss_Str_K, meta_loss_S_K, meta_loss_K = 0, 0, 0
            # sample initial parameter for different pathes 
            weights = list(map(lambda p: p + torch.normal(torch.zeros_like(p), noise_var), self.net.parameters()))
            # not pointer, will not change net.paramaters()
            for i in range(task_num):
                update_lr = self.update_lr
                (s_inds, q_inds) = self.get_stochastic_sample_idx()
                noise_var = np.sqrt(update_lr/self.temp)
                # 1. run the i-th task and compute loss for k=0
                # update task param on support set
                out = self.net(x_spt[i][s_inds], weights, bn_training=True)
                loss = self._loss(out, y_spt[i][s_inds])
                grad = torch.autograd.grad(loss, weights)
                weights = list(map(lambda p: p[1] - update_lr * p[0] + torch.normal(torch.zeros_like(p[1]), noise_var), zip(grad, weights)))
                #self.show_param(weights)
                #print("============================")
                for k in range(1, self.update_step):# inner updates    
                    # modify learning rate 
                    update_lr = self.update_lr * 1 / (np.float(k + 1))
                    noise_var = np.sqrt(update_lr/self.temp)
                    (s_inds, q_inds) = self.get_stochastic_sample_idx()
                    # update task specific params
                    out = self.net(x_spt[i][s_inds], weights, bn_training=True)
                    loss = self._loss(out, y_spt[i][s_inds])
                    grad = torch.autograd.grad(loss, weights)
                    #self.show_param(weights)
                    #print("*********************************")
                    weights = list(map(lambda p: p[1] - update_lr * p[0] + torch.normal(torch.zeros_like(p[1]), noise_var), zip(grad, weights)))
                # evaluate on different validation set
                
                out1 = self.net(x_qry[i][q_inds], weights, bn_training=True)
                loss1 = self._loss(out1, y_qry[i][q_inds])
                meta_loss_K += loss1
                out2 = self.net(x_spt[i][s_inds], weights, bn_training=True)
                loss2 = self._loss(out2, y_spt[i][s_inds])
                meta_loss_Str_K += loss2
                out3 = self.net(torch.cat((x_spt[i][s_inds], x_qry[i][q_inds]),0), weights, bn_training=True)
                loss3 = self._loss(out3, torch.cat((y_spt[i][s_inds], y_qry[i][q_inds]),0))
                meta_loss_S_K += loss3
                
            
            meta_loss_K = meta_loss_K / self.task_num    
            meta_loss_Str_K = meta_loss_Str_K / self.task_num
            meta_loss_S_K = meta_loss_S_K / self.task_num

            self.meta_optim.zero_grad()
            meta_loss_K.backward(retain_graph=True)
            u_g = [deepcopy(p.grad) for p in self.net.parameters()]#!!!!!
            self.meta_optim.zero_grad()
            meta_loss_Str_K.backward(retain_graph=True)
            u_g1 = [deepcopy(p.grad) for p in self.net.parameters()]
            self.meta_optim.zero_grad()
            meta_loss_S_K.backward()
            u_g2 = [deepcopy(p.grad) for p in self.net.parameters()]
            #self.meta_optim.zero_grad()
            #print(loss1, loss2, loss3)
            #print(loss1.cpu().detach().numpy(),loss2.cpu().detach().numpy(),loss3.cpu().detach().numpy())
            #print(u_g, u_g1, u_g2)
            g_norm.append(self.cal_grad_norm(u_g))
            g_incohen.append(self.cal_grad_incohence(u_g1,u_g2)) 
        
        g_norm_mean, g_norm_var, g_inco_mean, g_inco_var = np.mean(g_norm, axis=0), np.var(g_norm, axis=0), \
                    np.mean(g_incohen, axis=0), np.var(g_incohen, axis=0)
        if g_norm_mean > self.max_u_g_norm:
            self.max_u_g_norm = g_norm_mean
        self.outer_g_bound  +=  g_norm_mean * self.meta_lr * self.temp/ 2
        self.outer_g_inco_bound  += g_inco_mean * self.meta_lr * self.temp/ 2
        self.outer_lip_bound += self.max_u_g_norm * self.meta_lr  * self.temp/ 2
        #print last step information
        print('INFO\touter step:{}, eta_t: {}, g_norm: {}, g_norm_var:{}, g_inco:{}, g_inco_var:{}, loss:{}'\
                   .format(self.epoch, self.meta_lr, g_norm_mean, g_norm_var, g_inco_mean, g_inco_var, meta_loss_K.cpu().detach().numpy()))

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, sptsz, c_, h, w]
        :param y_spt:   [b, sptsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num = x_spt.size(0)
        self.spt_size = x_spt.size(1)
        self.qrt_size = x_qry.size(1)
        querysz = self.qrt_size
        stepwise_va_losses = [0 for _ in range(self.update_step + 1)]  # stepwise_va_losses[i] is the loss on step i
        if (self.epoch + 1) % self.outer_decay_step == 0:
            self.meta_lr = self.meta_lr * 0.96
        if (self.epoch + 1) % self.inner_decay_step == 0:
            self.update_lr = self.update_lr * 0.96

        if self.add_noise:
            noise_var = np.sqrt(self.last_meta_lr/self.temp)
            weights = list(map(lambda p: p + torch.normal(torch.zeros_like(p), noise_var), self.net.parameters()))
        else:
            weights = list(map(lambda p: p, self.net.parameters()))
        if self.task_type == 'classification':
            corrects = [0 for _ in range(self.update_step + 1)]
        else:
            predicts = []

        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            update_lr = self.update_lr
            (s_inds, q_inds) = self.get_stochastic_sample_idx()
            # 1. run the i-th task and compute loss for k=0
            out = self.net(x_spt[i][s_inds], weights, bn_training=True)
            loss = self._loss(out, y_spt[i][s_inds])
            grad = torch.autograd.grad(loss, weights)
            fast_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grad, weights)))
            no_noise_weights = fast_weights
            # add noise sgld
            if self.add_noise:
                noise_var = np.sqrt(update_lr/self.temp)
                fast_weights = list(map(lambda p: p[1] - update_lr * p[0] + torch.normal(torch.zeros_like(p[1]), noise_var), zip(grad, weights)))
                #print("fast weight")

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [sptsz, nway]
                out = self.net(x_qry[i][q_inds], self.net.parameters(), bn_training=True)
                loss_q = self._loss(out, y_qry[i][q_inds])
                stepwise_va_losses[0] += loss_q
                if self.task_type == 'classification':
                    pred_q = F.softmax(out, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i][q_inds]).sum().item()
                    corrects[0] = corrects[0] + correct

            # MC calculate bound
            if self.eva_bound:
                # first update use no noise U as initial params, random sample mc times adding noise
                g_norm_mean, g_norm_var, g_inco_mean, g_inco_var = self.inner_mc(x_spt[i], y_spt[i], x_qry[i], y_qry[i], update_lr, init_weights=None)
                print('DEBUG\touter step:{}, task:{}, inner step: {}, beta_tk: {}, g_norm: {}, g_norm_var:{}, g_inco:{}, g_inco_var:{}, loss:{}'\
                    .format(self.epoch, i, 1, update_lr, g_norm_mean, g_norm_var, g_inco_mean, g_inco_var, loss_q.cpu().detach().numpy()))
            else:
                print('DEBUG\touter step:{}, task:{}, inner step: {}, beta_tk: {}, loss:{}'\
                    .format(self.epoch, i, 1, update_lr, loss_q.cpu().detach().numpy()))
            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [sptsz, nway]
                out = self.net(x_qry[i][q_inds], fast_weights, bn_training=True)
                loss_q = self._loss(out, y_qry[i][q_inds])
                stepwise_va_losses[1] += loss_q
                if self.task_type == 'classification':
                    pred_q = F.softmax(out, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i][q_inds]).sum().item()
                    corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                 # evaluate bound initiate with last step no noise update
                if self.eva_bound:
                    g_norm_mean, g_norm_var, g_inco_mean, g_inco_var = self.inner_mc(x_spt[i], y_spt[i], x_qry[i], y_qry[i], update_lr, init_weights=no_noise_weights)
                    print('DEBUG\t outer setp:{}, task:{}, inner step: {}, beta_tk: {}, g_norm: {}, g_norm_var:{}, g_inco:{}, g_inco_var:{}, loss:{}'\
                            .format(self.epoch, i, k, update_lr, g_norm_mean, g_norm_var, g_inco_mean, g_inco_var, loss_q.cpu().detach().numpy()))
                else:
                    pass 
                    #print('DEBUG\t outer setp:{}, task:{}, inner step: {}, beta_tk: {}, loss:{}'\
                    #        .format(self.epoch, i, k, update_lr, loss_q.cpu().detach().numpy()))
                   # modify learning rate 
                #update_lr = self.update_lr * 1 / (np.float(k + 1))
               
                (s_inds, q_inds) = self.get_stochastic_sample_idx()
                # 1. run the i-th task and compute loss for k=1~K-1
                out = self.net(x_spt[i][s_inds], fast_weights, bn_training=True)
                loss = self._loss(out, y_spt[i][s_inds])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grad, fast_weights)))
                no_noise_weights = fast_weights
                if self.add_noise:
                    noise_var = np.sqrt(update_lr/self.temp)
                    fast_weights = list(map(lambda p: p + torch.normal(torch.zeros_like(p), noise_var), fast_weights))
               
                out = self.net(x_qry[i][q_inds], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = self._loss(out, y_qry[i][q_inds])
                stepwise_va_losses[k + 1] += loss_q
                if self.task_type == 'classification':
                    with torch.no_grad():
                        pred_q = F.softmax(out, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i][q_inds]).sum().item()  # convert to numpy
                        corrects[k + 1] = corrects[k + 1] + correct
            #if self.eva_bound:
            #    # print last step information
            #    print('INFO\touter step:{},task:{}, inner step: {}, beta_tk: {}, g_norm: {}, g_norm_var:{}, g_inco:{}, g_inco_var:{}, loss:{}'\
            #            .format(self.epoch, i, k, self.update_lr, g_norm_mean, g_norm_var, g_inco_mean, g_inco_var, loss_q.cpu().detach().numpy()))
            #else:
            #    print('INFO\touter step:{},task:{}, inner step: {}, beta_tk: {}, loss:{}'\
            #            .format(self.epoch, i, k, self.update_lr, loss_q.cpu().detach().numpy()))
            if self.task_type == 'regression':
                predicts.append(fast_weights[0].cpu().detach().numpy())

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = stepwise_va_losses[-1] / task_num
        if self.eva_bound:
            # outer mc
            self.outer_mc(x_spt, y_spt, x_qry, y_qry)
        else:
            print('INFO\touter step:{}, eta_t: {},  meta loss:{}'.format(self.epoch, self.meta_lr, loss_q.cpu().detach().numpy()))
            
        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()
        if self.task_type == 'classification':
            results = np.array(corrects) / (querysz * task_num)
        else:
            results = predicts
        print('Bound\touter step:{},eta_t:{}, beta_tk: {}, outer norm bound:{}, inner norm bound:{}, outer inco bound:{}, inner inco bound:{}'\
                         .format(self.epoch, self.meta_lr, self.update_lr, np.sqrt(self.outer_g_bound/(self.env_task_num*self.k_qry)),\
                              np.sqrt(self.inner_g_bound/(self.env_task_num*self.k_qry)), np.sqrt(self.outer_g_inco_bound/(self.env_task_num*self.k_qry)),\
                                   np.sqrt(self.inner_g_inco_bound/(self.env_task_num*self.k_qry))))
        norm_bound = np.sqrt((self.outer_g_bound + self.inner_g_bound)/(self.env_task_num*self.k_qry))
        inco_bound = np.sqrt((self.outer_g_inco_bound + self.inner_g_inco_bound)/(self.env_task_num*self.k_qry))
        lip_bound = np.sqrt((self.outer_lip_bound + self.inner_lip_bound)/(self.env_task_num*self.k_qry))
        self.epoch += 1
        return results, loss_q.cpu().detach().numpy(), norm_bound, inco_bound, lip_bound


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [sptsz, c_, h, w]
        :param y_spt:   [sptsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        # single sample
        #assert len(x_spt.shape) == 4

        
        querysz = x_qry.size(0)
        if self.task_type == 'classification':
            corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        out = net(x_spt)
        loss = self._loss(out, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [sptsz, nway]
            out = net(x_qry, net.parameters(), bn_training=True)
            # [sptsz]
            if self.task_type == 'classification':
                pred_q = F.softmax(out, dim=1).argmax(dim=1)
                # scalar
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [sptsz, nway]
            out = net(x_qry, fast_weights, bn_training=True)
            # [sptsz]
            if self.task_type == 'classification':
                pred_q = F.softmax(out, dim=1).argmax(dim=1)
                # scalar
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = self._loss(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            out = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = self._loss(out, y_qry)
            
            if self.task_type == 'classification':
                with torch.no_grad():
                    pred_q = F.softmax(out, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct
        
        if self.task_type == 'regression':
            predict = fast_weights[0].cpu().detach().numpy()

        del net
        if self.task_type == 'classification':
            result = np.array(corrects) / querysz
        else:
            result = predict                    
        return result, loss_q.cpu().detach().numpy()


   
def main():
    pass


if __name__ == '__main__':
    main()
