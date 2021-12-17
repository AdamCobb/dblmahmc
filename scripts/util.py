import torch
import hamiltorch

import torch.nn as nn
import torch.nn.functional as F


def metrics(samples, thinning, x_val, y_val, N_agents, net, tau_list, steps):
    acc = torch.zeros(N_agents, int(steps/thinning)-1)
    nll = torch.zeros(N_agents, int(steps/thinning)-1)
    for i in range(N_agents):
        mc_samples = [s for s in samples[i][::thinning]]
        pred_list, log_prob_list = hamiltorch.predict_model(net, x = x_val, y = y_val, samples=mc_samples, model_loss='multi_class_log_softmax_output', tau_out=1., tau_list=tau_list)
        ensemble_proba = F.softmax(pred_list[0], dim=-1)
        for s in range(1,len(mc_samples)):
            _, pred = torch.max(pred_list[:s].mean(0), -1)
            acc[i,s-1] = (pred.float() == y_val.flatten()).sum().float()/y_val.shape[0]
            ensemble_proba += F.softmax(pred_list[s], dim=-1)
            nll[i,s-1] = F.nll_loss(torch.log(mctestbed.util.central_smoothing(ensemble_proba.cpu()/(s+1))), y_val[:].long().cpu(), reduction='mean')
    return nll, acc

def metrics_reg(samples, thinning, x_val, y_val, N_agents, net, tau_list, steps, tau_out):
    error = torch.zeros(N_agents, int(steps/thinning)-1)
    for i in range(N_agents):
        mc_samples = [s for s in samples[i][::thinning]]
        pred_list, log_prob_list = hamiltorch.predict_model(net, x = x_val, y = y_val, samples=mc_samples, model_loss='regression', tau_out=tau_out, tau_list=tau_list)
        ensemble_proba = pred_list[0]
        for s in range(1,len(mc_samples)):
            pred = pred_list[:s].mean(0)
            error[i,s-1] = ((pred.float().flatten() - y_val.flatten())**2).mean(0)
    return error

def scheduler(n, milestones = [1, 10, 100], steps_per_milestones = [1, 2, 3]):
    m = 0
    for mile in milestones:
        if n <= mile:
            return steps_per_milestones[m]
        m += 1
    return steps_per_milestones[-1]
