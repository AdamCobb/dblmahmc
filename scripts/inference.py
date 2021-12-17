import torch
import hamiltorch
from util import *

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from hamiltorch.samplers import collect_gradients


# param_local = [m for m in torch.zeros(N_agents, params_init.shape[0])]

def decentralised_MALA(param_local, W, agent_log_prob_list, mass_list, step_size, steps, consensus_step_scheduler = lambda x: 1,
 gradient_tracking_scheduler = lambda x: 1, grad_est = None, local_grad_prev = None, MH_step= lambda x: 1, burn = 0):
    """Algorithm for decentralised Metropolis Adjusted Langevin Dynamics.

    Parameters
    ----------
    param_local : type
        Description of parameter `param_local`.
    W : type
        Description of parameter `W`.
    agent_log_prob_list : type
        Description of parameter `agent_log_prob_list`.
    mass_list : type
        Description of parameter `mass_list`.
    step_size : type
        Description of parameter `step_size`.
    steps : type
        Description of parameter `steps`.
    consensus_step_scheduler : type
        Description of parameter `consensus_step_scheduler`.
    gradient_tracking_scheduler : type
        Description of parameter `gradient_tracking_scheduler`.
    grad_est : type
        Description of parameter `grad_est`.
    local_grad_prev : type
        Description of parameter `local_grad_prev`.

    Returns
    -------
    type
        Description of returned object.

    """

    device = param_local[0].device
    N_agents = len(agent_log_prob_list)
    D = param_local[0].shape[0]
    num_rejected = torch.zeros(N_agents, )
    samples = torch.zeros(N_agents, steps, D)
    samples_pre_burn = torch.zeros(N_agents, D)

    if grad_est is None:
        # Initialise the gradients:
        grad_est = torch.zeros(N_agents, D).to(device)
        aleph_est = torch.zeros(N_agents,).to(device)
    if local_grad_prev is None:
        # Initialise the local gradients:
        local_grad_prev = torch.zeros(N_agents, D).to(device)
        local_quad_prev = torch.zeros(N_agents,).to(device)

    pbar = tqdm(range(burn + steps))
    # For Loop:
    for n in pbar:
        i = 0
        local_grad_current = torch.zeros(N_agents, D).to(device)
        local_momentum = torch.zeros(N_agents, D).to(device)
        local_quad_current = torch.zeros(N_agents,).to(device)
        for agent_fun, p in zip(agent_log_prob_list, param_local):
            # Sample momentum
            local_momentum[i] = hamiltorch.samplers.gibbs(p, mass=None)
            # gradient update: (message passing)
            local_grad_current[i] = params_grad(p.to(device), agent_fun)
            # hvp update: (message passing)
            vhp = torch.autograd.functional.vhp(agent_fun, p.clone().requires_grad_().to(device), local_momentum[i].to(device))[1].detach()
            local_quad_current[i] = - vhp.dot(local_momentum[i])
            i += 1
#         import pdb; pdb.set_trace()
        if gradient_tracking_scheduler(n) == 0:
            # Just set to local updates:
            grad_est = local_grad_current.clone().to(device)
            aleph_est = local_quad_current.clone().to(device)
        else:
            for s in range(gradient_tracking_scheduler(n)):
                # Perform:
                # g_i^{k+1} = ∑w_{ij}g_i^{k} + grad p^k+1 - grad p^{k}
                grad_est = torch.matmul(W.to(device), grad_est + local_grad_current - local_grad_prev)
                aleph_est = torch.matmul(W.to(device), aleph_est + local_quad_current - local_quad_prev)

        i = 0
        # set current local to previous
        local_grad_prev = local_grad_current.clone()
        local_quad_prev = local_quad_current.clone()

        # Run the Euler step and store all the agent updates
        i = 0
        for agent_fun, p, grad, aleph in zip(agent_log_prob_list, param_local, grad_est, aleph_est):
#             m = hamiltorch.samplers.gibbs(p, mass=None)
            # ham = hamiltonian(p, m, agent_fun, sampler=hamiltorch.Sampler.HMC, integrator=hamiltorch.Integrator.EXPLICIT, inv_mass=None)
            ret_params, ret_momenta = leapfrog(params=p.to(device), momentum=local_momentum[i].to(device), log_prob_func=agent_fun, steps=1, step_size=step_size, pass_grad = grad)

            # MH step # Might want to change to matrix mul at some point:
            error_term_2 = (grad.cpu() ** 2).sum(-1)
#             print(
            # min(0, (w_old - w_new)^T*dH/dw|w_old,p_old + (p_old - p_new)^T*dH/dw|w_old,p_old)
            H_delta = -0.5 * step_size ** 2 *(aleph.cpu() + error_term_2) # negative change in Hamiltoninian energy
            rho = min(0., H_delta.sum(0))
            if n % 100 == 0:
                print('aleph: ', aleph.cpu() * step_size ** 2, 'H_delta', H_delta.sum(0), rho, error_term_2)
            if MH_step(n):
                # Local MH step
                # In case nans or infs: reject
                if hamiltorch.util.has_nan_or_inf(ret_params):
                    num_rejected[i] += 1
                    
                    if n >= burn:
                        samples[i,n - burn] = p.cpu()
                    else:
                        samples_pre_burn[i] = p.cpu()
                elif rho >= torch.log(torch.rand(1)):
                    if n >= burn:
                        samples[i,n - burn] = ret_params.cpu()
                    else:
                        samples_pre_burn[i] = ret_params.cpu()
                else:
                    num_rejected[i] += 1
                    if n >= burn:
                        samples[i,n - burn] = p.cpu()
                    else:
                        samples_pre_burn[i] = p.cpu()
            else:
                if n >= burn:
                    samples[i,n - burn] = ret_params.cpu()
                else:
                    samples_pre_burn[i] = ret_params.cpu()
                # Reset MH error as not being used when MH step is switched off.
                aleph_est = torch.zeros(N_agents,).to(device)
            i += 1

            #HVP:

            # https://pytorch.org/docs/stable/autograd.html


        # Update local parameters at end of agent steps
        if n >= burn:
            param_local = samples[:,n - burn]
        else:
            param_local = samples_pre_burn

        # Consensus Step By averaging Neighbours (i.e all parameters here)
        for s in range(consensus_step_scheduler(n)):
#             import pdb; pdb.set_trace()
            param_local = torch.matmul(W, param_local)

    for a in range(N_agents):
        print(num_rejected[a])
        print('Agent {:}: Acceptance Rate {:.2f}'.format(a, 1. - num_rejected[a]/steps))
    return samples

def decentralised_ULA(param_local, steps, Adjacency, agent_log_prob_list, mass_list, step_size = 0.00032, consensus_step_size = 0.48, delta_step_size = 0.55, delta_consensus = 0.01, b1 = 230., b2 = 230 ):
    """Algorithm 1 in paper: A DECENTRALIZED APPROACH TO BAYESIAN LEARNING.

    Returns
    -------
    type
        Description of returned object.

    """

    device = param_local[0].device
    N_agents = len(agent_log_prob_list)
    D = param_local[0].shape[0]
    num_rejected = torch.ones(N_agents, )
    param_local = torch.stack(param_local).to(device)
    samples = torch.zeros(N_agents, steps, D)
    # Renaming to same convention as Algo 1
    alpha = step_size
    beta = consensus_step_size


    if not (delta_consensus >= 0 and delta_step_size > 0.5  + delta_consensus and delta_step_size < 1.):
        raise RuntimeError('Decay rates are not consistent with conditions.')

    pbar = tqdm(range(steps))
    # For Loop:
    for n in pbar:

        # Note that delta_consensus >= 0 and 1. > delta_step_size > 0.5  + delta_consensus
        # decay consensus step size beta, and general step size, alpha
        alpha = step_size / ((n + b1) ** delta_step_size)
        beta = consensus_step_size / ((n + b2) ** delta_consensus)

        # gradient update: (message passing) local gradient
        i = 0
        g = torch.zeros(N_agents, D)
        v = torch.zeros(N_agents, D)
        for agent_fun, p in zip(agent_log_prob_list, param_local):
            # g = ∇U = - log p(w)
            g[i] = - params_grad(p.to(device), agent_fun)
            v[i] = hamiltorch.samplers.gibbs(p, mass=None)
            i += 1


        # Adjacency matrix part
        w_hat = torch.zeros(N_agents,D).to(device)
        for i in range(N_agents):
            w_hat[i] = (torch.matmul(Adjacency[i].to(device) , (param_local[i] - param_local)))

        # update
        samples[:,n] = param_local.cpu() - beta * w_hat.cpu() - alpha * N_agents * g.cpu() + ((2 * alpha) ** 0.5) * v

        # Update local parameters at end of agent steps
        param_local = samples[:,n].to(device)

        if n % 1000 == 0:
            print('Step_size: ',alpha, ' Consensus: ', beta)

    return samples


def decentralised_stochastic_ULA(param_local, steps, Adjacency, agent_log_prob_list, dataloader_list, mass_list, step_size = 0.00032, consensus_step_size = 0.48, delta_step_size = 0.55, delta_consensus = 0.01, b1 = 230., b2 = 230 ):
    """Algorithm 1 in paper: A DECENTRALIZED APPROACH TO BAYESIAN LEARNING.

    Returns
    -------
    type
        Description of returned object.

    """

    num_mini_batches = math.ceil(len(dataloader_list[0]) / batch_size)
    N_agents = len(agent_log_prob_list)
    D = param_local[0].shape[0]
    num_rejected = torch.ones(N_agents, )
    params_local = torch.stack(params_local)
    samples = torch.zeros(N_agents, steps, D)
    # Renaming to same convention as Algo 1
    alpha = step_size
    beta = consensus_step_size


    if not (delta_consensus >= 0 and delta_step_size > 0.5  + delta_consensus and delta_step_size < 1.):
        raise RuntimeError('Decay rates are not consistent with conditions.')

    pbar = tqdm(range(steps))
    # For Loop:
    count = 0
    for n in pbar:
        dataiter_list = []
        for a in dataloader_list:
            dataiter_list.append(iter(a))
        samples_epoch = torch.zeros(N_agents, num_mini_batches, D)
        for k in range(num_mini_batches):
            # Note that delta_consensus >= 0 and 1. > delta_step_size > 0.5  + delta_consensus
            # decay consensus step size beta, and general step size, alpha
            alpha = step_size / ((count + b1) ** delta_step_size)
            beta = consensus_step_size / ((count + b2) ** delta_consensus)

            # gradient update: (message passing) local gradient
            i = 0
            g = torch.zeros(N_agents, D)
            v = torch.zeros(N_agents, D)
            for agent_fun, p in zip(agent_log_prob_list, param_local):
                # g = ∇U = - log p(w)
                g[i] = - params_grad(p, agent_fun)
                v[i] = hamiltorch.samplers.gibbs(p, mass=None)
                i += 1


            # Adjacency matrix part
            w_hat = torch.zeros(N,D)
            for i in range(N_agents):
                w_hat[i] = (torch.matmul(Adjacency[i] , (params_local[i] - params_local)))

            # update
            # ToDo: Maybe only collect once every epoch
            samples_epoch[:,count] = params_local - beta * w_hat - alpha * N_agents * g + torch.sqrt(2 * alpha) * v
            count += 1
    return samples


### 1st order Euler
def leapfrog(params, momentum, log_prob_func, steps=10, step_size=0.1, jitter=0.01, normalizing_const=1., softabs_const=1e6, explicit_binding_const=100, fixed_point_threshold=1e-20, fixed_point_max_iterations=6, jitter_max_tries=10, inv_mass=None, ham_func=None, store_on_GPU = True, debug=False, pass_grad = None):
    """This is a rather large function that contains all the various integration schemes used for HMC. Broadly speaking, it takes in the parameters
    and momentum and propose a new set of parameters and momentum. This is a key part of hamiltorch as it covers multiple integration schemes.

    Parameters
    ----------
    params : torch.tensor
        Flat vector of model parameters: shape (D,), where D is the dimensionality of the parameters.
    momentum : torch.tensor
        Flat vector of momentum, corresponding to the parameters: shape (D,), where D is the dimensionality of the parameters.
    log_prob_func : function
        A log_prob_func must take a 1-d vector of length equal to the number of parameters that are being sampled.
    steps : int
        The number of steps to take per trajector (often referred to as L).
    step_size : float
        Size of each step to take when doing the numerical integration.
    jitter : float
        Jitter is often added to the diagonal to the metric tensor to ensure it can be inverted.
        `jitter` is a float corresponding to scale of random draws from a uniform distribution.
    normalizing_const : float
        This constant is currently set to 1.0 and might be removed in future versions as it plays no immediate role.
    softabs_const : float
        Controls the "filtering" strength of the negative eigenvalues. Large values -> absolute value. See Betancourt 2013.
    explicit_binding_const : float
        Only relevant to Explicit RMHMC. Corresponds to the binding term in Cobb et al. 2019.
    fixed_point_threshold : float
        Only relevant for Implicit RMHMC. Sets the convergence threshold for 'breaking out' of the while loop for the generalised leapfrog.
    fixed_point_max_iterations : int
        Only relevant for Implicit RMHMC. Limits the number of fixed point iterations in the generalised leapforg.
    jitter_max_tries : float
        Only relevant for RMHMC. Number of attempts at resampling the jitter for the Fisher Information before raising a LogProbError.
    inv_mass : torch.tensor or list
        The inverse of the mass matrix. The inv_mass matrix is related to the covariance of the parameter space (the scale we expect it to vary). Currently this can be set
        to either a diagonal matrix, via a torch tensor of shape (D,), or a full square matrix of shape (D,D). There is also the capability for some
        integration schemes to implement the inv_mass matrix as a list of blocks. Hope to make that more efficient.
    ham_func : type
        Only related to semi-separable HMC. This part of hamiltorch has not been fully integrated yet.
    sampler : Sampler
        Sets the type of sampler that is being used for HMC: Choice {Sampler.HMC, Sampler.RMHMC, Sampler.HMC_NUTS}.
    integrator : Integrator
        Sets the type of integrator to be used for the leapfrog: Choice {Integrator.EXPLICIT, Integrator.IMPLICIT, Integrator.SPLITTING,
        Integrator.SPLITTING_RAND, Integrator.SPLITTING_KMID}.
    metric : Metric
        Determines the metric to be used for RMHMC. E.g. default is the Hessian hamiltorch.Metric.HESSIAN.
    store_on_GPU : bool
        Option that determines whether to keep samples in GPU memory. It runs fast when set to TRUE but may run out of memory unless set to FALSE.
    debug : int
        This is useful for checking how many iterations RMHMC takes to converge. Set to zero for no print statements.
    pass_grad : None or torch.tensor #(maybe make it a function one day)
        If set to a torch.tensor, it is used as the gradient  shape: (D,), where D is the number of parameters of the model.

    Returns
    -------
    ret_params : list
        List of parameters collected in the trajectory. Note that explicit RMHMC returns a copy of two lists.
    ret_momenta : list
        List of momentum collected in the trajectory. Note that explicit RMHMC returns a copy of two lists.

    """

    params = params.clone(); momentum = momentum.clone()
    # TodO detach graph when storing ret_params for memory saving
    # Calculate gradient at the start for the momentum once uptdate
    # TODO: check if I can use the derivative at the later step.
    # Page 190 Geometric Numerical Integration (You can go either way with momentum and potential step)
    # To use the tracked gradient, I need to use an order 1 Euler integrator rather than a symplectic one.
    # This will be a worse integrator but uses the correct gradient.
    momentum += step_size * pass_grad


    if inv_mass is None:
        # dparams_dt = momentum
        params = params + step_size * momentum #/normalizing_const
    else:
        #Assum G is diag here so 1/Mass = G inverse
        if type(inv_mass) is list:
            i = 0
            for block in inv_mass:
                it = block[0].shape[0]
                params[i:it+i] = params[i:it+i] + step_size * torch.matmul(block,momentum[i:it+i].view(-1,1)).view(-1) #/normalizing_const
                i += it
        elif len(inv_mass.shape) == 2:
            # dparams_dt = torch.matmul(inv_mass,momentum.view(-1,1)).view(-1)
            params = params + step_size * torch.matmul(inv_mass,momentum.view(-1,1)).view(-1) #/normalizing_const
        else:
            # dparams_dt = inv_mass * momentum
            params = params + step_size * inv_mass * momentum #/normalizing_const
#             p_grad = params_grad(params)
#             momentum += step_size * p_grad
    return params, momentum

def params_grad(p, log_prob_func):
    p = p.detach().requires_grad_()
    log_prob = log_prob_func(p)
    # log_prob.backward()
    p = collect_gradients(log_prob, p)
    # print(p.grad.std())
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return p.grad
