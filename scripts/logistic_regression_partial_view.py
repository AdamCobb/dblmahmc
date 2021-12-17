import torch
import hamiltorch
import URSABench as mctestbed
import argparse

import inference
from util import *

import torch.nn as nn
import torch.nn.functional as F

print('####################')
print('Version: ',hamiltorch.__version__)
print('####################')


'''set up hyperparameters of the experiments'''
parser = argparse.ArgumentParser(description='Logistic regression')
parser.add_argument('--num_samples', type=int, default=8000) #100
parser.add_argument('--device', type=int, default=0) #100
parser.add_argument('--step_size', type=float, default=0.001)
parser.add_argument('--thinning', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--MH_burn_in', type=int, default=2000)
parser.add_argument('--prior_prec', type=float, default=100.)
parser.add_argument('--milestones_con', nargs='+', default = [2000, 3000, 4000])
parser.add_argument('--milestones_track', nargs='+', default = [2000, 3000, 4000])
parser.add_argument('--steps_per_milestone_con', nargs='+', default = [1, 2, 3])
parser.add_argument('--steps_per_milestone_track', nargs='+', default = [1, 2, 3])
parser.add_argument('--save', type=str, help='Save name', default = 'MH')
parser.add_argument('--inference', type=str, default='HMC')

args = parser.parse_args()
print (args)


### Import data sets
from URSABench import datasets, models
hamiltorch.set_random_seed(args.seed)

model_cfg = getattr(models, 'MLP200MNIST')

loaders, num_classes = datasets.loaders(
            'MNIST',
            '../../data/',
            128,
            4,
            transform_train=model_cfg.transform_train,
            transform_test=model_cfg.transform_test,
            shuffle_train=True,
            use_validation=True,
            val_size=0.1,
            split_classes=None,
            imbalance  = False
        )

train_loader = loaders['train']
val_loader = loaders['test']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.device)
    print(torch.cuda.current_device())

x_val = []
y_val = []
for batch_idx, (data, target) in enumerate(val_loader):
#     print(batch_idx)
    x_val.append(data)#.to(device)
    y_val.append(target)#.to(device)
    # continue
x_val = torch.cat(x_val)
y_val = torch.cat(y_val)

x_train = []
y_train = []
for batch_idx, (data, target) in enumerate(train_loader):
#     print(batch_idx)
    x_train.append(data)#.to(device)
    y_train.append(target)#.to(device)
    # continue
x_train = torch.cat(x_train)
y_train = torch.cat(y_train)

### network

class Net(nn.Module):

    def __init__(self, layer_sizes, bias=True):
        super(Net, self).__init__()
        self.layer_sizes = layer_sizes
        self.bias = bias
        self.l1 = nn.Linear(layer_sizes[0], layer_sizes[1], bias = self.bias)

    def forward(self, x):
        x = x.view(-1, self.layer_sizes[0])
        x = self.l1(x)
        return x

layer_sizes = [784,10]
net = Net(layer_sizes)

#### Posterior
crit = nn.CrossEntropyLoss(reduction='sum')
log_regression = hamiltorch.util.make_functional(net)

def define_local_log_posterior(X, Y, prec_agent, n_agents):
    """
    X: (DxN)
    Y: (Nx1)
    """
    def agent_fun(param):
        """
        param: (D)
        messages: (MxD)
        """
        params_unflattened = hamiltorch.util.unflatten(net, param)
        output = log_regression(X, params=params_unflattened)

        log_likelihood = - (crit(output, Y.long().view(-1)))
        log_local_prior = - (prec_agent/2) * torch.sum(param ** 2)
#         log_consensus_prior = - (prec_messages/2) * torch.sum(torch.matmul((param.view(1,-1) - messages),(param.view(1,-1) - messages).t()).diag())

        log_post = log_likelihood + log_local_prior / n_agents # + log_consensus_prior
        return log_post
    return agent_fun


step_size = args.step_size
num_samples = args.num_samples
L = 1
net = Net(layer_sizes)
N_agents = 4

params_init = torch.zeros_like(hamiltorch.util.flatten(net).clone())
messages = torch.zeros(N_agents, params_init.shape[0])
X = x_train.clone() #.t()
Y = y_train.clone()
limit = -1#1000
tau_list = torch.tensor([1.,1.]) * 100
# Fully connected network
W = torch.ones(N_agents,N_agents)/N_agents # Doubly stochastic matrix set to 1/N_agents
thinning = args.thinning
N_agents = 4
prior_prec = args.prior_prec

if args.inference == 'HMC':
    # HMC
    L = 1
    steps = num_samples
    samples_MH = hamiltorch.sample_model(net.to(device), X.to(device), Y.to(device), params_init=params_init.to(device), model_loss='multi_class_linear_output', num_samples=steps,
                                   step_size=step_size, num_steps_per_sample=L, tau_list=tau_list, tau_out=1., store_on_GPU = False)

    nll_MH, acc_MH = metrics(torch.stack(samples_MH,0).unsqueeze(0), thinning, x_val, y_val, 1, net.to('cpu'), tau_list, num_samples)

if args.inference == 'dMALA' or 'DULA':

    X1 = torch.zeros_like(X[:limit].clone())
    X1[:,:,:14,:14] = X[:limit,:,:14,:14].clone()

    X2 = torch.zeros_like(X[:limit].clone())
    X2[:,:,14:,:14] = X[:limit,:,14:,:14].clone()

    X3 = torch.zeros_like(X[:limit].clone())
    X3[:,:,14:,14:] = X[:limit,:,14:,14:].clone()

    X4 = torch.zeros_like(X[:limit].clone())
    X4[:,:,:14,14:] = X[:limit,:,:14,14:].clone()

    X_list = [X1, X2, X3, X4]
    Agent_log_prob_func_list = []


    for X_local in X_list:
        Agent_log_prob_func_list.append(define_local_log_posterior(X_local.to(device), Y[:limit].to(device), prec_agent = prior_prec, n_agents=N_agents))

    hamiltorch.set_random_seed(0)
    mass_list = None

    milestones_con = args.milestones_con
    steps_per_milestone_con = args.steps_per_milestone_con
    milestones_track = args.milestones_track
    steps_per_milestone_track = args.steps_per_milestone_track

    consensus_step_scheduler = lambda x: scheduler(x, milestones = milestones_con, steps_per_milestones = steps_per_milestone_con)
    gradient_tracking_scheduler = lambda x: scheduler(x, milestones = milestones_track, steps_per_milestones = steps_per_milestone_track)
    MH_step = lambda x: 0 if x < args.MH_burn_in else 1

    param_local = [m for m in torch.zeros(N_agents, params_init.shape[0]).to(device)]
    if args.inference == 'dMALA':
        samples_MH = inference.decentralised_MALA(param_local, W, Agent_log_prob_func_list, mass_list, step_size,
                           num_samples, consensus_step_scheduler = consensus_step_scheduler,
                           gradient_tracking_scheduler = gradient_tracking_scheduler, grad_est = None,
                           local_grad_prev = None, MH_step=MH_step)


        nll_MH, acc_MH = metrics(samples_MH, thinning, x_val, y_val, N_agents, net, tau_list, num_samples)
    
    elif args.inference == 'DULA':
        A = torch.ones(N_agents,N_agents) - torch.eye(N_agents)
        samples_DULA = inference.decentralised_ULA(param_local, num_samples, A, Agent_log_prob_func_list, mass_list,
                                      step_size, consensus_step_size = 0.48, delta_step_size = 0.55,
                                      delta_consensus = 0.01, b1 = 230., b2 = 230 )
        
        nll_MH, acc_MH = metrics(samples_DULA, thinning, x_val, y_val, N_agents, net, tau_list, num_samples)

config_file = {'args':args, 'nll_MH': nll_MH, 'acc_MH': acc_MH}

torch.save(config_file, '../exp/MNIST_partial_config_{:}_{:}_'.format(args.seed,args.step_size)  + args.inference + args.save + '.npy')
