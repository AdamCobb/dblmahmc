import torch
import hamiltorch
# import URSABench as mctestbed
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
parser.add_argument('--num_samples', type=int, default=100000) #100
parser.add_argument('--device', type=int, default=0) #100
parser.add_argument('--step_size', type=float, default=0.0004)
parser.add_argument('--thinning', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--MH_burn_in', type=int, default=1000)
parser.add_argument('--prior_prec', type=float, default=1.)
parser.add_argument('--milestones_con', nargs='+', default = [1000, 2000, 3000])
parser.add_argument('--milestones_track', nargs='+', default = [1000, 2000, 3000])
parser.add_argument('--steps_per_milestone_con', nargs='+', default = [1, 5, 10])
parser.add_argument('--steps_per_milestone_track', nargs='+', default = [1, 5, 10])
parser.add_argument('--save', type=str, help='Save name', default = '')
parser.add_argument('--inference', type=str, default='HMC')

# Namespace(MH_burn_in=1000, device=0, inference='DULA', milestones_con=[1000, 2000, 3000], milestones_track=[1000, 2000, 3000], num_samples=4000, prior_prec=1.0, save='MH', seed=0, step_size=1e-07, steps_per_milestone_con=[1, 5, 10], steps_per_milestone_track=[1, 5, 10], thinning=10)

args = parser.parse_args()
print (args)


### Import data sets
# from URSABench import datasets, models
hamiltorch.set_random_seed(args.seed)

from sklearn.datasets import load_boston
import pandas as pd

bos = load_boston()
bos.keys()

df = pd.DataFrame(bos.data)
df.columns = bos.feature_names
df['Price'] = bos.target
df.head()


df.describe()


data = df[df.columns[:-1]]
data = data.apply(
    lambda x: (x - x.mean()) / x.std()
)

data['Price'] = df.Price

import numpy as np

X = data.drop('Price', axis=1).to_numpy()
Y = data['Price'].to_numpy()


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)



X_test = (X_test - X_train.mean(0))/X_train.std(0)
Y_test = (Y_test - Y_train.mean(0))/Y_train.std(0)
Y_train = (Y_train - Y_train.mean(0))/Y_train.std(0)
X_train = (X_train - X_train.mean(0))/X_train.std(0)



n_train = X_train.shape[0]
x_train = torch.tensor(X_train, dtype=torch.float)
x_val = torch.tensor(X_test, dtype=torch.float)
y_train = torch.tensor(Y_train, dtype=torch.float).view(-1, 1)
y_val = torch.tensor(Y_test, dtype=torch.float).view(-1, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.device)
    print(torch.cuda.current_device())

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

layer_sizes = [13,1]
net = Net(layer_sizes)

#### Posterior
crit = nn.MSELoss(reduction='sum')
log_regression = hamiltorch.util.make_functional(net)

def define_local_log_posterior(X, Y, prec_agent, out_prec, n_agents):
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

        log_likelihood = - 0.5 * out_prec * (crit(output.view(-1), Y.view(-1)))
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
out_prec = 10.

if args.inference == 'HMC':
    # HMC
    L = 1
#     step_size = 0.002
    steps = num_samples

#     hamiltorch.set_random_seed(123)

    samples = hamiltorch.sample_model(net, X, Y, params_init=params_init, model_loss='regression', num_samples=steps,
                                   step_size=step_size, num_steps_per_sample=L, tau_list=tau_list, tau_out=out_prec)

    MSE = metrics_reg(torch.stack(samples,0).unsqueeze(0), thinning, x_val, y_val, 1, net, tau_list, num_samples, out_prec)

if args.inference == 'dMALA' or 'DULA':

    X1 = torch.zeros_like(X[:].clone())
    X1[:,:3] = X[:,:3].clone()

    X2 = torch.zeros_like(X[:].clone())
    X2[:,3:6] = X[:,3:6].clone()

    X3 = torch.zeros_like(X[:].clone())
    X3[:,6:9] = X[:,6:9].clone()

    X4 = torch.zeros_like(X[:].clone())
    X4[:,9:] = X[:,9:].clone()

    X_list = [X1, X2, X3, X4]
    Agent_log_prob_func_list = []


    for X_local in X_list:

        Agent_log_prob_func_list.append(define_local_log_posterior(X_local.to(device), Y[:].to(device), prec_agent = prior_prec, out_prec = out_prec, n_agents=N_agents))

    # hamiltorch.set_random_seed(0)
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

        samples = inference.decentralised_MALA(param_local, W, Agent_log_prob_func_list, mass_list, step_size,
                           num_samples, consensus_step_scheduler = consensus_step_scheduler,
                           gradient_tracking_scheduler = gradient_tracking_scheduler, grad_est = None,
                           local_grad_prev = None, MH_step=MH_step)


        MSE = metrics_reg(samples, thinning, x_val, y_val, N_agents, net, tau_list, num_samples, out_prec)

    elif args.inference == 'DULA':
        A = torch.ones(N_agents,N_agents) - torch.eye(N_agents)
        samples = inference.decentralised_ULA(param_local, num_samples, A, Agent_log_prob_func_list, mass_list,
                                      step_size, consensus_step_size = 0.48, delta_step_size = 0.55,
                                      delta_consensus = 0.01, b1 = 230., b2 = 230 )

        MSE = metrics_reg(samples, thinning, x_val, y_val, N_agents, net, tau_list, num_samples, out_prec)



config_file = {'args':args, 'MSE': MSE, 'x_val':x_val, 'y_val':y_val}

torch.save(config_file, '../exp/boston_partial_config_{:}_{:}_'.format(args.seed,args.step_size)  + args.inference + args.save + '.npy')
