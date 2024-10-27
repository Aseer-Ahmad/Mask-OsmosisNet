# utils.py
from torch.optim import AdamW, Adam, SGD, RMSprop, Adagrad
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, LambdaLR
import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

def save_plot(loss_lists, x, legend_list, save_path):
    """
    Plots multiple data series from y against x and saves the plot.

    Parameters:
    x (list): A list of x-values.
    y (list of lists): A list containing lists of y-values.
    save_path (str): Path to save the resulting plot.
    """
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    for y_series in loss_lists:
        plt.plot(x, y_series)

    plt.xlabel('iter')
    plt.legend(legend_list, loc="best")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def getOptimizer(model, opt_config):

    optimizer    = opt_config['opt']
    lr           = opt_config['lr']
    momentum     = opt_config['momentum']
    weight_decay = opt_config['wd']
    
    opt = None
    
    if optimizer == "Adam":
        opt = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay = weight_decay)

    elif optimizer == "SGD":
        opt = SGD(model.parameters(), lr=lr, momentum = momentum, weight_decay = weight_decay)

    elif optimizer == "SGD-Nestrov":
        opt = SGD(model.parameters(), lr=lr, nesterov = True, momentum = momentum , weight_decay = weight_decay)

    elif optimizer == "AdamW":
        opt = AdamW(model.parameters(), lr=lr, weight_decay = weight_decay)

    elif optimizer == "RMSprop":
        opt = RMSprop(model.parameters(), lr=lr, weight_decay = weight_decay)

    elif optimizer == "Adagrad":
        opt = Adagrad(model.parameters(), lr=lr, momentum = momentum, weight_decay = weight_decay)

    return opt


def getScheduler(optim, scheduler):
    
    schdl = None

    if scheduler == "exp":
        schdl = ExponentialLR(optim, gamma=0.9)
    
    elif scheduler == "multiStep":
        schdl = MultiStepLR(optim, milestones=[1,2,3], gamma=0.1)

    elif scheduler == "lambdaLR":
        lambda1 = lambda epoch: epoch // 30
        lambda2 = lambda epoch: 0.95 ** epoch
        schdl = LambdaLR(optim, lr_lambda=[lambda2])

    return schdl


def check_gpu_memory():
    gpu_mem = torch.cuda.memory_allocated() / 1e6
    gpu_mem_max = torch.cuda.max_memory_allocated() / 1e6

    print(f"GPU Memory Allocated: {gpu_mem} MB")
    print(f"GPU Max Memory Allocated: {gpu_mem_max} MB")
    return gpu_mem, gpu_mem_max

def loadCheckpoint(model, optimizer, path):
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

def saveCheckpoint(model, optimizer, output_dir, fname):
    
    torch.save({
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(output_dir, fname))

class MyCustomTransform(torch.nn.Module):
    def forward(self, img):  
        return (img * 255).type(torch.uint8)
    
class MyCustomTransform2(torch.nn.Module):
    def forward(self, img):  
        return  torch.from_numpy(np.array(img)).unsqueeze(0)
    
def initialize_weights_he(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def init_weights_xavier(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def check_gradients(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2).item()  # L2 norm of gradients
            print(f"gradient norm in {p.name} layer : {param_norm}")
        else : 
            param_norm = 0.
            print(f"gradient norm in {p.name} layer : zero")
        total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    print(f"Total gradient norm: {total_norm}")
    return total_norm

def inspect_gradients(module, grad_input, grad_output):
    print(f"Layer: {module}")
    print(f"grad_input: {grad_input}")
    print(f"grad_output: {grad_output}\n")

def mean_density(mask):
    return torch.mean(torch.norm(mask, p = 1, dim = (2, 3)) / (mask.shape[2]*mask.shape[3]))

def hardRoundBinarize(mask):
    return torch.floor(mask + 0.5)

def get_bicgDict():
    bicg_mat = {
    "f_name" : [],
    "bicg_iter" : [],

    "restart" : [],
    "no_restart" : [],
    "no_restart_1f" : [],
    "no_restart_2f" : [],

    "p_forward_max"  : [],
    "p_forward_min"  : [],
    "p_forward_mean" : [],
    "p_backward_max"  : [],
    "p_backward_min"  : [],
    "p_backward_mean" : [],

    "r_forward_max"  : [],
    "r_forward_min"  : [],
    "r_forward_mean" : [],
    "r_backward_max"  : [],
    "r_backward_min"  : [],
    "r_backward_mean" : [],

    "r_old_forward_max"  : [],
    "r_old_forward_min"  : [],
    "r_old_forward_mean" : [],
    "r_old_backward_max"  : [],
    "r_old_backward_min"  : [],
    "r_old_backward_mean" : [],

    "s_forward_max"  : [],
    "s_forward_min"  : [],
    "s_forward_mean" : [],
    "s_backward_max"  : [],
    "s_backward_min"  : [],
    "s_backward_mean" : [],

    "sigma_forward_max"  : [],
    "sigma_forward_min"  : [],
    "sigma_forward_mean" : [],
    "sigma_backward_max"  : [],
    "sigma_backward_min"  : [],
    "sigma_backward_mean" : [],

    "v_forward_max"  : [],
    "v_forward_min"  : [],
    "v_forward_mean" : [],
    "v_backward_max"  : [],
    "v_backward_min"  : [],
    "v_backward_mean" : [],

    "r_0_forward_max"  : [],
    "r_0_forward_min"  : [],
    "r_0_forward_mean" : [],
    "r_0_backward_max"  : [],
    "r_0_backward_min"  : [],
    "r_0_backward_mean" : [],

    "alpha_forward_max"  : [],
    "alpha_forward_min"  : [],
    "alpha_forward_mean" : [],
    "alpha_backward_max"  : [],
    "alpha_backward_min"  : [],
    "alpha_backward_mean" : [],

    "t_forward_max"  : [],
    "t_forward_min"  : [],
    "t_forward_mean" : [],
    "t_backward_max"  : [],
    "t_backward_min"  : [],
    "t_backward_mean" : [],

    "beta_forward_max"  : [],
    "beta_forward_min"  : [],
    "beta_forward_mean" : [],
    "beta_backward_max"  : [],
    "beta_backward_min"  : [],
    "beta_backward_mean" : [],

    "omega_forward_max"  : [],
    "omega_forward_min"  : [],
    "omega_forward_mean" : [],
    "omega_backward_max"  : [],
    "omega_backward_min"  : [],
    "omega_backward_mean" : [],

    "grad_norm" : []
    }

    return bicg_mat

def get_dfStencil():

    df_stencils = {
        "iter" : [],
        "f_name" : [],

        "d1_forward_max"  : [],
        "d1_forward_min"  : [],
        "d1_forward_mean" : [],
        "d1_backward_max"  : [],
        "d1_backward_min"  : [],
        "d1_backward_mean" : [],

        "d2_forward_max"  : [],
        "d2_forward_min"  : [],
        "d2_forward_mean" : [],
        "d2_backward_max"  : [],
        "d2_backward_min"  : [],
        "d2_backward_mean" : [],

        "boo_forward_max"  : [],
        "boo_forward_min"  : [],
        "boo_forward_mean" : [],
        "boo_backward_max"  : [],
        "boo_backward_min"  : [],
        "boo_backward_mean" : [],

        "bmo_forward_max"  : [],
        "bmo_forward_min"  : [],
        "bmo_forward_mean" : [],
        "bmo_backward_max"  : [],
        "bmo_backward_min"  : [],
        "bmo_backward_mean" : [],

        "bop_forward_max"  : [],
        "bop_forward_min"  : [],
        "bop_forward_mean" : [],
        "bop_backward_max"  : [],
        "bop_backward_min"  : [],
        "bop_backward_mean" : [],

        "bpo_forward_max"  : [],
        "bpo_forward_min"  : [],
        "bpo_forward_mean" : [],
        "bpo_backward_max"  : [],
        "bpo_backward_min"  : [],
        "bpo_backward_mean" : [],

        "bom_forward_max"  : [],
        "bom_forward_min"  : [],
        "bom_forward_mean" : [],
        "bom_backward_max"  : [],
        "bom_backward_min"  : [],
        "bom_backward_mean" : [],

        "grad_norm" : []
        }


    return df_stencils

