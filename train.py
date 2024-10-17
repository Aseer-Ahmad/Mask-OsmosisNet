#train.py
import sys
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam, SGD, RMSprop, Adagrad
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, LambdaLR
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import gc
import pandas as pd
from torchvision import transforms
sns.set_theme()
import torch._dynamo
torch._dynamo.reset()
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from InpaintingSolver.bi_cg_nn import BiCG_Net
from InpaintingSolver.bi_cg import OsmosisInpainting

from utils import get_dfStencil, get_bicgDict

SEED = 1
torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(SEED)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

class InvarianceLoss(nn.Module):
    """
    Inverse variance loss 
    1 / ( var^2  + eps) 
    """
    def __init__(self):
        super(InvarianceLoss, self).__init__()

    def forward(self, mask, eps = 1e-6):
        return torch.mean(1. / (torch.var(mask, dim=(2,3)) + eps) )

class DensityLoss(nn.Module):
    """
    Density loss 
    | ||c||_1 / (nx * ny)  - d | 
    """
    def __init__(self, density = 0.1):
        super(DensityLoss, self).__init__()
        self.density = density

    def forward(self, mask, eps = 1e-6):
        h, w = mask.shape[2], mask.shape[3]
        return torch.mean( torch.abs( (torch.norm(mask, p = 1, dim = (2, 3))/ (h*w)) 
                                     - self.density) )

class MSELoss(nn.Module):
    """
    Means squared loss : 
    """
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, U, V):
        nxny = U.shape[2] * U.shape[3]
        return torch.mean(torch.norm(U-V, p = 2, dim = (2,3))**2 / nxny)

class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_steps: int, final_lr: float, base_lr: float = 1e-6, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.final_lr = final_lr
        self.base_lr = base_lr
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                self.base_lr + (self.final_lr - self.base_lr) * (self.last_epoch / self.warmup_steps)
                for _ in self.optimizer.param_groups
            ]
        else:
            # After warmup, we keep the learning rate fixed
            return [self.final_lr for _ in self.optimizer.param_groups]
    
class MyCustomTransform(torch.nn.Module):
    def forward(self, img):  
        return (img * 255).type(torch.uint8)
    
class MyCustomTransform2(torch.nn.Module):
    def forward(self, img):  
        return  torch.from_numpy(np.array(img)).unsqueeze(0)

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


class ModelTrainer():

    def __init__(self, output_dir, optimizer, scheduler, lr, weight_decay, momentum, train_batch_size, test_batch_size):
        
        self.output_dir= output_dir
        self.optimizer= optimizer
        self.scheduler= scheduler
        self.lr= lr
        self.weight_decay= weight_decay
        self.momentum = momentum
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device : {self.device}")

    def getOptimizer(self, model):
        
        opt = None
        
        if self.optimizer == "Adam":
            opt = Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay = self.weight_decay)

        elif self.optimizer == "SGD":
            opt = SGD(model.parameters(), lr=self.lr, momentum = self.momentum, weight_decay = self.weight_decay)

        elif self.optimizer == "SGD-Nestrov":
            opt = SGD(model.parameters(), lr=self.lr, nesterov = True, momentum = self.momentum , weight_decay = self.weight_decay)

        elif self.optimizer == "AdamW":
            opt = AdamW(model.parameters(), lr=self.lr, weight_decay = self.weight_decay)

        elif self.optimizer == "RMSprop":
            opt = RMSprop(model.parameters(), lr=self.lr, weight_decay = self.weight_decay)

        elif self.optimizer == "Adagrad":
            opt = Adagrad(model.parameters(), lr=self.lr, momentum = self.momentum, weight_decay = self.weight_decay)

        return opt

    def getScheduler(self, optim):
        
        schdl = None

        if self.scheduler == "exp":
            schdl = ExponentialLR(optim, gamma=0.9)
        
        elif self.scheduler == "multiStep":
            schdl = MultiStepLR(optim, milestones=[30,80], gamma=0.1)

        elif self.scheduler == "lambdaLR":
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
    
    def loadCheckpoint(self, model, optimizer, path):
        
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, optimizer

    def saveCheckpoint(self, model, optimizer, fname):
        
        torch.save({
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(self.output_dir, fname))

    def getDataloaders(self, train_dataset, test_dataset, img_size):

        transform = transforms.Compose([
                    transforms.Resize((img_size, img_size), antialias = True),
                    transforms.Grayscale(),
                    transforms.ToTensor(),   
                ])

        transform_norm = transforms.Compose([
                    transforms.Resize((img_size, img_size), antialias = True),
                    transforms.Grayscale(),
                    transforms.ToTensor(), 
                    transforms.Normalize(mean = [0.44531356896770125], std = [0.2692461874154524])
                ])
        
        transform_scale = transforms.Compose([
                    transforms.Resize((img_size, img_size), antialias = True),
                    transforms.Grayscale(),
                    MyCustomTransform2()
                ])

        def custom_collate_fn(batch):
            
            images       = []
            images_scale = []

            for item in batch :
                images.append(transform(item['image']))
                images_scale.append(transform_scale(item['image']))

            images       = torch.stack(images)
            images_scale = torch.stack(images_scale)
            
            return images, images_scale

        train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size=self.train_batch_size, collate_fn=custom_collate_fn)
        test_dataloader  = DataLoader(test_dataset, shuffle = True, batch_size=self.test_batch_size, collate_fn=custom_collate_fn)

        # train_dataloader = DataLoader(train_dataset, shuffle = False, batch_size=self.train_batch_size, worker_init_fn=seed_worker,generator=g)
        # test_dataloader  = DataLoader(test_dataset, shuffle = False, batch_size=self.test_batch_size, worker_init_fn=seed_worker,generator=g)

        print(f"train and test dataloaders created")
        print(f"total train batches  : {len(train_dataloader)}")
        print(f"total test  batches  : {len(test_dataloader)}")

        return train_dataloader, test_dataloader

    def hardRoundBinarize(self, mask):
        return torch.floor(mask + 0.5)

    def mean_density(self, mask):
        return torch.mean(torch.norm(mask, p = 1, dim = (2, 3)) / (mask.shape[2]*mask.shape[3]))

    def validate(self, model, test_dataloader, density, alpha1, alpha2):
        print("validating on test dataset")
        running_loss = 0.0

        invLoss = InvarianceLoss()
        denLoss  = DensityLoss(density)

        td_len = len(test_dataloader)
        
        model.eval()

        with torch.no_grad():
            st = time.time()
            for i, (X, X_scale) in enumerate(test_dataloader):

                X = X.to(self.device, dtype=torch.float64)

                mask = model(X) # non-binary [0,1]
                loss1 = invLoss(mask)
                # mask_bin = self.hardRoundBinarize(mask) # binarized {0,1}
                loss2 = denLoss(mask)

                osmosis = OsmosisInpainting(None, X, mask, mask, offset=0, tau=7000, device = self.device, apply_canny=False)
                osmosis.calculateWeights(False, False, False)
                loss3, tts = osmosis.solveBatchParallel(1, save_batch = [False], verbose = False)

                total_loss = loss3 + loss2 * alpha2 + loss1 * alpha1 

                running_loss += total_loss.item()
            et = time.time()
        
        val_loss = running_loss / ((i+1)*td_len)
        print(f"\nvalidation loss : {val_loss} , total running time : {(et-st)/60.} min")
        print()

        return val_loss

    # Function to check for exploding/vanishing gradients
    def check_gradients(self, model):
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

    # Register a hook to inspect gradients
    def inspect_gradients(self, module, grad_input, grad_output):
        print(f"Layer: {module}")
        print(f"grad_input: {grad_input}")
        print(f"grad_output: {grad_output}\n")

    def initialize_weights_he(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_weights_xavier(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def train(self, model, epochs, alpha1, alpha2, mask_density, img_size, resume_checkpoint_file, save_every, batch_plot_every, val_every, train_dataset ,test_dataset):
        
        # loss lists
        loss1_list, loss2_list, loss3_list, gradnorm_list, running_loss_list = [], [], [], [], []
        avg_den_list, epochloss_list = [], []
        val_list = []

        train_dataloader, test_dataloader = self.getDataloaders(train_dataset, test_dataset, img_size)

        optimizer = self.getOptimizer(model)
        scheduler = self.getScheduler(optimizer)
        # scheduler = WarmupScheduler(optimizer, warmup_steps=5, final_lr=self.lr, base_lr=1e-5)

        print(f"optimizer : {self.optimizer}, scheduler : {self.scheduler} loaded")

        if resume_checkpoint_file != None:
            print(f"loading checkpoint file : {resume_checkpoint_file}")
            model, optimizer = self.loadCheckpoint(model, optimizer, resume_checkpoint_file)
            print(f"model, opt, schdl loaded from checkpoint")

        print(f"optimizer : {optimizer}")
        print(f"scheduler : {scheduler}")

        model = model.double()
        model.to(self.device)
        
        # bicg_model = BiCG_Net(offset = 0, tau = 7000., b=self.train_batch_size, c=1, nx=img_size, ny=img_size)
        # bicg_model = bicg_model.double()
        # bicg_model.to(self.device)

        print(f"initializing weights using Kaiming/He Initialization")
        model.apply(self.initialize_weights_he)

        # Attach hooks to layers
        # for layer in model.children():
        #     layer.register_full_backward_hook(self.inspect_gradients)

        invLoss = InvarianceLoss()
        denLoss = DensityLoss(density = mask_density)
        mseLoss = MSELoss()

        torch.autograd.set_detect_anomaly(True)

        print("\nbeginning training ...")

        st_tt = time.time()

        for epoch in range(epochs):

            running_loss = 0.0
            running_mse = 0.0
            running_inv = 0.0
            skipped_batches = 0
            model.train()
            st = time.time()

            df_stencils = get_dfStencil()

            for i, (X, X_scale) in enumerate(train_dataloader, start = 1): 
                
                bicg_mat = get_bicgDict()
                
                print(f'Epoch {epoch}/{epochs} , batch {i}/{len(train_dataloader)} ')
                # df_stencils["f_name"].append(name)

                # data prep
                X = X.to(self.device, dtype=torch.float64) 

                # mask model
                mask  = model(X) # non-binary [0,1]

                # mask losses
                loss1 = invLoss(mask)
                loss2 = denLoss(mask)

                # osmosis solver
                osmosis = OsmosisInpainting(None, X, mask, mask, offset=8, tau=4096, device = self.device, apply_canny=False)
                osmosis.calculateWeights(d_verbose=False, m_verbose=False, s_verbose=False)
                
                if (i) % batch_plot_every == 0: 
                    save_batch = [True, os.path.join(self.output_dir, "imgs", f"batch_epoch_{str(epoch)}_iter_{str(i)}.png")]
                else:
                    save_batch = [False]
                df_stencils["iter"].append(i)
                loss3, tts, df_stencils, bicg_mat = osmosis.solveBatchParallel(df_stencils, bicg_mat, 1, save_batch = save_batch, verbose = False)
                
                # tts = 0
                # X_rec = bicg_model(X, mask, mask)
                # loss3 = mseLoss(X, X_rec)

                total_loss = loss3 + loss1 * alpha1 
                total_loss.backward()
                total_norm = self.check_gradients(model)

                
                # write forward backward stencils
                df_stencils["grad_norm"].append(total_norm)
                df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in df_stencils.items()]))
                df.to_csv( os.path.join(self.output_dir, "stencils.csv"), sep=',', encoding='utf-8', index=False, header=True)

                bicg_mat["grad_norm"].append(total_norm)
                df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in bicg_mat.items()]))
                df.to_csv( os.path.join(self.output_dir, f"bicg_wt_{i}.csv"), sep=',', encoding='utf-8', index=False, header=True)

                if total_norm > 50. :
                    skipped_batches += 1
                    print(f"skipping batch due to higher gradient norm : {total_norm}, total skipped : {skipped_batches}")
                    optimizer.zero_grad()
                    continue

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                running_loss += total_loss.item()
                running_mse  += loss3.item()
                running_inv  += loss1.item()
                avg_den = self.mean_density(mask)

                avg_den_list.append(avg_den.item())
                loss1_list.append(running_inv / (i - skipped_batches))
                loss2_list.append(loss2.item())
                loss3_list.append(running_mse / (i - skipped_batches))
                running_loss_list.append((running_loss / (i - skipped_batches)))
                gradnorm_list.append(total_norm)
                print(f"invariance loss : {loss1}, avg_den : {avg_den.item()}, ", end='')
                print(f"density loss : {loss2}, solver time : {str(tts)} sec , ", end='')
                print(f"mse loss : {loss3}, ", end='')
                print(f"total loss : {total_loss}, " , end = '')
                print(f"running loss : {running_loss / (i - skipped_batches)}" )

                if (i) % save_every == 0:
                    print("saving checkpoint")
                    fname = f"ckp_epoch_{str(epoch+1)}_iter_{str(i)}.pt"
                    self.saveCheckpoint(model, optimizer, fname)

                    # update mask distribution plot and save
                    print("plotting mask distribution")
                    fname = f"mdist_epoch_{str(epoch+1)}_iter_{str(i)}.png"
                    mask_flat = mask.reshape(-1).detach().cpu().numpy()
                    plot = sns.displot(mask_flat, kde=True)
                    fig = plot.figure
                    plot.set(xlabel='prob', ylabel='freq')
                    fig.savefig(os.path.join(self.output_dir, fname) ) 
                    plt.close(fig)

                # update plot and save
                clist = [l for l in range(1, len(loss1_list) + 1)]
                save_plot([np.log(loss1_list), np.log(loss3_list), np.log(running_loss_list)], clist, ["invloss", "mse", "runningloss"], os.path.join(self.output_dir, "all_losses.png"))
                save_plot([running_loss_list], clist, ["running loss"], os.path.join(self.output_dir, "runloss.png"))
                save_plot([loss1_list], clist, ["run invariance loss"], os.path.join(self.output_dir, "invloss.png"))
                save_plot([loss3_list], clist, ["run mse loss"], os.path.join(self.output_dir, "mseloss.png"))
                save_plot([loss2_list, avg_den_list], clist, ["density loss", "avg den"], os.path.join(self.output_dir, "loss_density.png"))
                save_plot([gradnorm_list], clist, ["grad norm"], os.path.join(self.output_dir, "gradnorm.png"))
                
                # update csv file and save
                train_dict = {
                    "grand norms" : gradnorm_list,
                    "density loss" : loss2_list,
                    "running invariance loss" : loss1_list,
                    "running mse loss" : loss3_list,
                    "running loss" : running_loss_list
                    }
                df = pd.DataFrame(train_dict)
                print(df.tail(20))
                fname = "data.csv"
                df.to_csv( os.path.join(self.output_dir, fname), sep=',', encoding='utf-8', index=False, header=True)

                # validate
                if (i) % val_every == 0:
                    val_loss = self.validate(model, test_dataloader, mask_density, alpha1, alpha2)
                    val_list.append(val_loss)

                # update val csv file and save
                val_dict = {
                    "val loss" : val_list
                    }
                df = pd.DataFrame(val_dict)
                fname = "val.csv"
                df.to_csv( os.path.join(self.output_dir, fname), sep=',', encoding='utf-8', index=False, header=True)

            et = time.time()
            print(f"total time for epoch : {str((et-st) / 60)} min")
            print(f"lr rate for the epoch : {optimizer.param_groups[0]['lr']}")
            epoch_loss = running_loss / (train_dataloader.__len__() - skipped_batches)
            epochloss_list.append(epoch_loss.item())
            print('Epoch [{}/{}], epoch Loss: {:.4f}'.format(epoch+1, epochs, epoch_loss))
            save_plot([epochloss_list], [i for i in range(epoch+1)], ["epoch loss"], os.path.join(self.output_dir, "epochloss.png"))

            print("\ncleaning torch mem and cache . End of epoch")
            torch.cuda.empty_cache()

        et_tt = time.time()
        print(f"total time for training : {str((et_tt-st_tt) / 3600)} hr")
