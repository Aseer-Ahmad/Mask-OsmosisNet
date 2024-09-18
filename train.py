#train.py
import sys
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam, SGD
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

from InpaintingSolver.bi_cg import OsmosisInpainting

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

    def __init__(self, output_dir, optimizer, scheduler, lr, weight_decay, train_batch_size, test_batch_size):
        
        self.output_dir= output_dir
        self.optimizer= optimizer
        self.scheduler= scheduler
        self.lr= lr
        self.weight_decay= weight_decay
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device : {self.device}")

    def getOptimizer(self, model):
        
        opt = None
        
        if self.optimizer == "Adam":
            opt = Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay = self.weight_decay)

        elif self.optimizer == "SGD":
            opt = SGD(model.parameters(), lr=self.lr, momentum = self.weight_decay)
              
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
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, optimizer

    def saveCheckpoint(self, model, optimizer, fname):
        
        torch.save({
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(self.output_dir, fname))

    def getDataloaders(self, train_dataset, test_dataset):

        transform_norm = transforms.Compose([
                    transforms.Resize((128, 128), antialias = True),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = [0.44531356896770125], std = [0.2692461874154524])
                ])

        transform = transforms.Compose([
                    transforms.Resize((128, 128), antialias = True),
                    transforms.Grayscale(),
                    transforms.ToTensor()
                ])

        def custom_collate_fn(batch):
            
            images      = []
            images_norm = []

            for item in batch :
                images.append(transform(item['image']))
                images_norm.append(transform_norm(item['image']))

            images      = torch.stack(images)
            images_norm = torch.stack(images_norm)
            
            return images, images_norm

        # train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size=self.train_batch_size, collate_fn=custom_collate_fn)
        # test_dataloader  = DataLoader(test_dataset, shuffle = True, batch_size=self.test_batch_size, collate_fn=custom_collate_fn)

        train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size=self.train_batch_size)
        test_dataloader  = DataLoader(test_dataset, shuffle = True, batch_size=self.test_batch_size)

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
            for i, (X, X_norm) in enumerate(test_dataloader):

                X = X.to(self.device, dtype=torch.float64)
                X_norm = X_norm.to(self.device, dtype=torch.float64)

                mask = model(X_norm) # non-binary [0,1]
                loss1 = invLoss(mask)
                mask_bin = self.hardRoundBinarize(mask) # binarized {0,1}
                loss2 = denLoss(mask_bin)

                mask_detach = mask_bin.detach().clone()
                osmosis = OsmosisInpainting(None, X, mask_detach, mask_detach, offset=1, tau=700, device = self.device, apply_canny=False)
                osmosis.calculateWeights(False, False, False)
                loss3, tts = osmosis.solveBatchParallel(10, save_batch = [False], verbose = False)

                total_loss = loss3 + loss2 * alpha2 + loss1 * alpha1 

                running_loss += total_loss
            et = time.time()
        
        print(f"\nvalidation loss : {running_loss / (i*td_len)} , total running time : {(et-st)/60.} min")
        print()

    # Function to check for exploding/vanishing gradients
    def check_gradients(self, model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.detach().data.norm(2)  # L2 norm of gradients
            print(f"gradient norm in {p.name} layer : {param_norm.item()}")
            total_norm += param_norm.item() ** 2
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

    def train(self, model, epochs, alpha1, alpha2, mask_density, resume_checkpoint_file, save_every, batch_plot_every, val_every, train_dataset ,test_dataset):
        
        # loss lists
        loss1_list, loss2_list, loss3_list, gradnorm_list, running_loss_list = [], [], [], [], []
        avg_den_list, epochloss_list = [], []

        train_dataloader, test_dataloader = self.getDataloaders(train_dataset, test_dataset)

        optimizer = self.getOptimizer(model)
        scheduler = self.getScheduler(optimizer)
        print(f"optimizer : {self.optimizer}, scheduler : {self.scheduler} loaded")

        if resume_checkpoint_file != None:
            print(f"loading checkpoint file : {resume_checkpoint_file}")
            model, optimizer = self.loadCheckpoint(model, optimizer, resume_checkpoint_file)
            print(f"model, opt, schdl loaded from checkpoint")


        print(f"optimizer : {optimizer}")
        print(f"scheduler : {scheduler}")

        model = model.double()
        model.to(self.device)
        
        print(f"initializing weights using Kaiming/He Initialization")
        model.apply(self.initialize_weights_he)

        # Attach hooks to layers
        # for layer in model.children():
        #     layer.register_full_backward_hook(self.inspect_gradients)

        invLoss = InvarianceLoss()
        denLoss  = DensityLoss(density = mask_density)

        print("\ncleaning torch mem and cache")
        torch.cuda.empty_cache()

        print("\nbeginning training ...")

        st_tt = time.time()

        for epoch in range(epochs):

            running_loss = 0.0
            model.train()
            st = time.time()
            
            for i, (X, X_norm) in enumerate(train_dataloader, start = 1): 
                
                print(f'Epoch {epoch}/{epochs} , batch {i}/{len(train_dataloader)} ')

                X = X.to(self.device, dtype=torch.float64)
                X_norm = X_norm.to(self.device, dtype=torch.float64)

                mask = model(X_norm) # non-binary [0,1]
                loss1 = invLoss(mask)
                # mask_bin = self.hardRoundBinarize(mask) # binarized {0,1} # evaluation step
                loss2 = denLoss(mask)

                mask_detach = mask.detach().clone()
                osmosis = OsmosisInpainting(None, X, mask_detach, mask_detach, offset=1, tau=700, device = self.device, apply_canny=False)
                osmosis.calculateWeights(False, False, False)
                # loss2, tts = osmosis.solveBatchSeq(100, save_batch = True, verbose = False)
                
                if (i) % batch_plot_every == 0: 
                    save_batch = [True, os.path.join(self.output_dir, "imgs", f"batch_epoch_{str(epoch)}_iter_{str(i)}.png")]
                else:
                    save_batch = [False]
                
                loss3, tts = osmosis.solveBatchParallel(100, save_batch = save_batch, verbose = False)

                # if torch.isnan(loss3):
                #     print(f"input X : {X}")

                total_loss = loss3 + loss2 * alpha2 + loss1 * alpha1 
                total_loss.backward()

                total_norm = self.check_gradients(model)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                running_loss += total_loss
                avg_den = self.mean_density(mask)

                avg_den_list.append(avg_den.item())
                loss1_list.append(loss1.item())
                loss2_list.append(loss2.item())
                loss3_list.append(loss3.item())
                running_loss_list.append((running_loss / i).item())
                gradnorm_list.append(total_norm)
                print(f"invariance loss : {loss1}, avg_den : {avg_den.item()}, ", end='')
                print(f"density loss : {loss2}, solver time : {str(tts)} sec , ", end='')
                print(f"mse loss : {loss3}, ", end='')
                print(f"total loss : {total_loss}, " , end = '')
                print(f"running loss : {running_loss / i}" )

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


                # update plot and save
                clist = [l for l in range(1, len(loss1_list) + 1)]
                save_plot([np.log(loss1_list), np.log(loss3_list), np.log(running_loss_list)], clist, ["invloss", "mse", "runningloss"], os.path.join(self.output_dir, "all_losses.png"))
                save_plot([running_loss_list], clist, ["running loss"], os.path.join(self.output_dir, "runloss.png"))
                save_plot([loss1_list], clist, ["invariance loss"], os.path.join(self.output_dir, "invloss.png"))
                save_plot([loss3_list], clist, ["mse loss"], os.path.join(self.output_dir, "mseloss.png"))
                save_plot([loss2_list, avg_den_list], clist, ["density loss", "avg den"], os.path.join(self.output_dir, "loss_density.png"))
                save_plot([gradnorm_list], clist, ["grad norm"], os.path.join(self.output_dir, "gradnorm.png"))
                
                # update csv file and save
                train_dict = {
                    "invariance loss" : loss1_list,
                    "mse loss" : loss3_list,
                    "density loss" : loss2_list,
                    "grand norms" : gradnorm_list,
                    "running loss" : running_loss_list
                }
                df = pd.DataFrame(train_dict)
                fname = "data.csv"
                df.to_csv( os.path.join(self.output_dir, fname), sep=',', encoding='utf-8', index=False, header=True)


            et = time.time()
            print(f"total time for epoch : {str((et-st) / 60)} min")

            epoch_loss = running_loss / train_dataloader.__len__()
            epochloss_list.append(epoch_loss.item())
            print('Epoch [{}/{}], epoch Loss: {:.4f}'.format(epoch+1, epochs, epoch_loss))
            save_plot([epochloss_list], [i for i in range(epoch+1)], ["epoch loss"], os.path.join(self.output_dir, "epochloss.png"))

            if (epoch + 1) % val_every == 0:
                self.validate(model, test_dataloader, mask_density, alpha1, alpha2)
        
        et_tt = time.time()
        print(f"total time for training : {str((et_tt-st_tt) / 3600)} hr")
