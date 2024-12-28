#train.py
import sys
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam, SGD, RMSprop, Adagrad
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, LambdaLR
import torch
import torch.nn as nn
import time
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
from InpaintingSolver.Solvers import OsmosisInpainting
# from InpaintingSolver.jacobi import OsmosisInpainting
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.transforms import Pad
import cv2
import numpy as np
import random

from utils import get_dfStencil, get_bicgDict, getOptimizer, getScheduler, loadCheckpoint, saveCheckpoint
from utils import initialize_weights_he, init_weights_xavier, save_plot, check_gradients
from utils import inspect_gradients, MyCustomTransform2, mean_density, normalize, OffsetEvolve

torch.backends.cuda.matmul.allow_tf32 = True
SEED = 5 # 4
torch.manual_seed(SEED)

# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

# g = torch.Generator()
# g.manual_seed(SEED)

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 10000)
# pd.set_option('display.max_colwidth', None)  


class ResidualLoss(nn.Module):
    """
    Rsidual Loss 
    (1 / nxny) || (1 - C)(\laplacian u - div ( d u)) - C (u - f) ||2 
    """
    def __init__(self, img_size, offset):
        super(ResidualLoss, self).__init__()
        self.pad = Pad(1, padding_mode = "symmetric")
        self.nxny = img_size * img_size
        self.offset = offset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, f, mask):
        '''
        x : evolved solution ; not padded
        f : guidance image   ; not padded
        '''
        u   = self.pad(x + self.offset)
        v   = self.pad(f + self.offset)

        # laplacian kernel
        lap_u_kernel = torch.tensor([[[[0., 1., 0.],
                                       [1.,-4., 1.],
                                       [0., 1., 0.]]]], dtype = torch.float64, device = self.device)
        lap_u = F.conv2d(u, lap_u_kernel)

        # row-direction filters  
        f1 = torch.tensor([[[[-1.], [1.]]]], dtype = torch.float64, device = self.device)
        f2 = torch.tensor([[[[.5], [.5]]]], dtype = torch.float64, device = self.device)
        d1_u = (F.conv2d(v, f1, padding='same') / F.conv2d(v, f2, padding='same')) * F.conv2d(u, f2, padding='same')
        dx_d1_u = d1_u[:, :, 1:-1, 1:-1] - d1_u[:, :, 0:-2, 1:-1]

        # col-direction filters
        f3 = torch.tensor([[[[-1., 1.]]]], dtype = torch.float64, device = self.device)
        f4 = torch.tensor([[[[.5, .5]]]], dtype = torch.float64, device = self.device)
        d2_u = (F.conv2d(v, f3, padding='same') / F.conv2d(v, f4, padding='same')) * F.conv2d(u, f4, padding='same')
        dy_d2_u = d2_u[:, :, 1:-1, 1:-1] - d2_u[:, :, 1:-1, 0:-2]

        #steady state 
        ss = lap_u - dx_d1_u - dy_d2_u 

        # residual loss
        return torch.mean(torch.norm((1 - mask) * ss - mask * (x - f), p = 2, dim = (2, 3)) / self.nxny)
        
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
    
def getDataloaders(train_dataset, test_dataset, img_size, train_batch_size, test_batch_size):

    transform = transforms.Compose([
                transforms.Resize((img_size, img_size), antialias = True),
                transforms.Grayscale(),
                transforms.ToTensor(),   
            ])

    transform_random = transforms.Compose([
                transforms.RandomCrop((img_size, img_size)),
                transforms.Grayscale(),
                transforms.ToTensor(),   
            ])
    
    transform_scale = transforms.Compose([
                # transforms.RandomCrop((img_size, img_size)),
                transforms.Resize((img_size, img_size), antialias = True),
                transforms.Grayscale(),
                MyCustomTransform2()
            ])

    def custom_collate_fn(batch):
        
        images       = []
        images_scale = []

        for item in batch :
            images.append(transform(item['image']))
            images_scale.append(transform_random(item['image']))

        images       = torch.stack(images)
        images_scale = torch.stack(images_scale)
        
        return images, images_scale

    train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size=train_batch_size, collate_fn=custom_collate_fn)
    test_dataloader  = DataLoader(test_dataset, shuffle = True, batch_size=test_batch_size, collate_fn=custom_collate_fn)

    # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, worker_init_fn=seed_worker, generator=g)
    # test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=test_batch_size, worker_init_fn=seed_worker, generator=g)

    print(f"train and test dataloaders created")
    print(f"total train batches  : {len(train_dataloader)}")
    print(f"total test  batches  : {len(test_dataloader)}")

    return train_dataloader, test_dataloader


class JointModelTrainer():

    def __init__(self, output_dir, opt1, opt2, scheduler1, scheduler2, train_batch_size, test_batch_size):
        
        self.output_dir= output_dir
        self.opt1 = opt1
        self.opt2 = opt2
        self.scheduler1= scheduler1,
        self.scheduler2= scheduler2,
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device : {self.device}")

    def train(self, maskModel,
                    inpModel,
                    epochs,
                    alpha1, 
                    alpha2,
                    offset,
                    tau, 
                    mask_density, 
                    img_size, 
                    model_1_ckp_file, 
                    model_2_ckp_file, 
                    save_every, 
                    batch_plot_every, 
                    val_every, 
                    skip_norm, 
                    max_norm, 
                    train_dataset,
                    test_dataset):
        
        # loss lists
        loss1_list, loss2_list, loss3_list, gradnorm_list, tot_mask_loss_list = [], [], [], [], []
        avg_den_list = []
        epochloss_MN_list, epochloss_IN_list = [],[]
        val_list = []

        train_dataloader, test_dataloader = getDataloaders(train_dataset, test_dataset, img_size, self.train_batch_size, self.test_batch_size)

        opt1 = getOptimizer(maskModel, self.opt1)
        opt2 = getOptimizer(inpModel , self.opt2)
        # scheduler = self.getScheduler(optimizer)
        # scheduler = WarmupScheduler(optimizer, warmup_steps=5, final_lr=self.lr, base_lr=1e-5)

        print(f"optimizer 1: {opt1}, optimizer 2 : {opt2} loaded")

        if model_1_ckp_file != None:
            print(f"loading checkpoint file : {model_1_ckp_file}")
            maskModel, opt1 = loadCheckpoint(maskModel, opt1, model_1_ckp_file)
            print(f"Mask model, opt, schdl loaded from checkpoint")

        if model_2_ckp_file != None:
            print(f"loading checkpoint file : {model_2_ckp_file}")
            inpModel, opt2 = loadCheckpoint(inpModel, opt2, model_2_ckp_file)
            print(f"Inpainting model, opt, schdl loaded from checkpoint")

        maskModel = maskModel.double()
        maskModel.to(self.device)

        inpModel = inpModel.double()
        inpModel.to(self.device)
 
        print(f"initializing weights using Kaiming/He Initialization")
        maskModel.apply(initialize_weights_he)
        inpModel.apply(initialize_weights_he)

        # losses 
        mseLoss  = MSELoss()
        maskLoss = InvarianceLoss()
        resLoss  = ResidualLoss(img_size=img_size, offset=offset)

        torch.autograd.set_detect_anomaly(True)

        print("\nbeginning training ...")

        st_tt = time.time()

        for epoch in range(epochs):

            running_tmloss = 0.0
            running_mloss = 0.0
            running_iloss = 0.0
            running_rloss = 0.0

            skipped_batches = 0

            maskModel.train()
            inpModel.train()

            st = time.time()

            for i, (X, X_crop) in enumerate(train_dataloader, start = 1): 
                                
                print(f'Epoch {epoch}/{epochs} , batch {i}/{len(train_dataloader)} ')
                
                X = X_crop.to(self.device, dtype=torch.float64) 

                # mask 
                mask  = maskModel(X) # non-binary [0,1]
                loss1 = maskLoss(mask) 

                # inpainting 
                mask = mask.detach()
                stack_X_mask = torch.cat((X, mask), dim=1)
                rec_X = inpModel(stack_X_mask)
                loss2 = mseLoss(rec_X, X)
                loss3 = resLoss(rec_X, X, mask)
                
                maskModelLoss = loss2 + alpha1 * loss1 
                
                maskModelLoss.backward(retain_graph=True)
                opt1.step()
                
                loss3.backward()
                opt2.step()
                
                opt1.zero_grad()
                opt2.zero_grad()

                # solve using solver
                osmosis = OsmosisInpainting(None, X, mask, mask, offset = offset, tau=tau, eps = 1e-9, device = self.device, apply_canny=False)
                osmosis.calculateWeights(d_verbose=False, m_verbose=False, s_verbose=False)                
                loss3, tts, max_k, df_stencils, U = osmosis.solveBatchParallel(None, None, kmax = 1, save_batch = [False], verbose = False)
                U = torch.transpose(U[:, :, 1:-1, 1:-1], 2, 3)

                if (i-1) % save_every == 0:
                    print("saving checkpoint")
                    fname = f"maskmodel_epoch_{str(epoch+1)}_iter_{str(i)}.pt"
                    saveCheckpoint(maskModel, opt1, self.output_dir, fname)
                    fname = f"inpmodel_epoch_{str(epoch+1)}_iter_{str(i)}.pt"
                    saveCheckpoint(inpModel, opt2, self.output_dir, fname)
                    
                if (i) % batch_plot_every == 0:
                    print("saving batch")
                    fname = f"batch_epoch_{str(epoch)}_iter_{str(i)}.png"
                    fname_path = os.path.join(self.output_dir, "imgs", fname)
                    out_save = torch.cat((
                                        (X * 255).reshape(self.train_batch_size*img_size, img_size),
                                        (mask * 255).reshape(self.train_batch_size*img_size, img_size),
                                        (U * 255).reshape(self.train_batch_size*img_size, img_size))
                                        , dim = 1).cpu().detach().numpy()
                    cv2.imwrite(fname_path, out_save)

                running_tmloss += maskModelLoss.item()
                running_mloss  += loss1.item()
                running_iloss  += loss2.item()
                running_rloss  += loss3.item()

                avg_den = mean_density(mask)

                avg_den_list.append(avg_den.item())
                loss1_list.append(running_mloss / (i - skipped_batches))
                loss2_list.append(running_iloss / (i - skipped_batches))
                loss3_list.append(running_rloss / (i - skipped_batches))
                tot_mask_loss_list.append((running_tmloss / (i - skipped_batches)))
                # gradnorm_list.append(total_norm)

                print(f"mask loss : {loss1}, avg_den : {avg_den.item()}, ", end='')
                print(f"mse loss : {loss2}, ", end='')
                print(f"residual loss : {loss3}, ", end='')
                print(f"total mask loss : {maskModelLoss}, " , end = '')

                # update plot and save
                clist = [l for l in range(1, len(loss1_list) + 1)]
                save_plot([tot_mask_loss_list], clist, ["total mask loss"], os.path.join(self.output_dir, "tot_mask_loss.png"))
                save_plot([loss1_list], clist, ["mask loss"], os.path.join(self.output_dir, "mask_loss.png"))
                save_plot([loss3_list], clist, ["residual loss"], os.path.join(self.output_dir, "residual_loss.png"))
                save_plot([loss2_list], clist, ["mse loss"], os.path.join(self.output_dir, "inpainting_loss.png"))
                # save_plot([gradnorm_list], clist, ["grad norm"], os.path.join(self.output_dir, "gradnorm.png"))
                
                # update csv file and save
                train_dict = {
                    # "grand norms" : gradnorm_list,
                    "running mse loss" : loss2_list,
                    "running mask loss" : loss1_list,
                    "running total(mse+mask) mask loss" : tot_mask_loss_list,
                    "running residual loss" : loss3_list,
                    }
                print()
                df = pd.DataFrame(train_dict)
                print(df.tail(20).to_string())
                fname = "data.csv"
                df.to_csv( os.path.join(self.output_dir, fname), sep=',', encoding='utf-8', index=False, header=True)

            et = time.time()

            print(f"total time for epoch : {str((et-st) / 60)} min")
            epoch_lossMN = running_tmloss / (train_dataloader.__len__() - skipped_batches)
            epochloss_MN_list.append(epoch_lossMN)
            epoch_lossIN = running_rloss / (train_dataloader.__len__() - skipped_batches)
            epochloss_IN_list.append(epoch_lossIN)
            
            print('Epoch [{}/{}], epoch Loss Mask Network: {:.4f}, epoch Loss Inpainting Network: {:.4f}'.format(epoch+1, epochs, epoch_lossMN, epoch_lossIN))
            save_plot([epochloss_MN_list], [i for i in range(epoch+1)], ["epoch MN loss"], os.path.join(self.output_dir, "epochloss_masknetwork.png"))
            save_plot([epochloss_IN_list], [i for i in range(epoch+1)], ["epoch IN loss"], os.path.join(self.output_dir, "epochloss_inpaintingnetwork.png"))

            print("\ncleaning torch mem and cache . End of epoch")
            torch.cuda.empty_cache()

        et_tt = time.time()
        print(f"total time for training : {str((et_tt-st_tt) / 3600)} hr")


class ModelTrainer():

    def __init__(self, output_dir, opt_config, scheduler, train_batch_size, test_batch_size):
        
        self.output_dir= output_dir
        self.opt_config= opt_config
        self.scheduler= scheduler
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device : {self.device}")

    def validate(self, model, test_dataloader, density, alpha1, alpha2, offset, tau, eps):
        print("validating on test dataset")
        avg_loss = 0.0

        invLoss = InvarianceLoss()
        denLoss  = DensityLoss(density)

        td_len = test_dataloader.__len__()
        df_stencils = get_dfStencil()
        bicg_mat = get_bicgDict()
        
        model.eval()

        with torch.no_grad():
            st = time.time()
            for i, (X, X_crop) in enumerate(test_dataloader):

                X = X_crop.to(self.device, dtype=torch.float64) + offset
                mask  = model(X) # non-binary [0,1]
                loss1 = invLoss(mask)
                loss2 = denLoss(mask)

                # for 2 mask prediction
                if mask.shape[1] == 2:
                    mask1 = mask[:, 0, :, :].unsqueeze(1)
                    mask2 = mask[:, 1, :, :].unsqueeze(1)
                else:
                    mask1 = mask
                    mask2 = mask

                osmosis = OsmosisInpainting(None, X, mask1, mask2, offset=0, tau=tau, eps = 1e-9, device = self.device, apply_canny=False)
                osmosis.calculateWeights(d_verbose=False, m_verbose=False, s_verbose=False)
                save_batch = [False]
                loss3, tts, max_k, df_stencils, bicg_mat = osmosis.solveBatchParallel(df_stencils, bicg_mat, 1, save_batch = save_batch, verbose = False)
                
                total_loss = loss3 + loss1 * alpha1 + loss2

                avg_loss += total_loss.item()

            et = time.time()
        
        val_loss = avg_loss / td_len
        print(f"\nvalidation loss : {val_loss} , total running time : {(et-st)/60.} min")
        print()
        model.train()
        return val_loss

    def train(self, model,
                    epochs,
                    alpha1, 
                    alpha2, 
                    mask_density, 
                    img_size, 
                    resume_checkpoint_file, 
                    save_every, 
                    batch_plot_every, 
                    val_every, 
                    skip_norm, 
                    max_norm, 
                    train_dataset,
                    test_dataset,
                    solver,
                    offset, 
                    offset_evl_steps,
                    tau, 
                    eps):
        
        # loss lists
        loss1_list, loss2_list, loss3_list, gradnorm_list, running_loss_list = [], [], [], [], []
        avg_den_list, epochloss_list, offset_list, lr_list = [], [], [], []
        val_list = []
        ttl_skipped_batches = 0
        iter = 0
        
        train_dataloader, test_dataloader = getDataloaders(train_dataset, test_dataset, img_size,  self.train_batch_size, self.test_batch_size)

        optimizer = getOptimizer(model, self.opt_config)
        scheduler = getScheduler(optimizer, self.scheduler)
        offsetEvol = OffsetEvolve(init_offset=0.004, final_offset=offset, max_iter = offset_evl_steps)
        # scheduler = WarmupScheduler(optimizer, warmup_steps=5, final_lr=self.lr, base_lr=1e-5)

        print(f"optimizer , scheduler  loaded")

        if resume_checkpoint_file != None:
            print(f"loading checkpoint file : {resume_checkpoint_file}")
            model, optimizer = loadCheckpoint(model, optimizer, resume_checkpoint_file)
            print(f"model, opt, schdl loaded from checkpoint")

        print(f"optimizer : {optimizer}")
        print(f"scheduler : {scheduler}")

        model = model.double()
        model.to(self.device)
        
        # bicg_model = BiCG_Net(offset = 0, tau = 7000., b=self.train_batch_size, c=1, nx=img_size, ny=img_size)
        # bicg_model = bicg_model.double()
        # bicg_model.to(self.device)

        print(f"initializing weights using Kaiming/He Initialization")
        model.apply(initialize_weights_he)

        # Attach hooks to layers
        # for layer in model.children():
        #     layer.register_full_backward_hook(inspect_gradients)

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

            for i, (X, X_crop) in enumerate(train_dataloader, start = 1): 
                
                bicg_mat = get_bicgDict()
                
                print(f'Epoch {epoch}/{epochs} , batch {i}/{len(train_dataloader)} ')
                # df_stencils["f_name"].append(name)

                # offset annealing
                offset = offsetEvol(iter)
                iter += 1
                X = X_crop.to(self.device, dtype=torch.float64) + offset

                # mask model
                mask  = model(X) # non-binary [0,1]

                # mask losses
                loss1 = invLoss(mask)
                loss2 = denLoss(mask)

                # for 2 mask prediction
                if mask.shape[1] == 2:
                    mask1 = mask[:, 0, :, :].unsqueeze(1)
                    mask2 = mask[:, 1, :, :].unsqueeze(1)
                else:
                    mask1 = mask
                    mask2 = mask

                # osmosis solver
                osmosis = OsmosisInpainting(None, X, mask1, mask2, offset=offset, tau=tau, eps = eps, device = self.device, apply_canny=False)
                osmosis.calculateWeights(d_verbose=False, m_verbose=False, s_verbose=False)
                
                if (i) % batch_plot_every == 0: 
                    save_batch = [True, os.path.join(self.output_dir, "imgs", f"batch_epoch_{str(epoch)}_iter_{str(i)}.png")]
                else:
                    save_batch = [False]
                df_stencils["iter"].append(i)
                loss3, tts, max_k, df_stencils, bicg_mat = osmosis.solveBatchParallel(df_stencils, bicg_mat, solver, kmax = 1, save_batch = save_batch, verbose = False)
                
                total_loss = loss3 + loss1 * alpha1 + loss2
                bp_st = time.time()
                total_loss.backward()
                bp_et = time.time()
                total_norm = check_gradients(model)

                # write forward backward stencils
                # df_stencils["grad_norm"].append(total_norm)
                # df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in df_stencils.items()]))
                # df.to_csv( os.path.join(self.output_dir, "stencils.csv"), sep=',', encoding='utf-8', index=False, header=True)
                # bicg_mat["grad_norm"].append(total_norm)
                # df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in bicg_mat.items()]))
                # df.to_csv( os.path.join(self.output_dir, f"bicg_wt_{i}.csv"), sep=',', encoding='utf-8', index=False, header=True)
                
                if (i-1) % save_every == 0:
                    print("saving checkpoint")
                    fname = f"ckp_epoch_{str(epoch+1)}_iter_{str(i-1)}.pt"
                    saveCheckpoint(model, optimizer, self.output_dir, fname)

                # if total_norm < skip_norm :
                #     m_max_norm = max_norm
                # elif total_norm > skip_norm and total_norm < 2 * skip_norm :
                #     m_max_norm = max_norm * 2
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = m_max_norm)
                # # elif total_norm > 2 * skip_norm and total_norm < 10 * skip_norm :
                # #     m_max_norm = max_norm * 3
                # # elif total_norm > 10 * skip_norm and total_norm < 20 * skip_norm :
                # #     m_max_norm = max_norm * 4
                # else :
                #     skipped_batches += 1
                #     ttl_skipped_batches += 1
                #     print(f"skipping batch due to higher gradient norm : {total_norm}, total skipped : {ttl_skipped_batches}")
                #     optimizer.zero_grad()
                #     continue
                
                # try : 
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = max_norm)
                # except Exception as e :
                #     skipped_batches += 1
                #     ttl_skipped_batches += 1
                #     print(f"exception caused at gradient clipping for total_norm : {total_norm} and max norm: {m_max_norm}")
                #     print("skipping batch")
                #     print(e)
                #     continue

                offset_list.append(offset)
                lr_list.append(optimizer.param_groups[0]['lr'])
                optimizer.step()
                if scheduler != None :
                    scheduler.step()
                optimizer.zero_grad()


                running_loss += total_loss.item()
                running_mse  += loss3.item()
                running_inv  += loss1.item()
                avg_den = mean_density(mask)

                avg_den_list.append(avg_den.item())
                loss1_list.append(running_inv / (i - skipped_batches))
                loss2_list.append(loss2.item())
                loss3_list.append(running_mse / (i - skipped_batches))
                running_loss_list.append((running_loss / (i - skipped_batches)))
                gradnorm_list.append(total_norm)
                print(f"invariance loss : {loss1}, avg_den : {avg_den.item()}, ", end='')
                print(f"density loss : {loss2}, solver time : {str(tts)} sec , ", end='')
                print(f"max iteration in solver : {max_k}, ", end ='')
                print(f'offset : {offset}, ', end = '')
                print(f"backprop time : {str(bp_et - bp_st)} sec, ", end ='')
                print(f"mse loss : {loss3}, ", end='')
                print(f"total loss : {total_loss}, " , end = '')
                print(f"running loss : {running_loss / (i - skipped_batches)}" )

                # if (i) % save_every == 0:
                #     # update mask distribution plot and save
                #     print("plotting mask distribution")
                #     fname = f"mdist_epoch_{str(epoch+1)}_iter_{str(i)}.png"
                #     mask_flat = mask.reshape(-1).detach().cpu().numpy()
                #     plot = sns.displot(mask_flat, kde=True)
                #     fig = plot.figure
                #     plot.set(xlabel='prob', ylabel='freq')
                #     fig.savefig(os.path.join(self.output_dir, fname) ) 
                #     plt.close(fig)

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
                    "running inv loss" : loss1_list,
                    "running mse loss" : loss3_list,
                    "running loss" : running_loss_list,
                    "offset" : offset_list,
                    "learning rate" : lr_list
                    }
                df = pd.DataFrame(train_dict)
                print(df.tail(20))
                fname = "data.csv"
                df.to_csv( os.path.join(self.output_dir, fname), sep=',', encoding='utf-8', index=False, header=True)

                # validate
                if (i) % val_every == 0:
                    val_loss = self.validate(model, test_dataloader, mask_density, alpha1, alpha2, offset, tau, eps)
                    val_list.append(val_loss)

                # update val csv file and save
                val_dict = {"val loss" : val_list}
                df = pd.DataFrame(val_dict)
                fname = "val.csv"
                df.to_csv( os.path.join(self.output_dir, fname), sep=',', encoding='utf-8', index=False, header=True)

            et = time.time()
            print(f"total time for epoch : {str((et-st) / 60)} min")
            print(f"lr rate for the epoch : {optimizer.param_groups[0]['lr']}")
            epoch_loss = running_loss / (train_dataloader.__len__() - skipped_batches)
            epochloss_list.append(epoch_loss)
            print('Epoch [{}/{}], epoch Loss: {:.4f}'.format(epoch+1, epochs, epoch_loss))
            save_plot([epochloss_list], [i for i in range(epoch+1)], ["epoch loss"], os.path.join(self.output_dir, "epochloss.png"))

            print("\ncleaning torch mem and cache . End of epoch")
            torch.cuda.empty_cache()

        et_tt = time.time()
        print(f"total time for training : {str((et_tt-st_tt) / 3600)} hr")
