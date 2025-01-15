import gc
import sys
import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms import Pad
from torchvision.transforms.functional import normalize
import torchvision
import numpy as np
import time
import torch.nn as nn

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.regression import MeanSquaredError

torch.set_printoptions(linewidth=3000)
torch.set_printoptions(precision=6)

class MSELoss(nn.Module):
    """
    Means squared loss : 
    """
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, U, V):
        nxny = U.shape[2] * U.shape[3]
        # U = normalize_(U)
        # V = normalize_(V)
        return torch.mean(torch.norm(U-V, p = 2, dim = (2,3))**2 / nxny)

class ResidualLoss(nn.Module):

    def __init__(self, img_size, offset):
        super(ResidualLoss, self).__init__()
        self.pad = Pad(1, padding_mode = "symmetric")
        self.nxny = img_size * img_size
        self.offset = offset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, f, mask):
        '''
        Rsidual Loss 
        (1 / nxny) || (1 - C)(\laplacian u - div ( d u)) - C (u - f) ||2 
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
        d1_u = mask * d1_u
        dx_d1_u = d1_u[:, :, 1:-1, 1:-1] - d1_u[:, :, 0:-2, 1:-1]

        # col-direction filters
        f3 = torch.tensor([[[[-1., 1.]]]], dtype = torch.float64, device = self.device)
        f4 = torch.tensor([[[[.5, .5]]]], dtype = torch.float64, device = self.device)
        d2_u = (F.conv2d(v, f3, padding='same') / F.conv2d(v, f4, padding='same')) * F.conv2d(u, f4, padding='same')
        d2_u = mask * d2_u
        dy_d2_u = d2_u[:, :, 1:-1, 1:-1] - d2_u[:, :, 1:-1, 0:-2]

        # steady state 
        ss = lap_u - dx_d1_u - dy_d2_u 

        # residual loss
        return torch.mean(torch.norm(ss, p = 2, dim = (2, 3)))  # / self.nxny
        # return torch.mean(torch.norm((1 - mask) * ss - mask * (x - f), p = 2, dim = (2, 3)) / self.nxny)
        
class DiffusionInpainting:

    def __init__(self, U, mask, tau, eps, hx = 1, hy = 1, device = None, apply_canny = False):
        # (b, c, h, w)

        self.U       = U   # guidance image
        self.batch   = U.size(0) 
        self.channel = U.size(1) 
        self.nx      = U.size(2) 
        self.ny      = U.size(3) 

        self.device = device

        if mask is None:
            self.mask = torch.ones_like(U)
        else :
            self.mask   = mask

        self.tau     = tau
        self.eps     = eps
                
        # pixel sizes x and y direction
        self.hx      = hx
        self.hy      = hy    

        # img 
        self.save_every  = 10
        self.apply_canny = apply_canny
        self.canny_mask  = None

        # others
        self.pad      = Pad(1, padding_mode = "symmetric") # mirror
        self.zero_pad = Pad(1,fill=0, padding_mode='constant') # zero padding

    def solveBatchParallel(self, df_stencils, bicg_mat, solver, kmax , save_batch = False, verbose = False):

        # init = self.U
        self.df_stencils = df_stencils
        self.bicg_mat = bicg_mat
        init = self.U.clone()
        X = self.U # torch.randn(self.U.shape).to(self.device)
        U = self.U
        tt = 0

        mse = MSELoss()

        # select solver
        if solver == "CG":
            solver = self.CG

        st = time.time()

        for i in range(kmax):

            # implicit 
            X, max_k = solver(x = U, b = X, kmax = 600, abs_eps = self.eps, verbose=verbose)
            U = X
            # loss = mse( init, init)
            # print(f"\rITERATION : {i+1}, loss : {loss.item()} ", end ='', flush=True)

            # explicit
            # X = self.explicitStep(U)
            # U = X

        # print()
        
        et = time.time()
        tt += (et-st)

        if save_batch[0]:
            fname = save_batch[1]
            
            # init = torch.transpose(init, 2, 3)
            # self.mask = torch.transpose(self.mask, 2, 3)
            # U = torch.transpose(U, 2, 3)
            
            out = torch.cat(( 
                            # (self.mask2 * 255.).reshape(self.batch*(self.nx+2), self.ny+2), 
                            # (self.canny_mask * 255.).reshape(self.batch*(self.nx+2), self.ny+2), 
                            (init * 255).reshape(self.batch*(self.nx+2), self.ny+2),
                            (self.mask * 255.).reshape(self.batch*(self.nx+2), self.ny+2), 
                            (U * 255.).reshape(self.batch*(self.nx+2), self.ny+2)
                            ),
                            dim = 1)
            # self.writePGMImage((self.normalize(U, 255).reshape(self.batch*(self.nx+2), self.ny+2) - self.offset).cpu().detach().numpy().T, fname)
            self.writePGMImage(out.cpu().detach().numpy().T, fname) 

        # mse loss 
        loss = mse( init, U)
        return loss, tt, max_k, self.df_stencils, U
        # return loss, tt, max_k, self.df_stencils, self.bicg_mat

    def writePGMImage(self, X, filename):
        # add comments for pgm img
        cv2.imwrite(filename, X[1:-1, 1:-1])
        # print(f"written to : {filename}")

    def prepareInp(self):
        """
        transposed ; Weickert transposed it in his C code
        """
        self.U     = self.pad(self.U)
        self.U     = torch.transpose(self.U, 2, 3)

        if self.mask != None :
            self.mask     = self.pad(self.mask)
            self.mask     = torch.transpose(self.mask, 2, 3)

        # since we transposed 
        self.nx, self.ny = self.ny, self.nx

        # apply mask
        self.setMask()

    def analyseImage(self, x, name):
        comm = ""
        # x = x[:, :, 1:self.nx+1, 1:self.ny+1]
        
        print(f"analyzing {name}")

        min_, max_, mean_, std_ = torch.min(x).item(), torch.max(x).item(), torch.mean(x).item(), torch.std(x).item()

        comm += f"min  : {min_}\n"
        comm += f"max  : {max_}\n"
        comm += f"mean : {mean_}\n"
        comm += f"std  : {std_}\n"

        print(comm)

        # comm += f"min  : {torch.amin(x, dim = (2, 3)) }\n"
        # comm += f"max  : {torch.amax(x, dim = (2, 3))}\n"
        # comm += f"mean : {torch.mean(x, dim = (2, 3))}\n"
        # comm += f"std  : {torch.std(x, dim = (2, 3))}\n"

        return comm, min_, max_, mean_, std_
        
    def createMaskfromCanny(self):
        output_batch = []
        images = (self.U * 255.).detach().cpu().numpy()

        for image in images:
            image = image.squeeze(0) # assuming grey scale image
            edges = cv2.Canny(image.astype(np.uint8), 100, 150) # make sure this outputs a certain density 
            print(f"mask created with densities : {np.count_nonzero(edges) / edges.size}")
            edges = np.expand_dims(edges, axis=0)
            output_batch.append(edges)

        output_batch = torch.tensor(np.stack(output_batch), device = self.device, dtype = torch.int8) * -1
        return output_batch
      
    def setMask(self , verbose = False):

        # get CannyMask
        if self.apply_canny : 
            self.canny_mask = self.createMaskfromCanny()
            self.mask = self.canny_mask

    def write_bicg_weights(self, x, name):
        comm, min_, max_, mean_, std_ = self.analyseImage(x, name)
        self.bicg_mat[name + "_max"].append(max_)
        self.bicg_mat[name + "_min"].append(min_)
        self.bicg_mat[name + "_mean"].append(mean_)

    def create_backward_hook(self, var_name):
        def hook(grad):
            comm, min_, max_, mean_, std_ = self.analyseImage(grad, var_name)
            self.df_stencils[var_name + "_max"].append(max_)
            self.df_stencils[var_name + "_min"].append(min_)
            self.df_stencils[var_name + "_mean"].append(mean_)
            # print(f"Gradient of {var_name}\n grad norm : {grad.norm()}\n grad stats:\n{comm}")
        return hook

    def create_backward_hook2(self, var_name):
        def hook(grad):
            comm, min_, max_, mean_, std_ = self.analyseImage(grad, var_name)
            self.bicg_mat[var_name + "_max"].append(max_)
            self.bicg_mat[var_name + "_min"].append(min_)
            self.bicg_mat[var_name + "_mean"].append(mean_)
            # print(f"Gradient of {var_name}\n grad norm : {grad.norm()}\n grad stats:\n{comm}")
        return hook

    def zeroPad(self, x):
        return self.zero_pad(x[ :, :, 1:self.nx+1, 1 :self.ny+1])

    def applyStencil(self, inp, verbose = False):
        """
        inp : (batch, channel, nx, ny)
        """
        # time savers
        rxx = self.tau / (self.hx * self.hx)
        ryy = self.tau / (self.hy * self.hy)

        # pad input
        inp     = self.pad(inp[:, :, 1:self.nx+1, 1:self.ny+1])

        # from top to bottom -> center, left, down, up, right
        res  = (1 + 2 * rxx + 2 * ryy) * inp[:, :, 1:self.nx+1, 1:self.ny+1] \
                                 - rxx * inp[:, :,  :self.nx,   1:self.ny+1] \
                                 - ryy * inp[:, :, 1:self.nx+1,  :self.ny  ] \
                                 - ryy * inp[:, :, 1:self.nx+1, 2:self.ny+2] \
                                 - rxx * inp[:, :, 2:self.nx+2, 1:self.ny+1]

        res = self.zero_pad(res)
        res = torch.where(self.mask == 1, inp, res)

        if verbose :
            self.analyseImage(res, "X")

        return res

    def CG(self, x, b, kmax, abs_eps, verbose = False):

        k       = torch.zeros((self.batch, self.channel), dtype=torch.long, device = self.device) 
        alpha   = torch.zeros((self.batch, self.channel), dtype=torch.float64, device = self.device) 
        beta    = torch.zeros((self.batch, self.channel), dtype=torch.float64, device = self.device) 

        r       = torch.zeros_like(x, dtype = torch.float64) 
        p       = torch.zeros_like(x, dtype = torch.float64) 
        q       = torch.zeros_like(x, dtype = torch.float64) 

        relative_eps = abs_eps**2
        
        b = torch.mul(b, self.mask)
        p = r = self.zeroPad(b - self.applyStencil(x))
        rho_0 = rho = rho_old = torch.sum( torch.mul(r, r), dim = (2, 3))

        # itearation counter, absolute residual and relative residual
        while ( (k < kmax) & (rho > abs_eps * self.nx * self.ny) & (rho > relative_eps * rho_0)).any():
            
            # =======================================
            # WHILE CONVERGENCE CONDITION
            # =======================================
            CONV_COND = (k < kmax) & (rho > abs_eps * self.nx * self.ny) & (rho > relative_eps * rho_0)
            CONV_COND_EXP = CONV_COND[:, :, None, None]
            
            # update solution
            q       = torch.where(CONV_COND_EXP, self.applyStencil(p), q)
            alpha   = torch.where(CONV_COND, rho / torch.sum( torch.mul(p, q), dim = (2, 3)), alpha)
            x       = torch.where(CONV_COND_EXP, x + alpha[:, :, None, None] * p, x)

            # update residual and its norm
            rho_old = torch.where(CONV_COND, rho, rho_old)
            r       = torch.where(CONV_COND_EXP, r - alpha[:, :, None, None] * q, r)
            rho     = torch.where(CONV_COND, torch.sum( torch.mul(r, r), dim = (2, 3)), rho)

            # update search direction
            beta    = torch.where(CONV_COND, rho / rho_old, beta)
            p       = torch.where(CONV_COND_EXP, r + beta[:, :, None, None] * p, p)
            k       = torch.where(CONV_COND, k + 1, k) 
            # print(f"k : {k}, RESIDUAL : {rho}")

        return x, torch.max(k)

    def explicitStep(self, inp):

        # time savers
        rxx = self.tau / (self.hx * self.hx)
        ryy = self.tau / (self.hy * self.hy)

        # pad input
        inp     = self.pad(inp[:, :, 1:self.nx+1, 1:self.ny+1])

        # explicit diffusion inpainting
        # from top to bottom -> center, left, down, up, right
        res  = (1 - 2 * rxx - 2 * ryy) * inp[:, :, 1:self.nx+1, 1:self.ny+1] \
                                 + rxx * inp[:, :,  :self.nx,   1:self.ny+1] \
                                 + ryy * inp[:, :, 1:self.nx+1,  :self.ny  ] \
                                 + ryy * inp[:, :, 1:self.nx+1, 2:self.ny+2] \
                                 + rxx * inp[:, :, 2:self.nx+2, 1:self.ny+1]
        
        res = self.zero_pad(res)
        res = torch.where(self.mask == 1, inp, res)
        
        return res
