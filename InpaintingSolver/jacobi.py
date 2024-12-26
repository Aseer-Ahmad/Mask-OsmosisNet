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


def normalize_(X, scale = 1.):
    b, c, _ , _ = X.shape
    X = X - torch.amin(X, dim=(2,3)).view(b,c,1,1)
    X = X / (torch.amax(X, dim=(2,3)).view(b,c,1,1) + 1e-7)
    X = X * scale

    return X


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

class OsmosisInpainting:

    def __init__(self, U, V, mask1, mask2, offset, tau, eps, hx = 1, hy = 1, device = None, apply_canny = False):
        # (b, c, h, w)

        self.V       = V   # guidance image
        self.batch   = V.size(0) 
        self.channel = V.size(1) 
        self.nx      = V.size(2) 
        self.ny      = V.size(3) 

        self.device = device

        if U is not None:
            self.U   = U   # original image
        else:
            self.U   = self.getInit_U()

        if mask1 is None or mask2 is None:
            self.mask1 = torch.ones_like(V)
            self.mask2 = torch.ones_like(V)
        else :
            self.mask1   = mask1
            self.mask2   = mask2

        self.offset  = offset
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

    def solveBatchParallel(self, df_stencils, bicg_mat, kmax , save_batch = False, verbose = False):

        # init = self.U
        self.df_stencils = df_stencils
        self.bicg_mat = bicg_mat
        X = self.U
        U = self.U
        tt = 0

        mse = MSELoss()

        # self.analyseImage(X, f"inital input")
        # self.analyseImage(self.mask1, f"input mask")

        st = time.time()

        for i in range(kmax):

            X, max_k = self.Jacobi(x = U, b = X, kmax = 600, eps = self.eps, verbose=verbose)
            U = X
            loss = mse( U, self.V)
            print(f"\rITERATION : {i+1}, loss : {loss.item()} ", end ='', flush=True)

        # print()
        
        et = time.time()
        tt += (et-st)

        # self.analyseImage(self.V, f"guidance")
        # self.analyseImage(U, f"solution")
        # comm = f"time for iteration : {str(et-st)} sec\n"
        # comm += f"total time         : {str(tt)} sec\n"
        # comm += self.getMetrics()
        # print(comm)

        if save_batch[0]:
            fname = save_batch[1]

            out = torch.cat(( 
                            ((self.V - self.offset) * 255.).reshape(self.batch*(self.nx+2), self.ny+2) , 
                            (self.mask1 * 255.).reshape(self.batch*(self.nx+2), self.ny+2), 
                            # (self.mask2 * 255.).reshape(self.batch*(self.nx+2), self.ny+2), 
                            # (self.canny_mask * 255.).reshape(self.batch*(self.nx+2), self.ny+2), 
                            # (init-self.offset).reshape(self.batch*(self.nx+2), self.ny+2),
                            ((U - self.offset) * 255.).reshape(self.batch*(self.nx+2), self.ny+2)
                            ),
                            dim = 1)
            # self.writePGMImage((self.normalize(U, 255).reshape(self.batch*(self.nx+2), self.ny+2) - self.offset).cpu().detach().numpy().T, fname)
            self.writePGMImage(out.cpu().detach().numpy().T, fname) 

        # print(torch.mean((self.normalize(U, 255) - self.normalize(self.V, 255)) ** 2, dim=(2, 3)))

        # mse loss 
        loss = mse(U , self.V)
        return loss, tt, max_k, self.df_stencils, U
        # return loss, tt, max_k, self.df_stencils, self.bicg_mat

    def calculateWeights(self, d_verbose = False, m_verbose = False, s_verbose = False):
        self.prepareInp()

        self.getDriftVectors(d_verbose)
        if d_verbose:
            print(f"drift vectors calculated")

        self.applyMask(m_verbose)
        if m_verbose:
            print(f"mask applied to drift vectors")

        self.getStencilMatrices(s_verbose)
        if s_verbose:
            print(f"stencils weights calculated")
        
    def normalize(self, X, scale = 1.):
        b, c, _ , _ = X.shape
        X = X - torch.amin(X, dim=(2,3)).view(b,c,1,1)
        X = X / (torch.amax(X, dim=(2,3)).view(b,c,1,1) + 1e-7)
        X = X * scale

        return X
        
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

        self.V     = self.pad(self.V)
        self.V     = torch.transpose(self.V, 2, 3)

        if self.mask1 != None :
            self.mask1     = self.pad(self.mask1)
            self.mask1     = torch.transpose(self.mask1, 2, 3)

        if self.mask2 != None:
            self.mask2     = self.pad(self.mask2)
            self.mask2     = torch.transpose(self.mask2, 2, 3)

        # since we transposed 
        self.nx, self.ny = self.ny, self.nx

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
        
    def getMetrics(self):
        metrics = ""
        psnr  = PeakSignalNoiseRatio().to(self.device)
        mse   = torch.nn.MSELoss()

        metrics += f"psnr : {str(( psnr(self.normalize(self.U), self.normalize(self.V)) ))}\n"
        metrics += f"mse  : {str(( mse(self.normalize(self.U), self.normalize(self.V)) ))}\n"
        
        return metrics

    def getInit_U(self):
        m  = torch.mean(self.V, dim = (2,3))
        
        # create a flat image ; avg gray val same as guidance
        u  = torch.ones_like(self.V, device = self.device) * m.view(self.batch, self.channel, 1, 1)

        # create a noisy image ; avg gray val same as guidance
        # noisy_tensor = torch.randn(self.V.shape, device=self.device) * 1.0 
        # current_mean = torch.mean(noisy_tensor, dim = (2,3))
        # u = noisy_tensor - current_mean[:, :, None, None] + m[:, :, None, None]

        return u

    def createMaskfromCanny(self):
        output_batch = []
        images = self.normalize(self.V, 255.).detach().cpu().numpy()

        for image in images:
            image = image.squeeze(0) # assuming grey scale image
            edges = cv2.Canny(image.astype(np.uint8), 100, 150) # make sure this outputs a certain density 
            print(f"mask created with densities : {np.count_nonzero(edges) / edges.size}")
            edges = np.expand_dims(edges, axis=0)
            output_batch.append(edges)

        output_batch = torch.tensor(np.stack(output_batch), device = self.device, dtype = torch.int8) * -1
        return output_batch
      
    def applyMask(self , verbose = False):

        # get CannyMask
        if self.apply_canny : 
            self.canny_mask = self.createMaskfromCanny()
            self.mask1 = self.canny_mask
            self.mask2 = self.canny_mask
            
        # self.hardRoundBinarize() 
        self.d1 = torch.mul(self.d1, self.mask1)
        self.d2 = torch.mul(self.d2, self.mask2)

        if verbose:
            self.analyseImage(self.mask1, "mask1")
            self.analyseImage(self.mask2, "mask2")
            self.analyseImage(self.d1, "d1")
            self.analyseImage(self.d2, "d2")

    def getDriftVectors(self, verbose = False):
        """
        # ∗ is convolution and .T is transpose
        # compute d1 = [-1/hx 1/hx] ∗ V / [.5 .5] ∗ v 
        # compute d2 = [-1/hy 1/hy].T ∗ V / [.5 .5].T ∗ v 
        """
        self.d1  = torch.zeros_like(self.V, dtype = torch.float64, device = self.device) + 1e-7
        self.d2  = torch.zeros_like(self.V, dtype = torch.float64, device = self.device) + 1e-7
                
        # row-direction filters  
        f1 = torch.tensor([[[[-1/self.hx], [1/self.hx]]]], dtype = torch.float64, device = self.device)
        f2 = torch.tensor([[[[.5], [.5]]]], dtype = torch.float64, device = self.device)
        
        # col-direction filters
        f3 = torch.tensor([[[[-1/self.hy, 1/self.hy]]]], dtype = torch.float64, device = self.device)
        f4 = torch.tensor([[[[.5, .5]]]], dtype = torch.float64, device = self.device)

        self.d1 = F.conv2d(self.V, f1, padding='same') / F.conv2d(self.V, f2, padding='same')  
        self.d2 = F.conv2d(self.V, f3, padding='same') / F.conv2d(self.V, f4, padding='same') 

        if verbose:
            self.analyseImage(self.d1, "d1")
            self.analyseImage(self.d2, "d2")
                
    def getStencilMatrices(self, verbose = False):

        self.boo  = torch.zeros_like(self.V, dtype = torch.float64, device = self.device)# C++ weickert init. has ones
        self.bop  = torch.zeros_like(self.V, dtype = torch.float64, device = self.device)# neighbour entries for [i+1,j]
        self.bpo  = torch.zeros_like(self.V, dtype = torch.float64, device = self.device)# neighbour entries for [i,j+1]
        self.bmo  = torch.zeros_like(self.V, dtype = torch.float64, device = self.device)# neighbour entries for [i-1,j]
        self.bom  = torch.zeros_like(self.V, dtype = torch.float64, device = self.device)# neighbour entries for [i,j-1]

        #time savers
        rx  = self.tau / (2 * self.hx)
        ry  = self.tau / (2 * self.hy)
        rxx = self.tau / (self.hx * self.hx)
        ryy = self.tau / (self.hy * self.hy)

        # x direction filter ; this is a backward difference kernel hence the extra 0 
        f1 = torch.tensor([[[[1], [-1], [0]]]], dtype = torch.float64, device = self.device)
        
        # y direction filter ; this is a backward difference kernel hence the extra 0 
        f2 = torch.tensor([[[[1, -1, 0]]]], dtype = torch.float64, device = self.device)

        # osmosis weights 
        boo = 1 + 2 * (rxx + ryy) \
                - rx * F.conv2d(self.d1, f1, padding='same') \
                - ry * F.conv2d(self.d2, f2, padding='same')
        
        # cloning stencils to avoid incorrect gradients due to in-place indexing
        self.boo = boo[:, :, 1:self.nx+1, 1:self.ny+1]

        # indexing to avoid boundaries being affected
        self.bpo = -rxx + rx * self.d1[:, :, 1:self.nx+1, 1:self.ny+1] # i, j+1
        self.bop = -ryy + ry * self.d2[:, :, 1:self.nx+1, 1:self.ny+1] # i+1, j
        self.bmo = -rxx - rx * self.d1[:, :,  :self.nx,   1:self.ny+1] # i-1, j
        self.bom = -ryy - ry * self.d2[:, :, 1:self.nx+1,  :self.ny  ] # i, j-1

        # create inverse central stencil for Jacobi
        self.inv_boo = 1 / self.boo

        # # create matrices for Jacobi
        # # x <= 1/D [b - (L + U) x]
        # # x <=  [-(L + U) / D] x  +   [1 / D] b  [Convenient form for Jacobi]
        # # x <=  T x  +   C b
        # diag_mat = torch.diag(torch.ones(self.V.shape[-1]))[None, None, :, :]
        # lu_diag_mat = torch.ones(self.V.shape[-1])[None, None, :, :] - diag_mat

        # # C = 1 / D ; [avoid infinities]
        # self.c_boo = 1 / (self.boo * diag_mat)
        # self.c_bpo = 1 / (self.bpo * diag_mat)
        # self.c_bop = 1 / (self.bop * diag_mat)
        # self.c_bmo = 1 / (self.bmo * diag_mat)
        # self.c_bom = 1 / (self.bom * diag_mat)

        # # T <= [-(L + U) / D]
        # self.t_boo = - (lu_diag_mat * self.boo) * self.c_boo
        # self.t_bpo = - (lu_diag_mat * self.bpo) * self.c_bpo
        # self.t_bop = - (lu_diag_mat * self.bop) * self.c_bop
        # self.t_bmo = - (lu_diag_mat * self.bmo) * self.c_bmo
        # self.t_bom = - (lu_diag_mat * self.bom) * self.c_bom
        
        if verbose :
            print(self.boo.shape)
            self.analyseImage(self.boo, "boo")
            print(self.bpo.shape)
            self.analyseImage(self.bpo, "bpo")
            print(self.bop.shape)
            self.analyseImage(self.bop, "bop")
            print(self.bmo.shape)
            self.analyseImage(self.bmo, "bmo")
            print(self.bom.shape)
            self.analyseImage(self.bom, "bom")

    def applyStencilGS(self, inp, boo, bmo, bom, bop, bpo, verbose = False):
        """
        inp : (batch, channel, nx, ny)
        """
        inp     = self.pad(inp[:, :, 1:self.nx+1, 1:self.ny+1])

        # print(inp.shape, boo.shape, bmo.shape, bom.shape, bop.shape, bpo.shape)

        # from top to bottom -> center, left, down, up, right
        res     = boo * inp[:, :, 1:self.nx+1, 1:self.ny+1] \
                + bmo * inp[:, :,  :self.nx,   1:self.ny+1] \
                + bom * inp[:, :, 1:self.nx+1,  :self.ny  ] \
                + bop * inp[:, :, 1:self.nx+1, 2:self.ny+2] \
                + bpo * inp[:, :, 2:self.nx+2, 1:self.ny+1]
                
        if verbose :
            self.analyseImage(res, "X")

        return self.zero_pad(res)

    def applyStencil_LU(self, inp, bmo, bom, bop, bpo, verbose = False):
        """
        (L + U) inp
        inp : (batch, channel, nx, ny)
        """
        inp     = self.pad(inp[:, :, 1:self.nx+1, 1:self.ny+1])

        # from top to bottom -> left, down, up, right
        # res     = boo * inp[:, :, 1:self.nx+1, 1:self.ny+1] \
        res     = bmo * inp[:, :,  :self.nx,   1:self.ny+1] \
                + bom * inp[:, :, 1:self.nx+1,  :self.ny  ] \
                + bop * inp[:, :, 1:self.nx+1, 2:self.ny+2] \
                + bpo * inp[:, :, 2:self.nx+2, 1:self.ny+1]
        
        if verbose :
            self.analyseImage(res, "X")

        return self.zero_pad(res)
    
    def applyStencil_D(self, inp, inv_boo, verbose = False):
        """
        (L + U) inp
        inp : (batch, channel, nx, ny)
        """
        inp     = self.pad(inp[:, :, 1:self.nx+1, 1:self.ny+1])

        res     = inv_boo * inp[:, :, 1:self.nx+1, 1:self.ny+1]
        
        if verbose :
            self.analyseImage(res, "X")

        return self.zero_pad(res)

    def zeroPadGS(self, x):
        return self.zero_pad(x[ :, :, 1:self.nx+1, 1 :self.ny+1])

    def Jacobi(self, x, b, kmax, eps, verbose = False):
        k       = torch.zeros((self.batch, self.channel), dtype=torch.long, device = self.device) 
        r_abs   = torch.zeros((self.batch, self.channel), dtype=torch.float64, device = self.device) 

        x_int       = torch.zeros_like(x, dtype = torch.float64) 

        r_0 = self.zeroPadGS(b - self.applyStencilGS(x, self.boo, self.bmo, self.bom, self.bop, self.bpo))      
        r_abs = torch.norm(r_0, dim = (2, 3), p = "fro")

        while ( (k < kmax) & (r_abs > eps * self.nx * self.ny) ).any():
            
            # =======================================
            # WHILE CONVERGENCE CONDITION
            # =======================================
            CONV_COND = (k < kmax) & (r_abs > eps * self.nx * self.ny)

            # JACKOBI ITERATION
            x_int = torch.where(CONV_COND[:, :, None, None], self.zeroPadGS(b - self.applyStencil_LU(x, self.bmo, self.bom, self.bop, self.bpo)), x_int)
            x     = torch.where(CONV_COND[:, :, None, None], self.zeroPadGS(self.applyStencil_D(x_int, self.inv_boo)), x)

            # residual calculation
            r_0   = torch.where(CONV_COND[:, :, None, None], self.zeroPadGS(b - self.applyStencilGS(x, self.boo, self.bmo, self.bom, self.bop, self.bpo)), r_0)      
            r_abs = torch.where(CONV_COND[:, :, None, None], torch.norm(r_0, dim = (2, 3), p = "fro"), r_abs)

            k     = torch.where(CONV_COND, k+1, k) 

        return x, torch.max(k)
