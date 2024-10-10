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
from torch.nn import Parameter


class BiCG_Module(nn.Module):
    def __init__(self, ):
        super(BiCG_Module, self).__init__()
        pass

    def forward(self, ):
        pass

def create_backward_hook(var_name):
    def hook(grad):
        # print(f"Gradient of {var_name}\n grad norm : {grad.norm()}\n grad stats:\n {self.analyseImage(grad, var_name)}")
        print(f"Gradient of {var_name}\n grad norm : {grad.norm()}")
    return hook

class BiCG_Net(nn.Module):
    def __init__(self, offset, tau, b, c, nx, ny, hx = 1., hy = 1.):
        super(BiCG_Net, self).__init__()

        # shape
        self.batch, self.channel = b, c
        self.nx, self.ny = nx, ny

        self.offset  = offset
        self.tau     = tau
                
        # pixel sizes x and y direction
        self.hx      = hx
        self.hy      = hy    

        # img 
        self.save_every  = 10

        # others
        self.pad      = Pad(1, padding_mode = "symmetric") # mirror
        self.zero_pad = Pad(1,fill=0, padding_mode='constant') # zero padding

    def forward(self, V, mask1, mask2):

        # drift matrices 
        self.d1  = torch.zeros((self.batch,self.channel,self.nx+2,self.ny+2), requires_grad = True)
        self.d2  = torch.zeros((self.batch,self.channel,self.nx+2,self.ny+2), requires_grad = True)

        # stencil matrices 
        self.boo  = torch.zeros((self.batch,self.channel,self.nx+2,self.ny+2), requires_grad = True) # C++ weickert init. has ones
        self.bop  = torch.zeros((self.batch,self.channel,self.nx+2,self.ny+2), requires_grad = True) # neighbour entries for [i+1,j]
        self.bpo  = torch.zeros((self.batch,self.channel,self.nx+2,self.ny+2), requires_grad = True) # neighbour entries for [i,j+1]
        self.bmo  = torch.zeros((self.batch,self.channel,self.nx+2,self.ny+2), requires_grad = True) # neighbour entries for [i-1,j]
        self.bom  = torch.zeros((self.batch,self.channel,self.nx+2,self.ny+2), requires_grad = True) # neighbour entries for [i,j-1]

        # bi-cg matrices


        #========
        # create init from guidance
        #========
        
        U     = torch.ones_like(V) * torch.mean(V, dim = (2,3), keepdim = True)

        U     = torch.transpose(self.pad(U), 2, 3) + self.offset
        V     = torch.transpose(self.pad(V), 2, 3) + self.offset
        mask1 = torch.transpose(self.pad(mask1), 2, 3)
        mask2 = torch.transpose(self.pad(mask2), 2, 3)
        self.nx, self.ny = self.ny, self.nx

        #========
        # calculate drift vectors                
        #========
        # row-direction filters  
        f1 = torch.tensor([[[[-1./self.hx], [1./self.hx]]]], dtype = torch.float64)
        f2 = torch.tensor([[[[.5], [.5]]]], dtype = torch.float64)
        
        # col-direction filters
        f3 = torch.tensor([[[[-1./self.hy, 1./self.hy]]]], dtype = torch.float64)
        f4 = torch.tensor([[[[.5, .5]]]], dtype = torch.float64)

        d1 = F.conv2d(V, f1) / F.conv2d(V, f2)
        d2 = F.conv2d(V, f3) / F.conv2d(V, f4) 

        self.d1 = self.d1.clone()
        self.d1[:, :, :self.nx+1, 1:self.ny+1] = d1[:, :, :, 1:self.ny+1] # convolved and reduced in row dir , hence one less
        self.d2 = self.d2.clone()
        self.d2[:, :, 1:self.nx+1, :self.ny+1] = d2[:, :, 1:self.nx+1, :] # convolved and reduced in col dir , hence one less
        
        #========
        # apply mask
        #========
        self.d1 = torch.mul(self.d1, mask1)
        self.d2 = torch.mul(self.d2, mask2)

        #========
        # calculate stencils 
        #========
        rx  = self.tau / (2. * self.hx)
        ry  = self.tau / (2. * self.hy)
        rxx = self.tau / (self.hx * self.hx)
        ryy = self.tau / (self.hy * self.hy)

        # x direction filter ; this is a backward difference kernel hence the extra 0 
        f1 = torch.tensor([[[[1], [-1], [0]]]], dtype = torch.float64)
        
        # y direction filter ; this is a backward difference kernel hence the extra 0 
        f2 = torch.tensor([[[[1, -1, 0]]]], dtype = torch.float64)

        # osmosis weights 
        boo = 1 + 2 * (rxx + ryy) - rx * F.conv2d(self.d1, f1, padding='same') - ry * F.conv2d(self.d2, f2, padding='same')
        
        # cloning stencils to avoid incorrect gradients due to in-place indexing
        self.boo = boo[:, :, 1:self.nx+1, 1:self.ny+1]

        # indexing to avoid boundaries being affected
        self.bpo = -rxx + rx * self.d1[:, :, 1:self.nx+1, 1:self.ny+1]
        self.bop = -ryy + ry * self.d2[:, :, 1:self.nx+1, 1:self.ny+1]
        self.bmo = -rxx - rx * self.d1[:, :,  :self.nx,   1:self.ny+1]
        self.bom = -ryy - ry * self.d2[:, :, 1:self.nx+1,  :self.ny  ]

        #========
        # solver
        #========
        x = U
        x = self.BiCGSTAB_GS(x = U, b = x, kmax = 30, eps = 1e-9, verbose=False)

        return x[:, :, 1:-1, 1:-1]


    def applyStencilGS(self, x, boo, bmo, bom, bop, bpo, verbose = False):
        """
        inp : (batch, channel, nx, ny)
        """
        inp     = self.pad(x[:, :, 1:self.nx+1, 1:self.ny+1])

        # from top to bottom => center, left, down, up, right
        res     = boo * inp[:, :, 1:self.nx+1, 1:self.ny+1] \
                + bmo * inp[:, :,  :self.nx  , 1:self.ny+1] \
                + bom * inp[:, :, 1:self.nx+1,  :self.ny  ] \
                + bop * inp[:, :, 1:self.nx+1, 2:self.ny+2] \
                + bpo * inp[:, :, 2:self.nx+2, 1:self.ny+1]                

        if verbose :
            self.analyseImage(res, "X")

        return self.zero_pad(res)

    def zeroPadGS(self, x):
        return self.zero_pad(x[ :, :, 1:self.nx+1, 1 :self.ny+1])

    def BiCGSTAB_GS(self, x, b, kmax=10000, eps=1e-9, verbose = False):
        
        k       = torch.zeros((self.batch, self.channel), dtype=torch.long)
        r_abs   = torch.zeros((self.batch, self.channel), dtype=torch.float64)
        v_abs   = torch.zeros((self.batch, self.channel), dtype=torch.float64)
        r0_abs  = torch.zeros((self.batch, self.channel), dtype=torch.float64)
        sigma   = torch.zeros((self.batch, self.channel), dtype=torch.float64) 
        alpha   = torch.zeros((self.batch, self.channel), dtype=torch.float64) 
        omega   = torch.zeros((self.batch, self.channel), dtype=torch.float64) 
        beta    = torch.zeros((self.batch, self.channel), dtype=torch.float64) 

        r_0     = torch.zeros_like(x, dtype = torch.float64)
        r       = torch.zeros_like(x, dtype = torch.float64) 
        r_old   = torch.zeros_like(x, dtype = torch.float64) 
        p       = torch.zeros_like(x, dtype = torch.float64) 
        v       = torch.zeros_like(x, dtype = torch.float64) 
        s       = torch.zeros_like(x, dtype = torch.float64) 
        t       = torch.zeros_like(x, dtype = torch.float64) 


        r_0 = self.applyStencilGS(x, self.boo, self.bmo, self.bom, self.bop, self.bpo) 
        p   = self.zeroPadGS(b - r_0)
        # print(self.analyseImage(p, "init p"))
        r_0 = r = p
        r0_abs = torch.norm(r_0, dim = (2, 3), p = "fro")
        r_abs  = r0_abs

        if verbose:
            print(f"r_abs : {r_abs}, shape : {r_abs.shape}")
    
        while ( (k < kmax) & (r_abs > eps * self.nx * self.ny) ).any():

            # =======================================
            # WHILE CONVERGENCE CONDITION
            # =======================================
            
            CONV_COND = (k < kmax) & (r_abs > eps * self.nx * self.ny) 
            if verbose:
                print(f"WHILE CONVERGENCE CONDITION :\n {CONV_COND} and shape : {CONV_COND.shape}")
            
            v = torch.where(CONV_COND[:, :, None, None], self.applyStencilGS(p, self.boo, self.bmo, self.bom, self.bop, self.bpo), v)
            sigma = torch.where(CONV_COND, torch.sum(torch.mul(v, r_0), dim = (2, 3)), sigma)
            v_abs = torch.where(CONV_COND, torch.norm(v, dim = (2, 3),  p = "fro"), v_abs)
            
            # print(self.analyseImage(sigma[:, :, None, None], "sigma"))
            # print(self.analyseImage(v, "v"))
            
            if verbose:
                print(f"k : {k}, sigma : {sigma}, vabs : {v_abs}")
            
            # =======================================
            # SET RESTART CONDITION
            # =======================================

            RES_COND = (sigma <= eps * v_abs * r0_abs)
            RES1_COND = CONV_COND & RES_COND
            RES1_COND_EXP = RES1_COND[:, :, None, None]

            if verbose:
                print(f"RESTART REQUIRED :\n {RES1_COND}, shape : {RES1_COND.shape}")

            p = torch.where(RES1_COND_EXP, self.zeroPadGS(b - self.applyStencilGS(x, self.boo, self.bmo, self.bom, self.bop, self.bpo)), p)
            r = torch.where(RES1_COND_EXP, p, r)
            r_0 = torch.where(RES1_COND_EXP, p, r_0)
            r0_abs  = torch.where(RES1_COND, torch.norm(r_0, dim = (2, 3), p = "fro"), r0_abs)
            r_abs = torch.where(RES1_COND, r0_abs, r_abs)
            k = torch.where(RES1_COND, k+1, k)

            # print(self.analyseImage(r_abs[:, :, None, None], "r_abs"))
            # print(self.analyseImage(p, "1st p"))

            if verbose:
                print(f"r_abs when restarted: {r_abs}")

            # =======================================
            # INVERSE RESTART CONDITION : systems that dont require restart
            # =======================================
            NOT_RES_COND = CONV_COND & (~RES_COND)

            if verbose:
                print(f"RESTART NOT REQUIRED :\n {NOT_RES_COND}")

            alpha = torch.where(NOT_RES_COND, torch.sum( torch.mul(r, r_0), dim = (2, 3)) / sigma , alpha)
            
            if verbose:
                print(f"k : {k}, alpha : {alpha}")

            s = torch.where(NOT_RES_COND[:, :, None, None], r - (alpha[:, :, None, None] * v), s)
            # print(self.analyseImage(alpha[:, :, None, None], "alpha"))
            # print(self.analyseImage(s, "s"))
            # =======================================
            # No RESTART and CONVERGENCE CONDITION 
            # =======================================

            CONV2_COND = torch.norm(s, dim = (2, 3), p = 'fro') <= eps * self.nx * self.ny
            CONV3_COND = NOT_RES_COND & CONV2_COND
            CONV3_COND_EXP = CONV3_COND[:, :, None, None]

            if verbose:
                print(f"RESTART NOT REQUIRED and CONV :\n {CONV3_COND}")

            x = torch.where(CONV3_COND_EXP, x + alpha[:, :, None, None] * p, x )
            r = torch.where(CONV3_COND_EXP, s, r)
            # print(self.analyseImage(x, "x"))
            # print(self.analyseImage(r, "r"))
            
            # =======================================
            # No RESTART and INVERSE CONVERGENCE CONDITION 
            # =======================================
            CONV4_COND = NOT_RES_COND & (~CONV2_COND)
            CONV4_COND_EXP = CONV4_COND[:, :, None, None]  # this is to match the dimention when matching using torch.where

            if verbose:
                print(f"RESTART NOT REQUIRED and ELSE CONV :\n {CONV4_COND}")

            t = torch.where(CONV4_COND_EXP, self.applyStencilGS(s, self.boo, self.bmo, self.bom, self.bop, self.bpo), t)
            # print(self.analyseImage(t, "t"))
            omega  = torch.where(CONV4_COND, torch.sum( torch.mul(t, s), dim = (2, 3)) / torch.sum(t**2, dim = (2, 3)), omega )            
            x = torch.where(CONV4_COND_EXP, x + (alpha[:, :, None, None] * p) + (omega[:, :, None, None] * s), x)
            # print(self.analyseImage(x, "x"))
            r_old = torch.where(CONV4_COND_EXP, r, r_old)
            r = torch.where(CONV4_COND_EXP, s - omega[:, :, None, None] * t, r)
            # print(self.analyseImage(r, "r"))
            beta = torch.where(CONV4_COND
                            , (alpha / omega) * (torch.sum(torch.mul(r, r_0), dim = (2, 3)) / torch.sum(torch.mul(r_old, r_0), dim = (2, 3)))
                            , beta)
            # print(self.analyseImage(omega[:, :, None, None], "omega"))
            # print(self.analyseImage(beta[:, :, None, None], "beta"))

            if verbose:
                print(f"k : {k} , omega : {omega}, beta : {beta}")

            p = torch.where(CONV4_COND_EXP, r + beta[:, :, None, None] * ( p - omega[:, :, None, None] * v), p)
            # print(self.analyseImage(p, "2nd p"))

            # =======================================
            # NOT REQUIRING RESTART SYSTEMS ; UPDATE
            # =======================================

            k  = torch.where(NOT_RES_COND, k+1, k) 
            r_abs = torch.where(NOT_RES_COND, torch.norm(r, dim = (2, 3), p = 'fro'), r_abs)

            # if verbose:
            print(f"k : {k}, RESIDUAL : {r_abs}")


            v_abs.register_hook(create_backward_hook("v_abs"))
            r0_abs.register_hook(create_backward_hook("r0_abs"))
            sigma.register_hook(create_backward_hook("sigma"))
            alpha.register_hook(create_backward_hook("alpha"))
            omega.register_hook(create_backward_hook("omega"))
            beta.register_hook(create_backward_hook("beta"))

            r_0.register_hook(create_backward_hook("r_0"))
            r.register_hook(create_backward_hook("r"))
            r_old.register_hook(create_backward_hook("r_old"))
            p.register_hook(create_backward_hook("p"))
            v.register_hook(create_backward_hook("v"))
            s.register_hook(create_backward_hook("s"))
            t.register_hook(create_backward_hook("t"))

        self.boo.register_hook(create_backward_hook("boo"))
        self.bpo.register_hook(create_backward_hook("bpo"))
        self.bop.register_hook(create_backward_hook("bop"))
        self.bom.register_hook(create_backward_hook("bom"))
        self.bmo.register_hook(create_backward_hook("bmo"))
        self.d1.register_hook(create_backward_hook("d1"))
        self.d2.register_hook(create_backward_hook("d2"))

        return x


