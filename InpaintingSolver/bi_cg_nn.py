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


class BiCG_Module(nn.Module):
    def __init__(self, ):
        super(BiCG_Module, self).__init__()

        pass

    def forward(self, ):
        pass

class BiCG_Net(nn.Module):
    def __init__(self, U, V, mask1, mask2, offset, tau, hx = 1, hy = 1, device = None, apply_canny = False):
        super(BiCG_Net, self).__init__()
        # (b, c, h, w)
        self.V       = V + offset  # guidance image
        self.batch   = V.size(0) 
        self.channel = V.size(1) 
        self.nx      = V.size(2) 
        self.ny      = V.size(3) 

        self.offset  = offset
        self.tau     = tau
                
        # pixel sizes x and y direction
        self.hx      = hx
        self.hy      = hy    

        # img 
        self.save_every  = 10
        self.apply_canny = apply_canny
        self.canny_mask  = None

        # drift matrices 
        self.d1  = torch.zeros_like(self.V, requires_grad = True)
        self.d2  = torch.zeros_like(self.V, requires_grad = True)

        # stencil matrices 
        self.boo  = torch.zeros_like(self.V, requires_grad = True)# C++ weickert init. has ones
        self.bop  = torch.zeros_like(self.V, requires_grad = True)# neighbour entries for [i+1,j]
        self.bpo  = torch.zeros_like(self.V, requires_grad = True)# neighbour entries for [i,j+1]
        self.bmo  = torch.zeros_like(self.V, requires_grad = True)# neighbour entries for [i-1,j]
        self.bom  = torch.zeros_like(self.V, requires_grad = True)# neighbour entries for [i,j-1]

        # bi-cg matrices 

        # others
        self.pad = Pad(1, padding_mode = "symmetric")


    def forward(self, V, mask1, mask2):
        
        #========
        # create init from guidance
        #========
        U  = torch.ones_like(V) * torch.mean(V, dim = (2,3), keepdim = True)

        U     = torch.transpose(self.pad(U), 2, 3)
        V     = torch.transpose(self.pad(V), 2, 3)
        mask1 = torch.transpose(self.pad(mask1), 2, 3)
        mask2 = torch.transpose(self.pad(mask2), 2, 3)
        self.nx, self.ny = self.ny, self.nx

        #========
        # calculate drift vectors                
        #========

        # row-direction filters  
        f1 = torch.tensor([[[[-1/self.hx], [1/self.hx]]]])
        f2 = torch.tensor([[[[.5], [.5]]]])
        
        # col-direction filters
        f3 = torch.tensor([[[[-1/self.hy, 1/self.hy]]]])
        f4 = torch.tensor([[[[.5, .5]]]])

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

        rx  = self.tau / (2 * self.hx)
        ry  = self.tau / (2 * self.hy)
        rxx = self.tau / (self.hx * self.hx)
        ryy = self.tau / (self.hy * self.hy)

        # x direction filter ; this is a backward difference kernel hence the extra 0 
        f1 = torch.tensor([[[[1], [-1], [0]]]])
        
        # y direction filter ; this is a backward difference kernel hence the extra 0 
        f2 = torch.tensor([[[[1, -1, 0]]]])

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


        x = self.BiCGSTAB_GS(x = U, b = x, kmax = 300, eps = 1e-9, verbose=verbose)
        U = x

        return U




    def applyStencilGS(self, inp, boo, bmo, bom, bop, bpo, verbose = False):
        """
        inp : (batch, channel, nx, ny)
        """
        pad_mirror = Pad(1, padding_mode = "symmetric")
        inp        = pad_mirror(inp[:, :, 1:self.nx+1, 1:self.ny+1])
        temp       = torch.zeros_like(inp, dtype = torch.float64)

        # print(inp.shape, boo.shape, bmo.shape, bom.shape, bop.shape, bpo.shape)

        # from top to bottom -> center, left, down, up, right
        res     = boo * inp[:, :, 1:self.nx+1, 1:self.ny+1] \
                + bmo * inp[:, :,  :self.nx,   1:self.ny+1] \
                + bom * inp[:, :, 1:self.nx+1,  :self.ny  ] \
                + bop * inp[:, :, 1:self.nx+1, 2:self.ny+2] \
                + bpo * inp[:, :, 2:self.nx+2, 1:self.ny+1]
                
        temp = temp.clone()
        temp[:, :, 1:self.nx+1, 1:self.ny+1 ] = res

        if verbose :
            self.analyseImage(temp, "X")

        return temp

    def zeroPadGS(self, x):
        t = torch.zeros_like(x, dtype = torch.float64, device = self.device)
        t = t.clone()
        t[:, :, 1:self.nx+1, 1 :self.ny+1] = x[ :, :, 1:self.nx+1, 1 :self.ny+1]
        return t

    def BiCGSTAB_GS(self, x, b, kmax=10000, eps=1e-9, verbose = False):
        
        k       = torch.zeros((self.batch, self.channel), dtype=torch.long, device = self.device)
        r_abs   = torch.zeros((self.batch, self.channel), dtype=torch.float64, device = self.device, requires_grad = True)
        v_abs   = torch.zeros((self.batch, self.channel), dtype=torch.float64, device = self.device, requires_grad = True)
        r0_abs  = torch.zeros((self.batch, self.channel), dtype=torch.float64, device = self.device, requires_grad = True)
        sigma   = torch.zeros((self.batch, self.channel), dtype=torch.float64, device = self.device, requires_grad = True) 
        alpha   = torch.zeros((self.batch, self.channel), dtype=torch.float64, device = self.device, requires_grad = True) 
        omega   = torch.zeros((self.batch, self.channel), dtype=torch.float64, device = self.device, requires_grad = True) 
        beta    = torch.zeros((self.batch, self.channel), dtype=torch.float64, device = self.device, requires_grad = True) 

        r_0     = torch.zeros_like(x, dtype = torch.float64, requires_grad = True)
        r       = torch.zeros_like(x, dtype = torch.float64, requires_grad = True) 
        r_old   = torch.zeros_like(x, dtype = torch.float64, requires_grad = True) 
        p       = torch.zeros_like(x, dtype = torch.float64, requires_grad = True) 
        v       = torch.zeros_like(x, dtype = torch.float64, requires_grad = True) 
        s       = torch.zeros_like(x, dtype = torch.float64, requires_grad = True) 
        t       = torch.zeros_like(x, dtype = torch.float64, requires_grad = True) 


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

        return x


