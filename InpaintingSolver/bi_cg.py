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
        return torch.mean(torch.norm(U-V, p = 2, dim = (2,3))**2 / nxny)

class OsmosisInpainting:

    def __init__(self, U, V, mask1, mask2, offset, tau, hx = 1, hy = 1, device = None, apply_canny = False):
        # (b, c, h, w)
        self.V       = V + offset  # guidance image
        self.batch   = V.size(0) 
        self.channel = V.size(1) 
        self.nx      = V.size(2) 
        self.ny      = V.size(3) 

        self.device = device

        if U is not None:
            self.U   = U + offset  # original image
        else:
            self.U   = self.getInit_U() + offset

        if mask1 is None or mask2 is None:
            self.mask1 = torch.ones_like(V)
            self.mask2 = torch.ones_like(V)
        else :
            self.mask1   = mask1
            self.mask2   = mask2

        self.offset  = offset
        self.tau     = tau
                
        # pixel sizes x and y direction
        self.hx      = hx
        self.hy      = hy    

        # img 
        self.save_every  = 10
        self.apply_canny = apply_canny
        self.canny_mask  = None

    def solve(self, kmax = 2, save_every = 10, verbose = False):
        
        self.save_every = save_every

        X = self.U.detach().clone()
        print(self.analyseImage(X, f"Initial img"))
        print()
        tt = 0

        for i in range(kmax):
            print(f"ITERATION : {i+1}")

            st = time.time()
            X, _, _ = self.BiCGSTAB(x = self.U, b = X, batch = 0, kmax = 10000, eps = 1e-9, verbose=verbose)
            et = time.time()
            tt += (et-st)
            self.U = X
            
            comm = self.analyseImage(self.U, f"evolving U at iter {i}")
            comm += f"time for iteration : {str(et-st)} sec\n"
            comm += f"total time         : {str(tt)} sec\n"
            comm += self.getMetrics()
            print(comm)

            if (i+1) % self.save_every == 0:
                self.U = self.U - self.offset
                
                fname = f"solved_{str(i+1)}.pgm"
                self.writePGMImage(self.U[0][0].numpy().T, fname)
                # self.writeToPGM(fname = fname, t = self.U[0][0].T, comments= comm)
                self.U = self.U + self.offset

    def solveBatchParallel(self, kmax = 100, save_batch = False, verbose = False):
                        
        X = self.U
        U = self.U
        tt = 0
        
        mse = MSELoss()

        st = time.time()

        for i in range(kmax):

            X = self.BiCGSTAB_GS(x = U, b = X, kmax = 3000, eps = 1e-9, verbose=verbose)
            U = X
            loss = mse( self.normalize(U), self.normalize(self.V))
            print(f"\rITERATION : {i+1}, loss : {loss.item()} ", end ='', flush=True)        
            
        et = time.time()
        tt += (et-st)

        self.U = X

        comm = self.analyseImage(X, f"solution")
        comm += f"time for iteration : {str(et-st)} sec\n"
        comm += f"total time         : {str(tt)} sec\n"
        comm += self.getMetrics()
        print(comm)

        if save_batch[0]:
            fname = save_batch[1]

            out = torch.cat( ( (self.V.type(torch.float64).reshape(self.batch*(self.nx+2), self.ny+2) - self.offset), 
                            (self.mask1 * 255.).reshape(self.batch*(self.nx+2), self.ny+2), 
                            # (self.canny_mask * 255.).reshape(self.batch*(self.nx+2), self.ny+2), 
                            # (init-self.offset)[0][0],
                            (self.U-self.offset).reshape(self.batch*(self.nx+2), self.ny+2)),
                            dim = 1)
            self.writePGMImage(out.cpu().detach().numpy().T, fname)

        # normalize solution and guidance
        U = self.normalize(self.U)
        V = self.normalize(self.V)
        
        # calculate loss self.U and self.V
        loss = mse(U, V)

        return loss, tt

    def solveBatchSeq(self, kmax , save_batch = False, verbose = False):

        tt = 0
        mse = MSELoss()

        total_loss = 0.

        st = time.time()

        for batch in range(self.batch):

            V = self.V[batch].unsqueeze(0).detach().clone()
            B = self.U[batch].unsqueeze(0).detach().clone()
            U = B.detach().clone()   #.detach().clone().to(self.device)
            init = B.detach().clone().to(self.device)
            
            if verbose:
                print(f"batch item : {batch+1} / {self.batch}")    

            loss = 0.
            count  = 5
            for i in range(kmax):
                B, restart, k = self.BiCGSTAB(x = U, b = B, batch = batch, kmax = 10000, eps = 1e-9, verbose=verbose)
                U = B

                # for systems stuck at the same loss for over [5] iterations
                if torch.abs(mse( self.normalize(U), self.normalize(V)) - loss) < 1e-5:
                    count -= 1

                loss = mse( self.normalize(U), self.normalize(V))

                if torch.isnan(loss):
                    print(f"U : {U}") 
                    print(f"norm U : {self.normalize(U)}")    
                    print(f"init : {init}")         
                print(f"\rITERATION : {i+1}, loss : {loss.item()} ", end ='', flush=True)        
                
                # for systems that are stuck in restart ; skip solving those systems
                if count == 0 or (restart == 1 and k == 10000):
                    break

            print()

            if save_batch:
                fname = f"solved_.pgm"
                out = torch.cat( ( (self.V - self.offset)[batch][0], 
                                (self.mask1 * 255.)[batch][0], 
                                # (init-self.offset)[0][0],
                                self.normalize(U-self.offset, scale=255.)[0][0] ), dim  = 0)
                self.writePGMImage(out.cpu().detach().numpy().T, fname)

            self.U[batch] = U[0]
        
        et = time.time()
        tt += (et-st)
                
        # normalize solution and guidance
        U = self.normalize(self.U)
        V = self.normalize(self.V)
        
        # calculate loss self.U and self.V
        loss = mse(U, V)
        
        
        return loss , tt
        
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
        
    def readPGMImage(self, pth):
        pgm = cv2.imread(pth, cv2.IMREAD_GRAYSCALE) 
        pgm_T = torch.tensor(pgm, dtype = torch.float64)
        self.nx, self.ny = pgm_T.size()
        pgm_T = pgm_T.reshape(1, 1, self.nx, self.ny)
        self.batch = 1
        self.channel = 1
        self.hx = 1
        self.hy = 1
        return pgm_T

    def writeToPGM(self, fname, t, comments):
        image = t.numpy()
        height, width = image.shape
        image = cv2.normalize(
                    image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        with open(fname, 'wb') as f:
            f.write(b'P5\n')

            if comments:
                for comment in comments:
                    f.write(f'# {comment}\n'.encode())

            f.write(f'{height} {width}\n'.encode())
            f.write(b'255\n')
            f.write(image.tobytes())

        print(f"written to : {fname}")

    def writePGMImage(self, X, filename):
        # add comments for pgm img
        cv2.imwrite(filename, X[1:-1, 1:-1])
        # print(f"written to : {filename}")

    def prepareInp(self):
        """
        transposed because Weickert transposed it in his C code
        """
        pad_mirror = Pad(1, padding_mode = "symmetric")

        self.U     = pad_mirror(self.U)
        self.U     = torch.transpose(self.U, 2, 3)

        self.V     = pad_mirror(self.V)
        self.V     = torch.transpose(self.V, 2, 3)

        if self.mask1 != None :
            self.mask1     = pad_mirror(self.mask1)
            self.mask1     = torch.transpose(self.mask1, 2, 3)

        if self.mask2 != None:
            self.mask2     = pad_mirror(self.mask2)
            self.mask2     = torch.transpose(self.mask2, 2, 3)

        # since we transposed 
        self.nx, self.ny = self.ny, self.nx

    def analyseImage(self, x, name):
        comm = ""
        x = x[:, :, 1:self.nx+1, 1:self.ny+1]
        
        print(f"analyzing {name}")
        
        comm += f"min  : {str(torch.min(x).item())}\n"
        comm += f"max  : {str(torch.max(x).item())}\n"
        comm += f"mean : {str(torch.mean(x).item())}\n"
        comm += f"std  : {str(torch.std(x).item())}\n"

        return comm
        
    def getMetrics(self):
        metrics = ""
        psnr  = PeakSignalNoiseRatio().to(self.device)
        mse   = torch.nn.MSELoss()

        metrics += f"psnr : {str(( psnr(self.U, self.V) ))}\n"
        metrics += f"mse  : {str(( mse(self.U, self.V) ))}\n"
        
        return metrics

    def getInit_U(self):
        m  = torch.mean(self.V, dim = (2,3))

        # create a flat image ; avg gray val same as guidance
        u  = torch.ones_like(self.V, device = self.device) * m.view(self.batch, self.channel, 1, 1)

        # create a noisy image ; avg gray val same as guidance
        return u

    def hardRoundBinarize(self):
        if self.mask1 != None and self.mask2 != None:
            self.mask1 =  torch.floor(self.mask1 + 0.5)
            self.mask2 =  torch.floor(self.mask2 + 0.5)

    def createMaskfromCanny(self):
        output_batch = []
        images = self.V.detach().cpu().numpy()

        for image in images:
            image = image.squeeze(0) # assuming grey scale image
            edges = cv2.Canny(image.astype(np.uint8), 100, 150)
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
        self.d1  = torch.zeros_like(self.V, dtype = torch.float64, device = self.device)
        self.d2  = torch.zeros_like(self.V, dtype = torch.float64, device = self.device)
                
        # row-direction filters  
        f1 = torch.tensor([[[[-1/self.hx], [1/self.hx]]]], dtype = torch.float64, device = self.device)
        f2 = torch.tensor([[[[.5], [.5]]]], dtype = torch.float64, device = self.device)
        
        # col-direction filters
        f3 = torch.tensor([[[[-1/self.hy, 1/self.hy]]]], dtype = torch.float64, device = self.device)
        f4 = torch.tensor([[[[.5, .5]]]], dtype = torch.float64, device = self.device)

        d1 = F.conv2d(self.V, f1) / F.conv2d(self.V, f2)
        d2 = F.conv2d(self.V, f3) / F.conv2d(self.V, f4) 

        # correcting for dimentionality reduced by using F.conv2d
        # eg : 1 dimension reduced for d1  changes nx+2 -> nx+2-1
        # also avoiding boundaries in the complementary direction
        self.d1[:, :, :self.nx+1, 1:self.ny+1] = d1[:, :, :, 1:self.ny+1] # convolved and reduced in row dir , hence one less
        self.d2[:, :, 1:self.nx+1, :self.ny+1] = d2[:, :, 1:self.nx+1, :] # convolved and reduced in col dir , hence one less

        if verbose:
            print(f"V shape : {self.V.size()}")
            print(f"V_padded : \n{self.V[0][0]}\n")
            print(f"d1 : {self.d1[0][0]}")
            print(f"d2 : {self.d2[0][0]}")
            print(self.analyseImage(self.d1, "d1"))
            print(self.analyseImage(self.d2, "d2"))
            
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
        
        self.boo[:, :, 1:self.nx+1, 1:self.ny+1] = boo[:, :, 1:self.nx+1, 1:self.ny+1]

        # indexing to avoid boundaries being affected
        self.bpo[:, :, 1:self.nx+1, 1:self.ny+1] = -rxx + rx * self.d1[:, :, 1:self.nx+1, 1:self.ny+1]
        self.bop[:, :, 1:self.nx+1, 1:self.ny+1] = -ryy + ry * self.d2[:, :, 1:self.nx+1, 1:self.ny+1]

        self.bmo[:, :, 1:self.nx+1, 1:self.ny+1] = -rxx - rx * self.d1[:, :, :self.nx, 1:self.ny+1]
        self.bom[:, :, 1:self.nx+1, 1:self.ny+1] = -ryy - ry * self.d2[:, :, 1:self.nx+1, :self.ny]
 
        # slice indexing here to avoid repetitive indexing in BiCG solver
        self.boo = self.boo[ :, :, 1:self.nx+1, 1:self.ny+1]
        self.bpo = self.bpo[ :, :, 1:self.nx+1, 1:self.ny+1]
        self.bop = self.bop[ :, :, 1:self.nx+1, 1:self.ny+1]
        self.bmo = self.bmo[ :, :, 1:self.nx+1, 1:self.ny+1]
        self.bom = self.bom[ :, :, 1:self.nx+1, 1:self.ny+1]

        if verbose :
            print(self.boo)
            print(self.analyseImage(self.boo, "boo"))
            print(self.bpo)
            print(self.analyseImage(self.bpo, "bpo"))
            print(self.bop)
            print(self.analyseImage(self.bop, "bop"))
            print(self.bmo)
            print(self.analyseImage(self.bmo, "bmo"))
            print(self.bom)
            print(self.analyseImage(self.bom, "bom"))

    def applyStencil(self, inp, batch, verbose = False):
        """
        inp : (batch, channel, nx, ny)
        """
        pad_mirror = Pad(1, padding_mode = "symmetric")
        inp        = pad_mirror(inp[:, :, 1:self.nx+1, 1:self.ny+1])

        temp       = torch.zeros_like(inp, device = self.device)

        center     = torch.mul(self.boo[batch], inp[:, :, 1:self.nx+1, 1:self.ny+1])    
         
        left       = torch.mul(self.bmo[batch], inp[:, :, :self.nx, 1:self.ny+1])
        
        down       = torch.mul(self.bom[batch], inp[:, :, 1:self.nx+1, 0:self.ny])
        
        up         = torch.mul(self.bop[batch], inp[:, :, 1:self.nx+1, 2:self.ny+2])
        
        right      = torch.mul(self.bpo[batch], inp[:, :, 2:self.nx+2, 1:self.ny+1])
        
        temp[:, :, 1:self.nx+1, 1:self.ny+1 ] = center + left + right + up + down
        
        if verbose :
            self.analyseImage(temp, "X")

        return temp

    def zeroPad(self, x):
        t = torch.zeros_like(x, device = self.device)
        t[:, :, 1:self.nx+1, 1 :self.ny+1] = x[:, :, 1:self.nx+1, 1 :self.ny+1]
        return t

    def BiCGSTAB(self, x, b, batch, kmax=10000, eps=1e-9, verbose = False):
        """
        Biconjugate gradient stabilised method without preconditioning for
        solving a linear system A x = b with an unsymmetric, pentadiagonal
        system matrix A that involves four 2D neighbours.
        Follows the description in A. Meister: Numerik linearer Gleichungssysteme.
        Vieweg, Braunschweig, 1999.
        x : old and new solution ; torch.Tensor batch*channel*nx*ny
        b : right hand side      ; torch.Tensor batch*channel*nx*ny
        """
        restart = 1       
        k = 0 

        while restart == 1:
            
            restart = 0
            
            r_0 = self.applyStencil(x, batch)  
            r_0 = r = p  = self.zeroPad(b - r_0)  
            r_abs = r0_abs = torch.norm(r_0, p = 'fro') # avoid boundary calculations
            
            if verbose:
                print(f"k : {k} , when restarted r_abs : {r_abs}")

            while k < kmax and  \
                    r_abs > eps * self.nx * self.ny and \
                    restart == 0:
                
                v = self.applyStencil(p, batch) # output contains zero boundaries
                sigma = torch.sum( torch.mul(v, r_0))
                v_abs = torch.norm(v, p = 'fro') # avoid boundary pixel cal
                
                if verbose:
                    print(f"k : {k} , sigma : {sigma}, v_abs : {v_abs}")

                if sigma <= eps * v_abs * r0_abs:

                    restart = 1
                    k += 1
                    if verbose:
                        print(f"restarting ... k : {k} , sigma : {sigma} , vabs : {v_abs}")

                else :

                    # r, s contains boundaries
                    alpha = torch.sum( torch.mul(r, r_0)) / sigma
                    s     = r - alpha * v
                    
                    if verbose:
                        print(f"k : {k} , alpha : {alpha}")

                    if torch.norm(s, p = 'fro') <= eps * self.nx * self.ny:
                        
                        # x, r contains boundaries
                        x = x + alpha * p 
                        r = s
                    
                    else :

                        t = self.applyStencil(s, batch)
                        omega = torch.sum( torch.mul(t, s)) / torch.sum(torch.mul(t, t))
                                        
                        x = x + alpha * p + omega * s
                        r_old = r
                        r = s - omega * t 
                        beta = (alpha / omega) * torch.sum(torch.mul(r, r_0)) / torch.sum(torch.mul(r_old, r_0))
                    
                        if verbose:
                            print(f"k : {k} , omega : {omega}, beta : {beta}")

                        p = r + beta * (p - omega * v)

                    k += 1
                    r_abs = torch.norm(r, p = 'fro')
                    
                    if verbose:
                        print(f"k : {k} , RESIDUAL : {r_abs}")

        return x, restart, k

    def applyStencilBatch(self, inp, COND, verbose = False):
        """
        inp : (batch, channel, nx, ny)
        """
        pad_mirror = Pad(1, padding_mode = "symmetric")
        inp        = pad_mirror(inp[ :, 1:self.nx+1, 1:self.ny+1])

        temp       = torch.zeros_like(inp, dtype = torch.float64)

        center     = self.boo[COND] * inp[ :, 1:self.nx+1, 1:self.ny+1]
         
        left       = self.bmo[COND] * inp[ :, :self.nx, 1:self.ny+1]
        
        down       = self.bom[COND] * inp[ :, 1:self.nx+1, :self.ny]
        
        up         = self.bop[COND] * inp[ :, 1:self.nx+1, 2:self.ny+2]
        
        right      = self.bpo[COND] * inp[ :, 2:self.nx+2, 1:self.ny+1]
        
        temp = temp.clone()
        temp[ :, 1:self.nx+1, 1:self.ny+1 ] = center + left + right + up + down
        
        if verbose :
            self.analyseImage(temp, "X")

        return temp

    def zeroPadBatch(self, x):
        t = torch.zeros_like(x, dtype = torch.float64, device = self.device)
        t = t.clone()
        t[:, 1:self.nx+1, 1 :self.ny+1] = x[:, 1:self.nx+1, 1 :self.ny+1]
        return t

    def BiCGSTAB_Batched(self, x, b, kmax=10000, eps=1e-9, verbose = False):
        
        restart = torch.ones( (self.batch, self.channel), dtype=torch.bool, device = self.device)
        k       = torch.zeros((self.batch, self.channel), dtype=torch.long, device = self.device, requires_grad = False)
        r_abs   = torch.zeros((self.batch, self.channel), dtype=torch.float64, device = self.device)
        r0_abs  = torch.zeros((self.batch, self.channel), dtype=torch.float64, device = self.device)
        v_abs   = torch.zeros((self.batch, self.channel), dtype=torch.float64, device = self.device)
        sigma   = torch.zeros((self.batch, self.channel), dtype=torch.float64, device = self.device)
        alpha   = torch.zeros((self.batch, self.channel), dtype=torch.float64, device = self.device)
        omega   = torch.zeros((self.batch, self.channel), dtype=torch.float64, device = self.device)
        beta    = torch.zeros((self.batch, self.channel), dtype=torch.float64, device = self.device)

        r_0     = torch.zeros_like(x, dtype = torch.float64)
        r       = torch.zeros_like(x, dtype = torch.float64)
        r_old   = torch.zeros_like(x, dtype = torch.float64)
        p       = torch.zeros_like(x, dtype = torch.float64)
        v       = torch.zeros_like(x, dtype = torch.float64)
        s       = torch.zeros_like(x, dtype = torch.float64)
        t       = torch.zeros_like(x, dtype = torch.float64)

        # variables to check info. to skip solving systems 
        r_abs_init = torch.zeros( (self.batch, self.channel), dtype = torch.float64, device = self.device)
        r_abs_last = torch.zeros( (self.batch, self.channel), dtype = torch.float64, device = self.device)
        r_abs_skip = torch.zeros( (self.batch, self.channel), dtype = torch.float64, device = self.device)
        stagnant_count = torch.zeros( (self.batch, self.channel), dtype = torch.float64, device = self.device)

        RES_COND = restart == 1
        p = p.clone()
        p[RES_COND]       = self.zeroPadBatch(b[RES_COND] - self.applyStencilBatch(x[RES_COND], RES_COND))
        r = r.clone()
        r_0[RES_COND]     = r[RES_COND] = p[RES_COND]
        r0_abs = r0_abs.clone()
        r0_abs[RES_COND]  = torch.norm(r_0[RES_COND], dim = (1, 2), p = "fro") #1R
        r_abs = r_abs.clone()
        r_abs[RES_COND]   = r0_abs[RES_COND]

        ttt = 0
        ttt1 = 0
        
        if verbose:
            print(f"r_abs : {r_abs}")

        while ( (k < kmax) & (r_abs > eps * self.nx * self.ny) ).any(): # and (restart == 0).any():

            # =======================================
            # WHILE CONVERGENCE CONDITION
            # =======================================
            
            CONV_COND = (k < kmax) & (r_abs > eps * self.nx * self.ny) # and (restart == 0)
            if verbose:
                print(f"WHILE CONVERGENCE CONDITION :\n {CONV_COND}")
            
            v = v.clone()
            st = time.time()
            v_ = v[CONV_COND] = self.applyStencilBatch(p[CONV_COND], CONV_COND)
            et = time.time()
            ttt += (et - st)
            
            sigma = sigma.clone()

            st = time.time()
            sigma[CONV_COND]  = torch.sum(torch.mul(v_, r_0[CONV_COND]), dim = (1, 2))
            et = time.time()
            ttt1 += (et - st)

            v_abs = v_abs.clone()
            v_abs[CONV_COND]  = torch.norm(v_, dim = (1, 2),  p = "fro")
            
            if verbose:
                print(f"k : {k}, sigma : {sigma}, vabs : {v_abs}")
            
            # =======================================
            # SET RESTART CONDITION
            # =======================================

            RES_COND = (sigma <= eps * v_abs * r0_abs)
            RES1_COND = CONV_COND & RES_COND
            if verbose:
                print(f"RESTART REQUIRED :\n {RES1_COND}")

            # r_0[RES1_COND]     = self.applyStencilBatch(x[RES1_COND], RES1_COND)
            p = p.clone()
            st = time.time()            
            p[RES1_COND]       = self.zeroPadBatch(b[RES1_COND] - self.applyStencilBatch(x[RES1_COND], RES1_COND))
            et = time.time()
            ttt += (et - st)

            r = r.clone()
            r_0 = r_0.clone()
            r_0[RES1_COND]     = r[RES1_COND] = p[RES1_COND]
            r0_abs = r0_abs.clone()
            r0_abs[RES1_COND]  = torch.norm(r_0[RES1_COND], dim = (1, 2), p = "fro") #2R
            r_abs = r_abs.clone()
            r_abs[RES1_COND]   = r0_abs[RES1_COND]
            k[RES1_COND] += 1 

            if verbose:
                print(f"r_abs when restarted: {r_abs}")

            # =======================================
            # INVERSE RESTART CONDITION : systems that dont require restart
            # =======================================
            NOT_RES_COND = CONV_COND & (~RES_COND)
            if verbose:
                print(f"RESTART NOT REQUIRED :\n {NOT_RES_COND}")

            # broadcast 1
            # alpha[~RES1_COND] => shape : torch.Size([x])
            r_ = r[NOT_RES_COND]
            alpha = alpha.clone()
            alpha[NOT_RES_COND] = torch.sum( torch.mul(r_, r_0[NOT_RES_COND]), dim = (1, 2)).view(-1) / sigma[NOT_RES_COND] #3R
            if verbose:
                print(f"k : {k}, alpha : {alpha}")
            # broadcast 2
            # s[~RES1_COND] => shape : torch.Size([b*c, h, w])
            s = s.clone()
            s[NOT_RES_COND]     = r_ - (alpha[NOT_RES_COND].view(-1, 1, 1) * v[NOT_RES_COND])
            
            # if verbose:
            #     print(f"k : {k} , alpha : {alpha}")

            # =======================================
            # No RESTART and CONVERGENCE CONDITION 
            # =======================================

            CONV2_COND = torch.norm(s, dim = (2, 3), p = 'fro') <= eps * self.nx * self.ny #4R
            CONV3_COND = NOT_RES_COND & CONV2_COND

            if verbose:
                print(f"RESTART NOT REQUIRED and CONV :\n {CONV3_COND}")

            # print(f"RES1_COND shape : {RES1_COND.shape}")
            # print(f"CONV2_COND shape : {CONV2_COND.shape}")
            # print(f"CONV3_COND shape : {CONV3_COND.shape}")

            # broadcast 3
            x = x.clone()
            x[CONV3_COND] += (alpha[CONV3_COND].view(-1, 1, 1) * p[CONV3_COND] )
            r = r.clone()
            r[CONV3_COND] = s[CONV3_COND]#.detach().clone()
                
            # =======================================
            # No RESTART and INVERSE CONVERGENCE CONDITION 
            # =======================================
            CONV4_COND = NOT_RES_COND & (~CONV2_COND)

            if verbose:
                print(f"RESTART NOT REQUIRED and ELSE CONV :\n {CONV4_COND.item()}")

            s_ = s[CONV4_COND]
            t = t.clone()
            st = time.time()
            t_ = t[CONV4_COND] = self.applyStencilBatch(s_, CONV4_COND)
            et = time.time()
            ttt += (et - st)

            omega = omega.clone()
            omega_ = omega[CONV4_COND] = torch.sum( torch.mul(t_, s_), dim = (1, 2)) / torch.sum(t_**2, dim = (1, 2))

            # broadcast 4        
            p_ = p[CONV4_COND]                        
            x  = x.clone()
            x[CONV4_COND] += (alpha[CONV4_COND].view(-1, 1, 1) * p_) + (omega_.view(-1, 1, 1) * s_)
            r_old = r_old.clone()
            r_old[CONV4_COND] = r[CONV4_COND]#.detach().clone()
            
            # broadcast 5
            r = r.clone()
            r_ = r[CONV4_COND] = s_ - (omega_.view(-1, 1, 1) * t_ )
            
            # 5R
            r_0_ = r_0[CONV4_COND]
            beta = beta.clone()
            beta[CONV4_COND] = (alpha[CONV4_COND] / omega_) \
                                * torch.sum(torch.mul(r_, r_0_), dim = (1, 2)) \
                                / torch.sum(torch.mul(r_old[CONV4_COND], r_0_), dim = (1, 2))
        
            if verbose:
                print(f"k : {k} , omega : {omega}, beta : {beta}")
            
            # broadcast 7
            p = p.clone()
            p[CONV4_COND] = r_ + beta[CONV4_COND].view(-1, 1, 1) * ( p_ - omega_.view(-1, 1, 1) * v[CONV4_COND])
            

            # =======================================
            # NOT REQUIRING RESTART SYSTEMS ; UPDATE
            # =======================================

            k[NOT_RES_COND] += 1 
            r_abs = r_abs.clone()
            r_abs[NOT_RES_COND] = torch.norm(r[NOT_RES_COND], dim = (1, 2), p = 'fro') #6R

            # =======================================
            # CONDITION TO SKIP SOLVING A SYTEM ; USEFUL FOR TRAINING with a NEURAL MODEL
            # =======================================
            
            # r_abs_diff_init  =  r_abs_init - r_abs
            # r_abs_diff_last  =  r_abs_last - r_abs
            # r_abs_last  = r_abs.detach().clone()
            
            # STAG_COND = torch.abs(r_abs_diff_last) == 0.
            # stagnant_count[STAG_COND] += 1.

            # check_iter = 200.
            # r_abs_diff_skip = torch.abs(torch.log10(r_abs_skip) - torch.log10(r_abs))

            # if (k % check_iter == 0).any() : 
            #     r_abs_skip  = r_abs.detach().clone()

            # # rabs have blown above 1e10 or
            # # is nan or 
            # # has stagnated ( 0. ) above [#] iter or
            # # rabs change is not in the magnitude of log10 for [#] iter
            # BREAK_COND = (CONV_COND) & ((torch.isnan(r_abs)) | 
            #                             (r_abs_diff_init < -1e10) | 
            #                             (stagnant_count > check_iter) | 
            #                             ((k % check_iter == 0) & (r_abs_diff_skip < 1.))
            #                             )
            # k[BREAK_COND] += kmax

            if verbose:
                print(f"k : {k}, RESIDUAL : {r_abs}")
            
            # print(f"{torch.cat((k, r_abs, r_abs_diff_last, CONV_COND), dim = 1)}") 

        print(f"total stencil time : {ttt}")
        print(f"total sigma time   : {ttt1}")

        return x



    def applyStencilGS(self, inp, boo, bmo, bom, bop, bpo, verbose = False):
        """
        inp : (batch, channel, nx, ny)
        """
        pad_mirror = Pad(1, padding_mode = "symmetric")
        inp        = pad_mirror(inp[:, :, 1:self.nx+1, 1:self.ny+1])
        temp       = torch.zeros_like(inp, dtype = torch.float64)

        center     = boo * inp[:, :, 1:self.nx+1, 1:self.ny+1]
         
        left       = bmo * inp[:, :, :self.nx, 1:self.ny+1]
        
        down       = bom * inp[:, :, 1:self.nx+1, :self.ny]
        
        up         = bop * inp[:, :, 1:self.nx+1, 2:self.ny+2]
        
        right      = bpo * inp[:, :, 2:self.nx+2, 1:self.ny+1]
                
        # if verbose :
        #     self.analyseImage(temp, "X")
        temp = temp.clone()
        temp[:, :, 1:self.nx+1, 1:self.ny+1 ] = center + left + right + up + down

        return temp

    def zeroPadGS(self, x):
        t = torch.zeros_like(x, dtype = torch.float64, device = self.device)
        t = t.clone()
        t[:, :, 1:self.nx+1, 1 :self.ny+1] = x[ :, :, 1:self.nx+1, 1 :self.ny+1]
        return t

    def BiCGSTAB_GS(self, x, b, kmax=10000, eps=1e-9, verbose = False):
        
        restart = torch.ones( (self.batch, self.channel), dtype=torch.bool, device = self.device)
        k       = torch.zeros((self.batch, self.channel), dtype=torch.long, device = self.device)
        r_abs   = torch.zeros((self.batch, self.channel), dtype=torch.float64, device = self.device)
        v_abs   = torch.zeros((self.batch, self.channel), dtype=torch.float64, device = self.device)
        r0_abs  = torch.zeros((self.batch, self.channel), dtype=torch.float64, device = self.device)
        sigma   = torch.zeros((self.batch, self.channel), dtype=torch.float64, device = self.device)
        alpha   = torch.zeros((self.batch, self.channel), dtype=torch.float64, device = self.device)
        omega   = torch.zeros((self.batch, self.channel), dtype=torch.float64, device = self.device)
        beta    = torch.zeros((self.batch, self.channel), dtype=torch.float64, device = self.device)

        r_0     = torch.zeros_like(x, dtype = torch.float64)
        r       = torch.zeros_like(x, dtype = torch.float64)
        r_old   = torch.zeros_like(x, dtype = torch.float64)
        p       = torch.zeros_like(x, dtype = torch.float64)
        v       = torch.zeros_like(x, dtype = torch.float64)
        s       = torch.zeros_like(x, dtype = torch.float64)
        t       = torch.zeros_like(x, dtype = torch.float64)

        # variables to check info. to skip solving systems 
        r_abs_init = torch.zeros( (self.batch, self.channel), dtype = torch.float64, device = self.device)
        r_abs_last = torch.zeros( (self.batch, self.channel), dtype = torch.float64, device = self.device)
        r_abs_skip = torch.zeros( (self.batch, self.channel), dtype = torch.float64, device = self.device)
        stagnant_count = torch.zeros( (self.batch, self.channel), dtype = torch.float64, device = self.device)

        r_0 = self.applyStencilGS(x, self.boo, self.bmo, self.bom, self.bop, self.bpo) 
        p   = self.zeroPadGS(b - r_0)
        r_0 = r = p
        r0_abs = torch.norm(r_0, dim = (2, 3), p = "fro")
        r_abs  = r0_abs

        if verbose:
            print(f"r_abs : {r_abs}")

        while ( (k < kmax) & (r_abs > eps * self.nx * self.ny) ).any(): # and (restart == 0).any():

            # =======================================
            # WHILE CONVERGENCE CONDITION
            # =======================================
            
            CONV_COND = (k < kmax) & (r_abs > eps * self.nx * self.ny) # and (restart == 0)
            if verbose:
                print(f"WHILE CONVERGENCE CONDITION :\n {CONV_COND}")
            
            v = torch.where(CONV_COND, self.applyStencilGS(p, self.boo, self.bmo, self.bom, self.bop, self.bpo), v)
            sigma = torch.where(CONV_COND, torch.sum(torch.mul(v, r_0), dim = (2, 3)), sigma)
            v_abs = torch.where(CONV_COND, torch.norm(v, dim = (2, 3),  p = "fro"), v_abs)
            
            if verbose:
                print(f"k : {k}, sigma : {sigma}, vabs : {v_abs}")
            
            # =======================================
            # SET RESTART CONDITION
            # =======================================

            RES_COND = (sigma <= eps * v_abs * r0_abs)
            RES1_COND = CONV_COND & RES_COND
            if verbose:
                print(f"RESTART REQUIRED :\n {RES1_COND}")

            p = torch.where(RES1_COND, self.zeroPadGS(b - self.applyStencilGS(x, self.boo, self.bmo, self.bom, self.bop, self.bpo)), p)
            # r_0 = r = p
            r = torch.where(RES1_COND, p, r)
            r_0 = torch.where(RES1_COND, p, r_0)
            r0_abs  = torch.where(RES1_COND, torch.norm(r_0, dim = (2, 3), p = "fro"), r0_abs)
            r_abs = torch.where(RES1_COND, r0_abs, r_abs)
            k = torch.where(RES1_COND, k+1, k)

            if verbose:
                print(f"r_abs when restarted: {r_abs}")

            # =======================================
            # INVERSE RESTART CONDITION : systems that dont require restart
            # =======================================
            NOT_RES_COND = CONV_COND & (~RES_COND)

            if verbose:
                print(f"RESTART NOT REQUIRED :\n {NOT_RES_COND}")

            alpha = torch.where(NOT_RES_COND, torch.sum( torch.mul(r, r_0), dim = (2, 3)).view(-1) / sigma , alpha)

            if verbose:
                print(f"k : {k}, alpha : {alpha}")

            s = torch.where(NOT_RES_COND, r - (alpha[:, :, None, None] * v), s)
            
            # =======================================
            # No RESTART and CONVERGENCE CONDITION 
            # =======================================

            CONV2_COND = torch.norm(s, dim = (2, 3), p = 'fro') <= eps * self.nx * self.ny
            CONV3_COND = NOT_RES_COND & CONV2_COND

            if verbose:
                print(f"RESTART NOT REQUIRED and CONV :\n {CONV3_COND}")

            x = torch.where(CONV3_COND, x + alpha[:, :, None, None] * p, x )
            r = torch.where(CONV3_COND, s, r)

            # =======================================
            # No RESTART and INVERSE CONVERGENCE CONDITION 
            # =======================================
            CONV4_COND = NOT_RES_COND & (~CONV2_COND)

            if verbose:
                print(f"RESTART NOT REQUIRED and ELSE CONV :\n {CONV4_COND}")

            t = torch.where(CONV4_COND, self.applyStencilGS(s, self.boo, self.bmo, self.bom, self.bop, self.bpo), t)
            omega  = torch.where(CONV4_COND, torch.sum( torch.mul(t, s), dim = (2, 3)) / torch.sum(t**2, dim = (2, 3)), omega )            
            x = torch.where(CONV4_COND, x + (alpha[:, :, None, None] * p) + (omega[:, :, None, None] * s), x)
            r_old = torch.where(CONV4_COND, r, r_old)
            r = torch.where(CONV4_COND, s - omega[:, :, None, None] * t, r)
            beta = torch.where(CONV4_COND
                            , (alpha / omega) * (torch.sum(torch.mul(r, r_0), dim = (2, 3)) / torch.sum(torch.mul(r_old, r_0), dim = (2, 3)))
                            , beta)
            
            # beta = (alpha / omega) * torch.sum(torch.mul(r, r_0), dim = (2, 3)) / torch.sum(torch.mul(r_old, r_0), dim = (2, 3))

            # beta[CONV4_COND] = (alpha[CONV4_COND] / omega[CONV4_COND]) \
            #         * torch.sum(torch.mul(r[CONV4_COND], r_0[CONV4_COND]), dim = (1, 2)) \
            #         / torch.sum(torch.mul(r_old[CONV4_COND], r_0[CONV4_COND]), dim = (1, 2))

            if verbose:
                print(f"k : {k} , omega : {omega}, beta : {beta}")

            p = torch.where(CONV4_COND, r + beta[:, :, None, None] * ( p - omega[:, :, None, None] * v), p)

            # =======================================
            # NOT REQUIRING RESTART SYSTEMS ; UPDATE
            # =======================================

            k  = torch.where(NOT_RES_COND, k+1, k) 
            r_abs = torch.where(NOT_RES_COND, torch.norm(r, dim = (2, 3), p = 'fro'), r_abs)

            if verbose:
                print(f"k : {k}, RESIDUAL : {r_abs}")

        return x
