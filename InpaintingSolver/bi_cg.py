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
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.regression import MeanSquaredError
import torch.nn as nn

torch.set_printoptions(linewidth=2000)

class MSELoss(nn.Module):
    """
    Means squared loss : 
    """
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, U, V):
        nxny = U.shape[2] * U.shape[3]
        return torch.mean(torch.norm(U-V, p = 2, dim = (2,3))**2 / nxny)

def get_variable_memory_usage():
    variables = gc.get_objects()  # Get all objects tracked by the garbage collector
    for var in variables:
        try:
            var_size = sys.getsizeof(var) / (1024 * 1024)  # Get the size of the variable
            var_name = [k for k, v in locals().items() if id(v) == id(var)]
            # if var_name:  # If the variable name exists
            #     print(f"{var_name[0]}: {var_size} bytes")
        except:
            # Ignore objects that cannot be sized or traced back to a name
            pass
        
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

        
    def solve(self, kmax = 2, save_every = 10, verbose = False):
        
        self.save_every = save_every

        X = self.U.detach().clone()
        print(self.analyseImage(X, f"Initial img"))
        print()
        tt = 0

        for i in range(kmax):
            print(f"ITERATION : {i+1}")

            st = time.time()
            X = self.BiCGSTAB(x = self.U, b = X, batch = 0, kmax = 10000, eps = 1e-9, verbose=verbose)
            et = time.time()
            tt += (et-st)
            self.U = X.detach().clone()
            
            comm = self.analyseImage(self.U, f"evolving U at iter {i}")
            comm += f"time for iteration : {str(et-st)} sec\n"
            comm += f"total time         : {str(tt)} sec\n"
            #calculate metrics
            comm += self.getMetrics()
            print(comm)

            if i % self.save_every == 0:
                self.U = self.U - self.offset
                
                fname = f"kani_{str(i+1)}.pgm"
                self.writePGMImage(self.U[0][0].numpy().T, fname)
                # self.writeToPGM(fname = fname, t = self.U[0][0].T, comments= comm)
                self.U = self.U + self.offset
                

    def solveBatch(self, kmax = 10000, save_batch = False, verbose = False):

        tt = 0
        mse = MSELoss()

        total_loss = 0.

        st = time.time()

        for batch in range(self.batch):

            V = self.V[batch].unsqueeze(0)
            B = self.U[batch].unsqueeze(0)
            U = B      #.detach().clone().to(self.device)
            # init = B.detach().clone().to(self.device)
            
            if verbose:
                print(f"batch item : {batch+1} / {self.batch}")    

            for i in range(kmax):
                B = self.BiCGSTAB(x = U, b = B, batch = batch, kmax = 10000, eps = 1e-9, verbose=verbose)
                U = B     #.detach().clone()
                loss = mse( self.normalize(U), self.normalize(V))
                print(f"\rITERATION : {i+1}, loss : {loss.item()} ", end ='', flush=True)        
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
        # print(f"drift vectors calculated")

        self.applyMask(m_verbose)
        # print(f"mask applied to drift vectors")

        self.getStencilMatrices(s_verbose)
        # print(f"stencils weights calculated")
        
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
        # V is casted float32 since conv2d only accepts that
        # self.V     = self.V.type(torch.float32)  

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
        
    def getStencilMatrices(self, verbose = False):

        self.boo  = torch.zeros_like(self.V, device = self.device)# C++ weickert init. has ones
        self.bop  = torch.zeros_like(self.V, device = self.device)# neighbour entries for [i+1,j]
        self.bpo  = torch.zeros_like(self.V, device = self.device)# neighbour entries for [i,j+1]
        self.bmo  = torch.zeros_like(self.V, device = self.device)# neighbour entries for [i-1,j]
        self.bom  = torch.zeros_like(self.V, device = self.device)# neighbour entries for [i,j-1]

        #time savers
        rx  = self.tau / (2.0 * self.hx)
        ry  = self.tau / (2.0 * self.hy)
        rxx = self.tau / (self.hx * self.hx)
        ryy = self.tau / (self.hy * self.hy)

        # x direction filter ; this is a backward difference kernel hence the extra 0 
        f1 = torch.tensor([1., -1., 0.], dtype = torch.float64, device = self.device).reshape(1, 1, 3, 1)
        
        # y direction filter ; this is a backward difference kernel hence the extra 0 
        f2 = torch.tensor([1., -1., 0.], dtype = torch.float64, device = self.device).reshape(1, 1, 1, 3)

        # osmosis weights 
        boo = 1. + 2. * (rxx + ryy) \
                - rx * F.conv2d(self.d1, f1, padding='same') \
                - ry * F.conv2d(self.d2, f2, padding='same')
        self.boo[:, :, 1:self.nx+1, 1:self.ny+1] = boo[:, :, 1:self.nx+1, 1:self.ny+1]

        # indexing to avoid boundaries being affected
        self.bpo[:, :, 1:self.nx+1, 1:self.ny+1] = -rxx + rx * self.d1[:, :, 1:self.nx+1, 1:self.ny+1]
        self.bop[:, :, 1:self.nx+1, 1:self.ny+1] = -ryy + ry * self.d2[:, :, 1:self.nx+1, 1:self.ny+1]

        self.bmo[:, :, 1:self.nx+1, 1:self.ny+1] = -rxx - rx * self.d1[:, :, :self.nx, 1:self.ny+1]
        self.bom[:, :, 1:self.nx+1, 1:self.ny+1] = -ryy - ry * self.d2[:, :, 1:self.nx+1, :self.ny]
 
        if verbose :
            print(self.boo)
            self.analyseImage(self.boo, "boo")
            print(self.bpo)
            self.analyseImage(self.bpo, "bpo")
            print(self.bop)
            self.analyseImage(self.bop, "bop")
            print(self.bmo)
            self.analyseImage(self.bmo, "bmo")
            print(self.bom)
            self.analyseImage(self.bom, "bom")

    def getMetrics(self):
        metrics = ""
        psnr  = PeakSignalNoiseRatio()
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
    
    def createMaskfromCanny(self):
        img = self.V.numpy().astype(np.uint8)
        print(img)
        edge = cv2.Canny(img, 100, 150 )
        print(edge) 

    def hardRoundBinarize(self):
        if self.mask1 != None and self.mask2 != None:
            self.mask1 =  torch.floor(self.mask1 + 0.5)
            self.mask2 =  torch.floor(self.mask2 + 0.5)
            
    def applyMask(self , verbose = False):
        
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
        f1 = torch.tensor([-1./self.hx, 1./self.hx], dtype = torch.float64, device = self.device).reshape(1, 1, 2, 1)
        f2 = torch.tensor([.5, .5], dtype = torch.float64, device = self.device).reshape(1, 1, 2, 1)
        
        # col-direction filters
        f3 = torch.tensor([-1./self.hy, 1./self.hy], dtype = torch.float64, device = self.device).reshape(1, 1, 1, 2)
        f4 = torch.tensor([.5, .5], dtype = torch.float64, device = self.device).reshape(1, 1, 1, 2)

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
            self.analyseImage(self.d1, "d1")
            self.analyseImage(self.d2, "d2")
            
    def applyStencil(self, inp, batch, verbose = False):
        """
        inp : (batch, channel, nx, ny)
        """
        pad_mirror = Pad(1, padding_mode = "symmetric")
        inp        = pad_mirror(inp[:, :, 1:self.nx+1, 1:self.ny+1])

        temp       = torch.zeros_like(inp, device = self.device)

        center     = torch.mul(self.boo[batch, :, 1:self.nx+1, 1:self.ny+1],
                            inp[:, :, 1:self.nx+1, 1:self.ny+1])    
         
        left       = torch.mul(self.bmo[batch, :, 1:self.nx+1, 1:self.ny+1],
                            inp[:, :, :self.nx, 1:self.ny+1])
        
        down       = torch.mul(self.bom[batch, :, 1:self.nx+1, 1:self.ny+1],
                            inp[:, :, 1:self.nx+1, 0:self.ny])
        
        up         = torch.mul(self.bop[batch, :, 1:self.nx+1, 1:self.ny+1],
                            inp[:, :, 1:self.nx+1, 2:self.ny+2])
        
        right      = torch.mul(self.bpo[batch, :, 1:self.nx+1, 1:self.ny+1],
                            inp[:, :, 2:self.nx+2, 1:self.ny+1])
        
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
            r_abs = r0_abs = torch.norm(r_0[:, :, 1:self.nx+1, 1:self.ny+1], p = 'fro') # avoid boundary calculations
            
            if verbose:
                print(f"k : {k} , r_abs : {r_abs}")

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
                    if verbose:
                        print(f"restarting ... k : {k} , sigma : {sigma} , vabs : {v_abs}")

                else :

                    # r, s contains boundaries
                    alpha = torch.sum( torch.mul(r[:, :, 1:self.nx+1, 1:self.ny+1], r_0[:, :, 1:self.nx+1, 1:self.ny+1])) / sigma
                    s     = r - alpha * v
                    
                    if verbose:
                        print(f"k : {k} , alpha : {alpha}")

                    if torch.norm(s[:, :, 1:self.nx+1, 1:self.ny+1], p = 'fro') <= eps * self.nx * self.ny:
                        
                        # x, r contains boundaries
                        x = x + alpha * p 
                        r = s
                    
                    else :

                        t = self.applyStencil(s, batch)
                        omega = torch.sum( torch.mul(t, s)) / torch.sum(torch.mul(t, t))
                                        
                        x = x + alpha * p + omega * s
                        r_old = r
                        r = s - omega * t 
                        beta = (alpha / omega) * torch.sum(torch.mul(r[:, :, 1:self.nx+1, 1:self.ny+1], r_0[:, :, 1:self.nx+1, 1:self.ny+1])) / torch.sum(torch.mul(r_old[:, :, 1:self.nx+1, 1:self.ny+1], r_0[:, :, 1:self.nx+1, 1:self.ny+1]))
                    
                        if verbose:
                            print(f"k : {k} , omega : {omega}, beta : {beta}")

                        p = r + beta * (p - omega * v)

                    k += 1
                    r_abs = torch.norm(r[:, :, 1:self.nx+1, 1:self.ny+1], p = 'fro')
                    
                    if verbose:
                        print(f"k : {k} , RESIDUAL : {r_abs}")

        return x

    def applyStencilBatch(self, inp, COND, verbose = False):
        """
        inp : (batch, channel, nx, ny)
        """
        pad_mirror = Pad(1, padding_mode = "symmetric")
        inp        = pad_mirror(inp[ :, 1:self.nx+1, 1:self.ny+1])

        temp       = torch.zeros_like(inp)

        center     = torch.mul(self.boo[COND][ :, 1:self.nx+1, 1:self.ny+1],
                            inp[ :, 1:self.nx+1, 1:self.ny+1])    
         
        left       = torch.mul(self.bmo[COND][ :, 1:self.nx+1, 1:self.ny+1],
                            inp[ :, :self.nx, 1:self.ny+1])
        
        down       = torch.mul(self.bom[COND][ :, 1:self.nx+1, 1:self.ny+1],
                            inp[ :, 1:self.nx+1, 0:self.ny])
        
        up         = torch.mul(self.bop[COND][ :, 1:self.nx+1, 1:self.ny+1],
                            inp[ :, 1:self.nx+1, 2:self.ny+2])
        
        right      = torch.mul(self.bpo[COND][ :, 1:self.nx+1, 1:self.ny+1],
                            inp[ :, 2:self.nx+2, 1:self.ny+1])
        
        temp[ :, 1:self.nx+1, 1:self.ny+1 ] = center + left + right + up + down
        
        if verbose :
            self.analyseImage(temp, "X")

        return temp

    def BiCGSTAB_Batched(self, x, b, kmax=10000, eps=1e-9, verbose = False):
        
        restart = torch.ones( (self.batch, self.channel), dtype=torch.bool)
        k       = torch.zeros((self.batch, self.channel), dtype=torch.long)
        r_abs   = torch.zeros((self.batch, self.channel), dtype=torch.float64)
        v_abs   = torch.zeros((self.batch, self.channel), dtype=torch.float64)
        r0_abs  = torch.zeros((self.batch, self.channel), dtype=torch.float64)
        sigma   = torch.zeros((self.batch, self.channel), dtype=torch.float64)
        alpha   = torch.zeros((self.batch, self.channel), dtype=torch.float64)
        omega   = torch.zeros((self.batch, self.channel), dtype=torch.float64)
        beta    = torch.zeros((self.batch, self.channel), dtype=torch.float64)

        r_0     = torch.zeros_like(x)
        r       = torch.zeros_like(x)
        r_old   = torch.zeros_like(x)
        p       = torch.zeros_like(x)
        v       = torch.zeros_like(x)
        s       = torch.zeros_like(x)
        t       = torch.zeros_like(x)


        # check if any of restart is True or not
        while restart.any():    

            # set only those restart to zero
            RES_COND = restart == 1

            # using condition to select only those batch, channel that required restart
            restart[RES_COND] = 0
            r_0[RES_COND]     = self.applyStencilBatch(x[RES_COND], RES_COND)  
            r_0[RES_COND]     = r[RES_COND] = p[RES_COND] = b[RES_COND] - r_0[RES_COND]
            r_abs[RES_COND]   = r0_abs[RES_COND] = torch.norm(r_0[RES_COND][:, 1:self.nx+1, 1:self.ny+1], dim = (1, 2), p = "fro")

            print(f"r_abs : {r_abs}")
            #  check if any batch, channel system fails the convergence condition
            while (k < kmax).any() and (r_abs > eps * self.nx * self.ny).any() and (restart == 0).any():

                # =======================================
                # WHILE CONVERGENCE CONDITION
                # =======================================
                
                CONV_COND = (k < kmax) and (r_abs > eps * self.nx * self.ny) and (restart == 0)
                print(f"WHILE CONVERGENCE CONDITION : {CONV_COND}")

                v[CONV_COND] = self.applyStencilBatch(p[CONV_COND], CONV_COND)
                sigma[CONV_COND]  = torch.sum(torch.mul(v[CONV_COND], r_0[CONV_COND]), dim = (1, 2))
                v_abs[CONV_COND]  = torch.norm(v[CONV_COND], dim = (1, 2),  p = "fro")
              
                print(f"k : {k}, sigma : {sigma}, vabs : {v_abs}")
                # =======================================
                # RESTART CONDITION
                # =======================================

                RES1_COND = sigma <= eps * v_abs * r0_abs
                restart[RES1_COND] == 1

                print(f"RESTART CONDITION : {RES1_COND}")
                # =======================================
                # INVERSE RESTART CONDITION : systems that dont require restart
                # =======================================

                # broadcast 1
                # alpha[~RES1_COND] => shape : torch.Size([x])
                alpha[~RES1_COND] = torch.sum( torch.mul(r[~RES1_COND][ :, 1:self.nx+1, 1:self.ny+1],
                                                          r_0[~RES1_COND][ :, 1:self.nx+1, 1:self.ny+1]), dim = (1, 2)).view(-1) / sigma[~RES1_COND]
                # broadcast 2
                # s[~RES1_COND] => shape : torch.Size([b*c, h, w])
                s[~RES1_COND]     = r[~RES1_COND] - (alpha[~RES1_COND].view(-1, 1, 1) * v[~RES1_COND])
                
                # if verbose:
                #     print(f"k : {k} , alpha : {alpha}")

                # =======================================
                # No RESTART and CONVERGENCE CONDITION 
                # =======================================

                CONV2_COND = torch.norm(s[:, :, 1:self.nx+1, 1:self.ny+1], dim = (2, 3), p = 'fro') <= eps * self.nx * self.ny
                CONV3_COND = (~RES1_COND) & CONV2_COND

                # print(f"RES1_COND shape : {RES1_COND.shape}")
                # print(f"CONV2_COND shape : {CONV2_COND.shape}")
                # print(f"CONV3_COND shape : {CONV3_COND.shape}")

                # broadcast 3
                x[CONV3_COND] = x[CONV3_COND] + (alpha[CONV3_COND].view(-1, 1, 1) * p[CONV3_COND] )
                r[CONV3_COND] = s[CONV3_COND].detach().clone()
                    
                # =======================================
                # No RESTART and INVERSE CONVERGENCE CONDITION 
                # =======================================
                CONV4_COND = (~RES1_COND) & (~CONV2_COND)

                t[CONV4_COND] = self.applyStencilBatch(s[CONV4_COND], CONV4_COND)
                omega[CONV4_COND] = torch.sum( torch.mul(t[CONV4_COND], s[CONV4_COND]), dim = (1, 2)) / torch.sum(torch.mul(t[CONV4_COND], t[CONV4_COND]), dim = (1, 2))

                # broadcast 4                                
                x[CONV4_COND] = x[CONV4_COND] + (alpha[CONV4_COND].view(-1, 1, 1) * p[CONV4_COND]) + (omega[CONV4_COND].view(-1, 1, 1) * s[CONV4_COND])
                r_old[CONV4_COND] = r[CONV4_COND].detach().clone()
                
                # broadcast 5
                r[CONV4_COND] = s[CONV4_COND] - (omega[CONV4_COND].view(-1, 1, 1) * t[CONV4_COND] )
                
                beta[CONV4_COND] = (alpha[CONV4_COND] / omega[CONV4_COND]) \
                                    * torch.sum(torch.mul(r[CONV4_COND][ :, 1:self.nx+1, 1:self.ny+1], r_0[CONV4_COND][ :, 1:self.nx+1, 1:self.ny+1]), dim = (1, 2)) \
                                    / torch.sum(torch.mul(r_old[CONV4_COND][ :, 1:self.nx+1, 1:self.ny+1], r_0[CONV4_COND][ :, 1:self.nx+1, 1:self.ny+1]), dim = (1, 2))
            
                # if verbose:
                #     print(f"k : {k} , omega : {omega}, beta : {beta}")
                
                # broadcast 7
                p[CONV4_COND] = r[CONV4_COND] + (beta[CONV4_COND].view(-1, 1, 1) * p[CONV4_COND]) - (omega[CONV4_COND].view(-1, 1, 1) * v[CONV4_COND])
             
                k[~RES1_COND] += 1 
                r_abs[~RES1_COND] = torch.norm(r[~RES1_COND][ :, 1:self.nx+1, 1:self.ny+1], dim = (1, 2), p = 'fro')

                print(f"k : {k}, r_abs : {r_abs}")
