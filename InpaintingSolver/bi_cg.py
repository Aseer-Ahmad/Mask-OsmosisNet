import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms import Pad
import torchvision
import numpy as np
import time
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.regression import MeanSquaredError

torch.set_printoptions(linewidth=2000)

class OsmosisInpainting:

    def __init__(self, U, V, mask1, mask2, offset, tau, hx = 1, hy = 1, apply_canny = False):
        # (b, c, h, w)
        self.V       = V + offset  # guidance image
        if U is not None:
            self.U   = U + offset  # original image
        else:
            self.U   = self.getInit_U() + offset
        self.mask1   = mask1
        self.mask2   = mask2
        self.offset  = offset
        self.tau     = tau
        
        self.batch   = V.size(0) 
        self.channel = V.size(1) 
        self.nx      = V.size(2) 
        self.ny      = V.size(3) 
        
        # pixel sizes x and y direction
        self.hx      = hx
        self.hy      = hy    

        # img 
        self.save_every  = 10
        self.apply_canny = apply_canny

        self.device = None
        
    def solve(self, kmax = 2, save_every = 10, verbose = False):
        
        self.save_every = save_every

        X = self.U.detach().clone()
        print(self.analyseImage(X, f"Initial img"))
        print()
        tt = 0

        for i in range(kmax):
            print(f"ITERATION : {i+1}")

            st = time.time()
            X = self.BiCGSTAB(x = self.U, b = X, kmax = 10000, eps = 1e-9, verbose=verbose)
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
                
                fname = f"scarf_{str(i+1)}.pgm"
                self.writePGMImage(self.U[0][0].numpy().T, fname)
                # self.writeToPGM(fname = fname, t = self.U[0][0].T, comments= comm)

                self.U = self.U + self.offset
                

    def calculateWeights(self, d_verbose = False, m_verbose = False, s_verbose = False):
        self.prepareInp()

        self.getDriftVectors(d_verbose)
        print(f"drift vectors calculated")

        # self.applyMask(m_verbose)
        # print(f"mask applied to drift vectors")

        self.getStencilMatrices(s_verbose)
        print(f"stencils weights calculated")
        
        print()

    def normalize(self):
        pass

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
        print(f"written to : {filename}")

       
    def prepareInp(self):
        """
        transposed because Weickert transposed it in his C code
        V is casted float32 since conv2d only accepts that
        """
        pad_mirror = Pad(1, padding_mode = "symmetric")

        self.U     = pad_mirror(self.U)
        self.U     = torch.transpose(self.U, 2, 3)

        self.V     = pad_mirror(self.V)
        self.V     = torch.transpose(self.V, 2, 3)
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
        comm += f"min  : {str(torch.min(x).item())}\n"
        comm += f"max  : {str(torch.max(x).item())}\n"
        comm += f"mean : {str(torch.mean(x).item())}\n"
        comm += f"std  : {str(torch.std(x).item())}\n"

        return comm
        
    def getStencilMatrices(self, verbose = False):

        self.boo  = torch.zeros_like(self.V)# but weickert init. has ones
        self.bop  = torch.zeros_like(self.V)# neighbour entries for [i+1,j]
        self.bpo  = torch.zeros_like(self.V)# neighbour entries for [i,j+1]
        self.bmo  = torch.zeros_like(self.V)# neighbour entries for [i-1,j]
        self.bom  = torch.zeros_like(self.V)# neighbour entries for [i,j-1]

        #time savers
        rx  = self.tau / (2.0 * self.hx)
        ry  = self.tau / (2.0 * self.hy)
        rxx = self.tau / (self.hx * self.hx)
        ryy = self.tau / (self.hy * self.hy)

        # x direction filter ; this is a backward difference kernel hence the extra 0 
        f1 = torch.tensor([1., -1., 0.], dtype = torch.float64).reshape(1, 1, 3, 1)
        
        # y direction filter ; this is a backward difference kernel hence the extra 0 
        f2 = torch.tensor([1., -1., 0.], dtype = torch.float64).reshape(1, 1, 1, 3)

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
        m  = torch.mean(self.V)

        # create a flat image ; avg gray val same as guidance
        u  = torch.zeros_like(self.V) + m

        # create a noisy image ; avg gray val same as guidance
        return u
    

    def createMaskfromCanny(self):
        img = self.V.numpy().astype(np.uint8)
        print(img)
        edge = cv2.Canny(img, 100, 150 )
        print(edge) 

    def binarizeMask(self):
        if self.mask1 != None and self.mask2 != None:
            self.mask1 = (self.mask1 > 0).float()
            self.mask2 = (self.mask2 > 0).float()
            
    def applyMask(self , verbose = False):
        self.binarizeMask() 
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
        self.d1  = torch.zeros_like(self.V, dtype = torch.float64)
        self.d2  = torch.zeros_like(self.V, dtype = torch.float64)
                
        # row-direction filters  
        f1 = torch.tensor([-1./self.hx, 1./self.hx], dtype = torch.float64).reshape(1, 1, 2, 1)
        f2 = torch.tensor([.5, .5], dtype = torch.float64).reshape(1, 1, 2, 1)
        
        # col-direction filters
        f3 = torch.tensor([-1./self.hy, 1./self.hy], dtype = torch.float64).reshape(1, 1, 1, 2)
        f4 = torch.tensor([.5, .5], dtype = torch.float64).reshape(1, 1, 1, 2)

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
            

    def applyStencil(self, inp, verbose = False):
        """
        inp : (batch, channel, nx, ny)
        """
        pad_mirror = Pad(1, padding_mode = "symmetric")
        inp        = pad_mirror(inp[:, :, 1:self.nx+1, 1:self.ny+1])

        temp       = torch.zeros_like(inp)

        center     = torch.mul(self.boo[:, :, 1:self.nx+1, 1:self.ny+1],
                            inp[:, :, 1:self.nx+1, 1:self.ny+1])    
         
        left       = torch.mul(self.bmo[:, :, 1:self.nx+1, 1:self.ny+1],
                            inp[:, :, :self.nx, 1:self.ny+1])
        
        down       = torch.mul(self.bom[:, :, 1:self.nx+1, 1:self.ny+1],
                            inp[:, :, 1:self.nx+1, 0:self.ny])
        
        up         = torch.mul(self.bop[:, :, 1:self.nx+1, 1:self.ny+1],
                            inp[:, :, 1:self.nx+1, 2:self.ny+2])
        
        right      = torch.mul(self.bpo[:, :, 1:self.nx+1, 1:self.ny+1],
                            inp[:, :, 2:self.nx+2, 1:self.ny+1])
        
        temp[:, :, 1:self.nx+1, 1:self.ny+1 ] = center + left + right + up + down
        
        if verbose :
            self.analyseImage(temp, "X")

        return temp


    def zeroPad(self, x):
        t = torch.zeros_like(x)
        t[:, :, 1:self.nx+1, 1 :self.ny+1] = x[:, :, 1:self.nx+1, 1 :self.ny+1]
        return t

    def BiCGSTAB(self, x, b, kmax=10000, eps=1e-9, verbose = False):
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
            
            r_0 = self.applyStencil(x)  
            r_0 = r = p  = b - r_0  
            r_abs = r0_abs = torch.norm(r_0[:, :, 1:self.nx+1, 1:self.ny+1], p = 'fro') # avoid boundary calculations
            
            # print(f"k : {k} , r_abs : {r_abs}")

            while k < kmax and  \
                    r_abs > eps * self.nx * self.ny and \
                    restart == 0:
                
                v = self.applyStencil(p) # output contains zero boundaries
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
                        r = s.detach().clone()
                    
                    else :

                        t = self.applyStencil(s)
                        omega = torch.sum( torch.mul(t, s)) / torch.sum(torch.mul(t, t))
                                        
                        x = x + alpha * p + omega * s
                        r_old = r.detach().clone()
                        r = s - omega * t 
                        beta = (alpha / omega) * torch.sum(torch.mul(r[:, :, 1:self.nx+1, 1:self.ny+1], r_0[:, :, 1:self.nx+1, 1:self.ny+1])) / torch.sum(torch.mul(r_old[:, :, 1:self.nx+1, 1:self.ny+1], r_0[:, :, 1:self.nx+1, 1:self.ny+1]))
                    
                        if verbose:
                            print(f"k : {k} , omega : {omega}, beta : {beta}")

                        p = r + beta * (p - omega * v)

                    k += 1
                    r_abs = torch.norm(r[:, :, 1:self.nx+1, 1:self.ny+1], p = 'fro')
                    
                    if verbose:
                        print(f"k : {k} , RESIDUAL : {r_abs}")
        print()
        
        return x



    def BiCGSTAB_Batched(self, x, b, kmax=10000, eps=1e-9, verbose = False):
        
        restart = torch.ones((self.batch, self.channel),  dtype=torch.bool)
        k       = torch.zeros((self.batch, self.channel), dtype=torch.long)
        r_abs   = torch.zeros((self.batch, self.channel), dtype=torch.long)
        v_abs   = torch.zeros((self.batch, self.channel), dtype=torch.long)
        r0_abs  = torch.zeros((self.batch, self.channel), dtype=torch.long)
        sigma   = torch.zeros((self.batch, self.channel), dtype=torch.long)

        r_0     = torch.zeros_like(x)
        r       = torch.zeros_like(x)
        p       = torch.zeros_like(x)
        v       = torch.zeros_like(x)

        # check if any of restart is True or not
        while restart.any():    

            # set only those restart to zero
            RES_COND = restart == 1

            # using condition to select only those batch, channel that required restart
            restart[RES_COND] = 0
            r_0[RES_COND]     = self.applyStencil(x)  
            r_0[RES_COND]     = r[RES_COND] = p[RES_COND] = b[RES_COND] - r_0[RES_COND]
            r_abs[RES_COND]   = r0_abs[RES_COND] = torch.norm(r_0[RES_COND][:, 1:self.nx+1, 1:self.ny+1], p = "fro")

            #  check if any batch, channel system fails the convergence condition
            while (k < kmax).any() and (r_abs > eps * self.nx * self.ny).any() and (restart == 0).any():

                CONV_COND = (k < kmax) and (r_abs > eps * self.nx * self.ny) and (restart == 0)

                v[CONV_COND] = self.applyStencil(p)
                # sum, norm across only batch, channel
                sigma[CONV_COND]  = torch.sum(torch.mul(v[CONV_COND], r_0[CONV_COND]))
                v_abs[CONV_COND]  = torch.norm(v[CONV_COND], p = "fro")
              
                if 
             
             

    # def BiCGSTAB_batched(self, B, X0=None, max_iter=10000, tol=1e-9):
    #     """
    #     Solve AX = B using BiCGStab method without preconditioning for batched, multi-channel inputs.
        
    #     Args:
    #     A: coefficient tensor of shape (batch, channel, nx, ny)
    #     B: right-hand side tensor of shape (batch, channel, ny, m)
    #     X0: initial guess tensor (if None, use zeros)
    #     max_iter: maximum number of iterations
    #     tol: tolerance for convergence
        
    #     Returns:
    #     X: solution tensor of shape (batch, channel, nx, m)
    #     info: convergence information (0: converged, 1: not converged)
    #     """

    #     print(f"\nstarting BiCGStab")
        

    #     B = B.to(torch.float64)
    #     batch, channel, nx, ny = B.shape
        
    #     if X0 is None:
    #         X = torch.zeros_like(B) # torch.zeros(batch, channel, nx, ny, dtype=torch.float64)
    #     else:
    #         X = X0.clone().to(torch.float64)
        
    #     R = B - self.applyStencil(X) # torch.matmul(A, X)
    #     R_tilde = R.clone()
    #     rho = alpha = omega = torch.ones(batch, channel, 1, 1, dtype=torch.float64)
    #     V = P = torch.zeros_like(X)
        
    #     for i in range(max_iter):
    #         rho_new = torch.sum(R_tilde * R, dim=(-2, -1), keepdim=True)
    #         beta = (rho_new / rho) * (alpha / omega)
    #         P = R + beta * (P - omega * V)
    #         V = self.applyStencil(P) # torch.matmul(A, P)
    #         alpha = rho_new / torch.sum(R_tilde * V, dim=(-2, -1), keepdim=True)
    #         H = X + alpha * P
    #         S = R - alpha * V
    #         T = self.applyStencil(S) # torch.matmul(A, S)
    #         omega = torch.sum(T * S, dim=(-2, -1), keepdim=True) / torch.sum(T * T, dim=(-2, -1), keepdim=True)
    #         X = H + omega * S
    #         R = S - omega * T
            
    #         rho = rho_new
            
    #         residual_norm = torch.norm(R.reshape(batch, channel, -1), dim=-1)

    #         print(f"Iteration {i+1} , residual norm : {residual_norm}")

    #         if torch.all(residual_norm < tol):
    #             return X, 0
        
    #     return X, 1
