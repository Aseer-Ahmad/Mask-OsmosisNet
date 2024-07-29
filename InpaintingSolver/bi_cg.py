import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms import Pad
import torchvision


torch.set_printoptions(linewidth=2000)

class OsmosisInpainting:

    def __init__(self, U, V, mask, offset, tau, hx = 1, hy = 1):
        # (b, c, h, w)
        self.U       = U + offset  # original image
        self.V       = V + offset  # guidance image
        self.mask    = mask
        self.offset  = offset
        self.tau     = tau

        self.batch   = U.size(0) 
        self.channel = U.size(1) 
        self.nx      = U.size(2) 
        self.ny      = U.size(3) 
        
        # pixel sizes x and y direction
        self.hx      = hx
        self.hy      = hy      

    def solve(self, kmax = 1):

        X = self.U.detach().clone()

        for i in range(kmax):
            X, info = self.BiCGSTAB_batched(X)

    def calculateWeights(self):
        self.prepareInp()

        self.getDriftVectors()
        print(f"drift vectors calculated")

        self.getStencilMatrices()
        print(f"weight stencils calculated")


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
        self.V     = self.V.type(torch.float32)  

        # since we transposed 
        self.nx, self.ny = self.ny, self.nx

    def analyseImage(self, x, name):
        print(f"analyzing {name} : size : {x.size()}")
        print(f"min  : {torch.min(x)}")
        print(f"max  : {torch.max(x)}")
        print(f"mean : {torch.mean(x)}")
        print(f"std  : {torch.std(x)}")
        print()

    def getStencilMatrices(self, verbose = False):
        # self.boo = torch.zeros(self.batch, self.channel, self.nx+2, self.ny+2) # but weickert init. has ones
        # self.bop = torch.zeros(self.batch, self.channel, self.nx+2, self.ny+2) # neighbour entries for [i+1,j]
        # self.bpo = torch.zeros(self.batch, self.channel, self.nx+2, self.ny+2) # neighbour entries for [i,j+1]
        # self.bmo = torch.zeros(self.batch, self.channel, self.nx+2, self.ny+2) # neighbour entries for [i-1,j]
        # self.bom = torch.zeros(self.batch, self.channel, self.nx+2, self.ny+2) # neighbour entries for [i,j-1]

        self.boo  = torch.zeros_like(self.V)
        self.bop  = torch.zeros_like(self.V)
        self.bpo  = torch.zeros_like(self.V)
        self.bmo  = torch.zeros_like(self.V)
        self.bom  = torch.zeros_like(self.V)

        #time savers
        rx  = self.tau / (2.0 * self.hx)
        ry  = self.tau / (2.0 * self.hy)
        rxx = self.tau / (self.hx * self.hx)
        ryy = self.tau / (self.hy * self.hy)

        # x direction filter ; this is a backward difference kernel hence the extra 0 
        f1 = torch.tensor([1., -1., 0]).reshape(1, 1, 3, 1)
        
        # y direction filter ; this is a backward difference kernel hence the extra 0 
        f2 = torch.tensor([1., -1., 0]).reshape(1, 1, 1, 3)

        # osmosis weights 
        boo = 1. + 2. * (rxx + ryy) \
                - rx * F.conv2d(self.d1, f1, padding='same') \
                - ry * F.conv2d(self.d2, f2, padding='same')
        self.boo[:, :, 1:self.nx+1, 1:self.ny+1] = boo[:, :, 1:self.nx+1, 1:self.ny+1]

        # unclean indexing to avoid boundaries being affected
        self.bpo[:, :, 1:self.nx+1, 1:self.ny+1] = -rxx + rx * self.d1[:, :, 1:self.nx+1, 1:self.ny+1]
        self.bop[:, :, 1:self.nx+1, 1:self.ny+1] = -ryy + ry * self.d2[:, :, 1:self.nx+1, 1:self.ny+1]

        # boundaries require cleaning
        self.bmo[:, :, 1:, :] = -rxx - rx * self.d1[:, :, :self.nx+1, :]
        self.bom[:, :, :, 1:] = -ryy - ry * self.d2[:, :, :, :self.ny+1]
 
        if verbose :
            print(self.boo)
            print(self.bpo)
            print(self.bop)
            print(self.bmo)
            print(self.bom)


    def getDriftVectors(self, verbose = False):
        """
        # ∗ is convolution and .T is transpose
        # compute d1 = [-1/hx 1/hx] ∗ V / [.5 .5] ∗ v 
        # compute d2 = [-1/hy 1/hy].T ∗ V / [.5 .5].T ∗ v 
        """
        # self.d1 = torch.zeros(self.batch, self.channel, self.nx+2, self.ny+2)
        # self.d2 = torch.zeros(self.batch, self.channel, self.nx+2, self.ny+2)
        self.d1  = torch.zeros_like(self.V)
        self.d2  = torch.zeros_like(self.V)
                
        # row-direction filters  
        f1 = torch.tensor([-1./self.hx, 1./self.hx]).reshape(1, 1, 2, 1)
        f2 = torch.tensor([.5, .5]).reshape(1, 1, 2, 1)
        
        # col-direction filters
        f3 = torch.tensor([-1./self.hy, 1./self.hy]).reshape(1, 1, 1, 2)
        f4 = torch.tensor([.5, .5]).reshape(1, 1, 1, 2)

        d1 = F.conv2d(self.V, f1) / F.conv2d(self.V, f2)
        d2 = F.conv2d(self.V, f3) / F.conv2d(self.V, f4) 

        # correcting for dimentionality reduced by using F.conv2d
        # eg : 1 dimension reduced for d1  changes nx+2 -> nx+2-1
        self.d1[:, :, :self.nx+1, :] = d1  
        self.d2[:, :, :, :self.ny+1] = d2
        
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
        This input should be padded and trasnposed along with offset added to it.
        """
        r = torch.zeros_like(inp)

        center = torch.mul(self.boo[:, :, 1:self.nx+1, 1:self.ny+1],
                            inp[:, :, 1:self.nx+1, 1:self.ny+1])    
         
        left   = torch.mul(self.bmo[:, :, 1:self.nx+1, 1:self.ny+1],
                            inp[:, :, :self.nx, 1:self.ny+1])
        
        down = torch.mul(self.bom[:, :, 1:self.nx+1, 1:self.ny+1],
                            inp[:, :, 1:self.nx+1, 0:self.ny])
        
        up     = torch.mul(self.bop[:, :, 1:self.nx+1, 1:self.ny+1],
                            inp[:, :, 1:self.nx+1, 2:self.ny+2])
        
        right  = torch.mul(self.bpo[:, :, 1:self.nx+1, 1:self.ny+1],
                            inp[:, :, 2:self.nx+2, 1:self.ny+1])
        
        r[:, :, 1:self.nx+1, 1:self.ny+1 ] = center + left + right + up + down
        
        if verbose :
            self.analyseImage(r, "X")

        return r

    def BiCGSTAB_batched(self, B, X0=None, max_iter=10000, tol=1e-9):
        """
        Solve AX = B using BiCGStab method without preconditioning for batched, multi-channel inputs.
        
        Args:
        A: coefficient tensor of shape (batch, channel, nx, ny)
        B: right-hand side tensor of shape (batch, channel, ny, m)
        X0: initial guess tensor (if None, use zeros)
        max_iter: maximum number of iterations
        tol: tolerance for convergence
        
        Returns:
        X: solution tensor of shape (batch, channel, nx, m)
        info: convergence information (0: converged, 1: not converged)
        """

        print(f"\nstarting BiCGStab")
        
        B = B.to(torch.float64)
        batch, channel, nx, ny = B.shape
        
        if X0 is None:
            X = torch.zeros_like(B) # torch.zeros(batch, channel, nx, ny, dtype=torch.float64)
        else:
            X = X0.clone().to(torch.float64)
        
        R = B - self.applyStencil(X) # torch.matmul(A, X)
        R_tilde = R.clone()
        rho = alpha = omega = torch.ones(batch, channel, 1, 1, dtype=torch.float64)
        V = P = torch.zeros_like(X)
        
        for i in range(max_iter):
            rho_new = torch.sum(R_tilde * R, dim=(-2, -1), keepdim=True)
            beta = (rho_new / rho) * (alpha / omega)
            P = R + beta * (P - omega * V)
            V = self.applyStencil(P) # torch.matmul(A, P)
            alpha = rho_new / torch.sum(R_tilde * V, dim=(-2, -1), keepdim=True)
            H = X + alpha * P
            S = R - alpha * V
            T = self.applyStencil(S) # torch.matmul(A, S)
            omega = torch.sum(T * S, dim=(-2, -1), keepdim=True) / torch.sum(T * T, dim=(-2, -1), keepdim=True)
            X = H + omega * S
            R = S - omega * T
            
            rho = rho_new
            
            residual_norm = torch.norm(R.reshape(batch, channel, -1), dim=-1)

            print(f"Iteration {i+1} , residual norm : {residual_norm}")

            if torch.all(residual_norm < tol):
                return X, 0
        
        return X, 1


    def BiCGSTAB(self, x, b, kmax, eps=1e-9):
        """
        solving system Ax=b
        x : old and new solution ; torch.Tensor batch*channel*nx*ny
        b : right hand side      ; torch.Tensor batch*channel*nx*ny
        """
        restart = 1       
        k = 0 

        while restart == 1:
            
            restart = 0
            
            r_0 = r = p  = b - self.applyStencil(x)
            r_abs = r0_abs = torch.norm(r_0, p = 'fro')

            while k < kmax and  \
                    r_abs > eps * self.nx * self.ny and \
                    restart == 0:
                
                v = self.applyStencil(p)
                sigma = torch.sum((torch.mul(v, r_0)))

                v_abs = torch.norm(v, p = 'fro')

                if sigma <= eps * v_abs * r0_abs:

                    restart = 1
                    print(f"restarting ... k : {k} , sigma : {sigma} , vabs : {v_abs}")

                else :

                    alpha = torch.sum((torch.mul(r, r_0))) / sigma
                    s     = r - alpha * v

                    if torch.norm(s, p = 'fro') <= eps * self.nx * self.ny:

                        x = x + alpha * p 
                        r = s.detach().clone()
                    
                    else :

                        t = self.applyStencil(s)
                        omega = torch.sum((torch.mul(t, s))) / torch.sum((torch.mul(t, t))) 
                        x = x + alpha * p + omega * s
                        r_old = r.detach().clone()
                        r = s - omega * t 
                        beta = (alpha / omega) * torch.sum((torch.mul(r, r_0))) / torch.sum((torch.mul(r_old, r_0))) 
                        p = r + beta * (p - omega * v)

                    k += 1
                    r_abs = torch.norm(r, p = 'fro')
                    print(f"iteration : {k} , residual : {r_abs}")

        return x
