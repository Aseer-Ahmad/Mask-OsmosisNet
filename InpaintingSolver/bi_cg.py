import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms import Pad
import torchvision

class OsmosisInpainting:

    def initialize(self, U, V, mask, offset, hx = 1, hy = 1):
        # (b, c, h, w)
        self.U       = U  # original image
        self.V       = V  # guidance image
        self.mask    = mask
        self.offset  = offset

        self.batch   = U.size(0) 
        self.channel = U.size(1) 
        self.nx      = U.size(2) #rows
        self.ny      = U.size(3) #column
        
        # pixel sizes x and y direction
        self.hx      = hx
        self.hy      = hy

        # stencil matrix
        self.boo     = None # center pixel entries [i,j]
        self.bop     = None # neighbour entries for [i+1,j]
        self.bpo     = None # neighbour entries for [i,j+1]
        self.bmo     = None # neighbour entries for [i-1,j]
        self.bom     = None # neighbour entries for [i,j-1]

        # drift vectors
        self.d1      = None
        self.d2      = None
        
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
        

    def analyseImage(self, x, name):
        print(f"analyzing {name} : size : {x.size()}")
        print(f"min  : {torch.min(x)}")
        print(f"max  : {torch.max(x)}")
        print(f"mean : {torch.mean(x)}")
        print(f"std  : {torch.std(x)}")
        print()

    def getStencilMatrices(self, tau):
        self.boo = torch.ones(self.batch, self.channel, self.nx+2, self.ny+2)
        self.bop = torch.zeros(self.batch, self.channel, self.nx+2, self.ny+2)
        self.bpo = torch.zeros(self.batch, self.channel, self.nx+2, self.ny+2)
        self.bmo = torch.zeros(self.batch, self.channel, self.nx+2, self.ny+2)
        self.bom = torch.zeros(self.batch, self.channel, self.nx+2, self.ny+2)

        #time savers
        rx  = tau / (2.0 * self.hx)
        ry  = tau / (2.0 * self.hy)
        rxx = tau / (self.hx * self.hx)
        ryy = tau / (self.hy * self.hy)

        # x direction filter
        f1 = torch.tensor([-1., 1.]).reshape(1, 1, 1, 2)
        
        # y direction filter
        f2 = torch.tensor([-1., 1.]).reshape(1, 1, 2, 1)

        # osmosis weights 
        self.boo = 1. + 2. * (rxx + ryy) - rx * F.conv2d(self.d1, f1) - ry * F.conv2d(self.d2, f2)
        
        self.bpo = -rxx + rx * self.d1
        self.bop = -ryy + ry * self.d2

        self.bmo[:, :, 1:, 1:] = -rxx + ry * self.d1[:, :, :self.nx+1, :self.ny+1]
        self.bom[:, :, 1:, 1:] = -ryy + ry * self.d2[:, :, :self.nx+1, :self.ny+1]
 

    def getDriftVectors(self, verbose = False):
        """
        # ∗ is convolution and .T is transpose
        # compute d1 = [-1/hx 1/hx] ∗ V / [.5 .5] ∗ v 
        # compute d2 = [-1/hy 1/hy].T ∗ V / [.5 .5].T ∗ v 
        """
        if self.nx is None or self.ny is None:
            self.nx      = self.V.size(2)
            self.ny      = self.V.size(3)
            self.channel = self.V.size(1)
            self.batch   = self.V.size(0)
            self.hx      = 1
            self.hy      = 1

        self.d1 = torch.zeros(self.batch, self.channel, self.nx+2, self.ny+2)
        self.d2 = torch.zeros(self.batch, self.channel, self.nx+2, self.ny+2)

        pad_mirror = Pad(1, padding_mode = "symmetric")
        V_padded   = pad_mirror(self.V)
        # transposed because Weickert transposed it in his code
        V_padded   = torch.transpose(V_padded, 2, 3)
        # because F.conv2d accepts only float32 
        V_padded   = V_padded.type(torch.float32)  
        
        # x-direction filters
        f1 = torch.tensor([-1./self.hx, 1./self.hx]).reshape(1, 1, 2, 1)
        f2 = torch.tensor([.5, .5]).reshape(1, 1, 2, 1)
        
        # y-direction filters
        f3 = torch.tensor([-1./self.hy, 1./self.hy]).reshape(1, 1, 1, 2)
        f4 = torch.tensor([.5, .5]).reshape(1, 1, 1, 2)

        d1 = F.conv2d(V_padded, f1) / F.conv2d(V_padded, f2)
        d2 = F.conv2d(V_padded, f3) / F.conv2d(V_padded, f4) 
        
        # correcting for dimentionality reduced by using F.conv2d
        # eg : 1 dimension reduced for d1  changes nx+2 -> nx+2-1
        self.d1[:, :, :self.nx+1, :] = d1  
        self.d2[:, :, :, :self.ny+1] = d2
        
        if verbose:
            print(f"V shape : {self.V.size()}, V padded shape : {V_padded.size()}")
            print(f"V_padded : \n{V_padded}\n")
            self.analyseImage(self.d1, "d1")
            self.analyseImage(self.d2, "d2")
            

    def applyStencil(self, inp):
        """
        inp : (batch, channel, nx, ny)
        """
    
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
        
        inp    = center + left + right + up + down
        
        return inp
    
    def Osmosis(self, kmax, tau, offset, eps=1e-9):
        f = self.U.copy() 
        self.BiCGSTAB(kmax, x, b, kmax, eps)


    def BiCGSTAB_batched(self, B, X0=None, max_iter=1000, tol=1e-6):
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
        # A = A.to(torch.float64)
        B = B.to(torch.float64)
        batch, channel, nx, ny = B.shape
        
        if X0 is None:
            X = torch.zeros(batch, channel, nx, ny, dtype=torch.float64)
        else:
            X = X0.clone().to(torch.float64)
        
        R = B - self.applyStencil(X)#torch.matmul(A, X)
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

            if torch.all(residual_norm < tol):
                return X, 0
        
        return X, 1


    def BiCGSTAB(self, x, b, kmax, eps):
        """
        solving system Ax=b
        x : old and new solution ; torch.Tensor batch*channel*nx*ny
        b : right hand side      ; torch.Tensor batch*channel*nx*ny
        """
        restart = torch.ones((self.batch, self.channel, 1, 1))        
        k = 0 

        while restart == 1:
          
            restart = 0
            
            r_0 = r = p  = torch.sub(b, self.applyStencil(x))
            r_abs = r0_abs = torch.norm(r_0, p = 'fro')

            while k < kmax and  \
                  r_abs > eps * self.nx * self.ny and \
                  restart == 0:
                
                v = self.applyStencil(p)
                sigma = torch.dot(v, r_0)
                v_abs = torch.norm(v, p = 'fro')

                if sigma <= eps * v_abs * r0_abs:

                    restart = 1
                
                else :

                    alpha = torch.div(torch.dot(r, r_0), sigma)
                    s     = torch.sub(r, torch.mul(v, alpha))
                    
                    if torch.norm(s, p = 'fro') <= eps * self.nx * self.ny:

                        x = torch.add(x, torch.mul(v, alpha))
                        r = s
                    
                    else :

                        t = self.applyStencil(s)
                        omega = torch.div(torch.dot(t, s), torch.norm(t, p = 'fro'))
                        x = torch.add(torch.add(x, torch.mul(p, alpha)), torch.mul(s, omega))
                        r_old = r
                        r = torch.sub(s, torch.mul(t, omega))
                        beta = (alpha/omega) * (torch.dot(r, r_0)/torch.dot(r_old, r_0))
                        p = torch.add(torch.mul(torch.sub(p, torch.mul(v, omega)), beta), r)

                    k += 1
                    r_abs = torch.norm(r, p = 'fro')

                    
