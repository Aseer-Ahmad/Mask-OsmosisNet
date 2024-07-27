import torch
import torch.nn.functional as F
from torchvision.transforms import Pad

class OsmosisInpainting():
    def __init__(self, U, V, mask, hx, hy):
        # (b, c, h, w)
        self.U       = U  # original image
        self.V       = V  # guidance image
        self.mask    = mask
        
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
        
    def analyseImage(self):
        pass

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
 

    def getDriftVectors(self):
        self.d1 = torch.zeros(self.batch, self.channel, self.nx+2, self.ny+2)
        self.d2 = torch.zeros(self.batch, self.channel, self.nx+2, self.ny+2)

        pad_mirror = Pad(1, padding_mode = "symmetric")
        V_padded   = pad_mirror(self.V)

        # * is convolution and .T is transpose
        # compute d1 = [-1/hx 1/hx] * V / [.5 .5] * v 
        # compute d2 = [-1/hy 1/hy].T * V / [.5 .5].T * v 

        # x-direction filters
        f1 = torch.tensor([-1./self.hx, 1./self.hx]).reshape(1, 1, 1, 2)
        f2 = torch.tensor([.5, .5]).reshape(1, 1, 1, 2)
        
        # y-direction filters
        f3 = torch.tensor([-1./self.hy, 1./self.hy]).reshape(1, 1, 2, 1)
        f4 = torch.tensor([.5, .5]).reshape(1, 1, 2, 1)

        self.d1 = torch.div(F.conv2d(V_padded, f1), F.conv2d(V_padded, f2)) 
        self.d2 = torch.div(F.conv2d(V_padded, f3), F.conv2d(V_padded, f4)) 

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

                    
