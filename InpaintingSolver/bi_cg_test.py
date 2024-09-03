import torch 
from scipy.sparse.linalg import bicg

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

    return x, restart, k


if __name__ == '__main__':

    # A = torch.tensor([[[3., 2., -1.], [2., -2., 4.], [-1.,.5,-1.]], 
    #                  [[1., 3., -1.], [5., -2., 5.], [1.,.3,-4.]]]).reshape(1, 2, 3, 3)
    # B = torch.tensor([[[1.,1.],
    #                   [-2.,-2.],
    #                   [0.,-0.]
    #                   ],
    #                   [[2.,0.],
    #                   [-1.,3.],
    #                   [1.,1.]
    #                   ]]).reshape(1, 2, 3, 2)

    # A = torch.randn((1, 1, 600, 600), dtype = torch.float64) 
    # B = torch.randn((1, 1, 600, 600), dtype = torch.float64) 

    # X = BiCGSTAB_batched1(A, B)


    A = torch.tensor([[3., 2., -1.], [2., -2., 4.], [-1.,.5,-1.]])
    B = torch.tensor([1., -2., 0.]).reshape(3, 1)

    # A = torch.randn((600, 600), dtype = torch.float64)
    # B = torch.randn((600, 600), dtype = torch.float64)
    X = B.detach().clone()

    print(A.size(), B.size())
    BiCGSTAB(A, X, B, 1000)