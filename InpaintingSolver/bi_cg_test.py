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
        v_ = v[CONV_COND] = self.applyStencilBatch(p[CONV_COND], CONV_COND)
        sigma = sigma.clone()
        sigma[CONV_COND]  = torch.sum(torch.mul(v_, r_0[CONV_COND]), dim = (1, 2))
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
        p[RES1_COND]       = self.zeroPadBatch(b[RES1_COND] - self.applyStencilBatch(x[RES1_COND], RES1_COND))
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
        t_ = t[CONV4_COND] = self.applyStencilBatch(s_, CONV4_COND)
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

    return x

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