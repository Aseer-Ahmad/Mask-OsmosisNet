import torch 
from scipy.sparse.linalg import bicg

def BiCGSTAB(A, x, b, kmax, eps=1e-9):
    """
    solving system Ax=b
    x : old and new solution ; torch.Tensor batch*channel*nx*ny
    b : right hand side      ; torch.Tensor batch*channel*nx*ny
    """
    restart = 1       
    k = 0 

    nx, ny = A.size()

    while restart == 1:
        
        restart = 0
        
        r_0 = r = p  = b - torch.matmul(A, x)
        r_abs = r0_abs = torch.norm(r_0, p = 'fro')

        while k < kmax and  \
                r_abs > eps * nx * ny and \
                restart == 0:
            
            v = torch.matmul(A, p)
            sigma = torch.sum((torch.mul(v, r_0)))
            v_abs = torch.norm(v, p = 'fro')

            if sigma <= eps * v_abs * r0_abs:

                restart = 1
                print(f"restarting ... k : {k} , sigma : {sigma} , vabs : {v_abs}")

            else :

                alpha = torch.sum((torch.mul(r, r_0))) / sigma
                s     = r - alpha * v

                if torch.norm(s, p = 'fro') <= eps * nx * ny:

                    x = x + alpha * p 
                    r = s.detach().clone()
                
                else :

                    t = torch.matmul(A, s)
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

def BiCGSTAB_batched(A, B, X0=None, max_iter=1000, tol=1e-9):
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
    A = A.to(torch.float64)
    B = B.to(torch.float64)
    batch, channel, nx, ny = B.shape
    
    if X0 is None:
        X = torch.zeros(batch, channel, nx, ny, dtype=torch.float64, device=A.device)
    else:
        X = X0.clone().to(torch.float64)
    
    R = B - torch.matmul(A, X)
    R_tilde = R.clone()
    
    rho = alpha = omega = torch.ones(batch, channel, 1, 1, dtype=torch.float64, device=A.device)
    V = P = torch.zeros_like(X)
    
    for i in range(max_iter):
        rho_new = torch.sum(R_tilde * R, dim=(-2, -1), keepdim=True)
        beta = (rho_new / rho) * (alpha / omega)
        P = R + beta * (P - omega * V)
        V = torch.matmul(A, P)
        alpha = rho_new / torch.sum(R_tilde * V, dim=(-2, -1), keepdim=True)
        H = X + alpha * P
        S = R - alpha * V
        T = torch.matmul(A, S)
        omega = torch.sum(T * S, dim=(-2, -1), keepdim=True) / torch.sum(T * T, dim=(-2, -1), keepdim=True)
        X = H + omega * S
        R = S - omega * T
        
        rho = rho_new
        
        residual_norm = torch.norm(R.reshape(batch, channel, -1), dim=-1)
        print(f"iteration : {i+1} , residual norm : {residual_norm}")
        if torch.all(residual_norm < tol):
            return X, 0
    
    return X, 1

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


    # A = torch.tensor([[3., 2., -1.], [2., -2., 4.], [-1.,.5,-1.]])
    # B = torch.tensor([1., -2., 0.]).reshape(3, 1)

    A = torch.randn((50, 50), dtype = torch.float64)
    B = torch.randn((50, 50), dtype = torch.float64)
    X = B.detach().clone()

    print(A.size(), B.size())
    BiCGSTAB(A, X, B, 1000)

