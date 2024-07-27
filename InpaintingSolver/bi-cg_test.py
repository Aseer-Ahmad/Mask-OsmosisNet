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
        
        # r0  = torch.matmul(A, x)
        r_0 = r = p  = b - torch.matmul(A, x)
        r_abs = r0_abs = torch.norm(r_0, p = 'fro')

        while k < kmax and  \
                r_abs > eps * nx * ny and \
                restart == 0:
            
            v = torch.matmul(A, p)
            sigma = torch.matmul(v.T, r_0)
            v_abs = torch.norm(v, p = 'fro')

            if sigma <= eps * v_abs * r0_abs:

                restart = 1
                print(f"restarting ... k : {k}")
            else :

                alpha = torch.div(torch.matmul(r.T, r_0), sigma)
                s     = torch.sub(r, torch.mul(v, alpha))
                if torch.norm(s, p = 'fro') <= eps * nx * ny:

                    x = torch.add(x, torch.mul(v, alpha))
                    r = torch.Tensor(s)
                
                else :

                    t = torch.matmul(A, s)
                    omega = torch.div(torch.matmul(t.T, s), torch.norm(t, p = 'fro')) #scalar
                    x = torch.add(torch.add(x, torch.mul(p, alpha)), torch.mul(s, omega))
                    r_old = torch.Tensor(r)
                    r = torch.sub(s, torch.mul(t, omega))
                    beta = (alpha/omega) * torch.div(torch.matmul(r.T, r_0),torch.matmul(r_old.T, r_0))
                    p = torch.add(r, torch.mul(torch.sub(p, torch.mul(v, omega)), beta))

                k += 1
                r_abs = torch.norm(r, p = 'fro')

            # print(x)
            # print(r_abs)
            # print()

    return x

def BiCGSTAB_batched(A, B, X0=None, max_iter=1000, tol=1e-6):
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
    batch, channel, nx, ny = A.shape
    m = B.shape[-1]
    
    if X0 is None:
        X = torch.zeros(batch, channel, nx, m, dtype=torch.float64, device=A.device)
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
        if torch.all(residual_norm < tol):
            return X, 0
    
    return X, 1


if __name__ == '__main__':

    A = torch.tensor([[[3., 2., -1.], [2., -2., 4.], [-1.,.5,-1.]], 
                     [[1., 3., -1.], [5., -2., 5.], [1.,.3,-4.]]]).reshape(1, 2, 3, 3)
    B = torch.tensor([[[1.,1.],
                      [-2.,-2.],
                      [0.,-0.]
                      ],
                      [[2.,0.],
                      [-1.,3.],
                      [1.,1.]
                      ]]).reshape(1, 2, 3, 2)

    print(A.size(), B.size())
    X = BiCGSTAB_batched(A, B)
    print(X)

# tensor([[[[ 0.1833,  0.6167],
#           [ 0.5513, -0.2436],
#           [-0.1628, -0.1141]]]]

