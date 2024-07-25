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
        
        r0  = torch.matmul(A, x)
        r_0 = r = p  = torch.sub(b, r0)
        r_abs = r0_abs = torch.norm(r_0, p = 'fro')
        print(r_abs)
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


if __name__ == '__main__':
    # A = torch.tensor([[3., 2., -1.], [2., -2., 4.], [-1.,.5,-1.]])
    # x = torch.randn((3,1))
    # b = torch.tensor([[1.,-2.,0.]]).reshape(3,1)
    # print(A.size(), x.size(), b.size())

    # x = BiCGSTAB_1(A, x, b, 10000)
    # print(x)


    batch_size = 1
    channels = 1
    nx, ny = 5, 5  # Size of the matrix
    A = torch.rand(batch_size, channels, nx, ny)  # Random coefficient matrix
    x = torch.zeros(batch_size, channels, nx, ny)  # Initial guess
    b = torch.rand(batch_size, channels, nx, ny)  # Random right-hand side

    # Solve Ax = b
    x_solution = BiCGSTAB_1(A, x, b, kmax=1000, eps=1e-6)
    print(x_solution)


    # b = torch.tensor([[3.,-2.,1.]])
    # alphs = 2
    # print(torch.mul(b, alphs))


    # A = A.numpy()
    # b = b.numpy()
    # x, exitCode = bicg(A, b,maxiter = 1000,  atol=1e-9)
    # print(exitCode, x)
    

