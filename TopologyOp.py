import torch
import numpy as np
import time
import torch.sparse
from matplotlib import colors
import matplotlib.pyplot as plt
from torch_sparse_solve import solve
"""
This is just a trial file for the implementation of topology optimization in 2D and 3D
With both single load and worst case 

https://github.com/arjendeetman/TopOpt-MMA-Python
"""
# torch.set_default_tensor_type('torch.cuda.DoubleTensor')
torch.set_default_tensor_type('torch.DoubleTensor')

#element stiffness matrix
def lk():
    E = 1
    nu = 0.3
    k = np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
    KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
    [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
    [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
    [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
    [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
    [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
    [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
    [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ]);
    return (KE)
# Optimality criterion
def oc(nelx,nely,x,volfrac,dc,dv,g):
    l1 = 0
    l2 = 1e9
    move = 0.2
    # reshape to perform vector operations
    xnew = torch.zeros(nelx*nely)
    while (l2-l1)/(l1+l2)>1e-3:
        lmid = 0.5*(l2+l1)
        xnew[:] = torch.maximum(torch.tensor(0.0),torch.maximum(x-move,torch.minimum(torch.tensor(1.0),torch.minimum(x+move,x*torch.sqrt(-dc/dv/lmid)))))
        gt = g+torch.sum((dv*(xnew-x)))
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
    return (xnew,gt)


class TopoOpt():
    def __init__(self):
        self.device = 'cpu'



        self.asd = 1



    def Opt2D(self, nelx, nely, volfrac, p=3, rmin=1.5, filter_m='Density', xsolver='OC'):
        print("Minimum complicance problem with OC")
        print(f"Number of degrees:{str(nelx)} x {str(nely)} = {str(nelx*nely)}")
        print(f"Volume fration: {volfrac}, Penalty p: {p}, Fileter radius: {rmin}")
        print(f"Filter method: {filter_m}")
        print(f"Optimizer: {xsolver}")

        # max and min stiffness
        E_min = torch.tensor(1e-9)
        E_max = torch.tensor(1.0)

        #Degree of freedoms
        NDOF = 2*(nelx+1)*(nely+1)

        n = nelx*nely
        x = volfrac*torch.ones(n)
        xPhys = torch.clone(x)
        dc = torch.zeros((nely,nelx))

        #Initialize the OC
        if xsolver=='OC':
            xold1 = torch.clone(x)
            g = 0

        KE= lk()
        KE_T = torch.from_numpy(KE)
        edofMat = torch.zeros((n,8),dtype=torch.int)
        for elx in range(nelx):
            for ely in range(nely):
                el = ely+elx*nely
                n1 = (nely+1)*elx+ely
                n2 = (nely+1)*(elx+1)+ely
                temp = np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3, 2*n2, 2*n2+1, 2*n1, 2*n1+1])
                edofMat[el,:] = torch.from_numpy(temp)

        #Construct the index pointers for the coo format
        iK = torch.kron(edofMat, torch.ones((8,1)))
        iK = torch.flatten(iK)
        jK = torch.kron(edofMat, torch.ones((1,8)))
        jK = torch.flatten(jK)

        #Filter: Build and assemble the index + data vectors for the coo matrix format
        nfilter = int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
        iH = torch.zeros(nfilter)
        jH = torch.zeros(nfilter)
        sH = torch.zeros(nfilter)

        cc = 0
        for i in range(nelx):
            for j in range(nely):
                row = i*nely+j
                kk1 = int(np.maximum(i-(np.ceil(rmin)-1),0))
                kk2 = int(np.minimum(i+np.ceil(rmin),nelx))
                ll1 = int(np.maximum(j-(np.ceil(rmin)-1),0))
                ll2 = int(np.minimum(j+np.ceil(rmin),nely))
                for k in range(kk1,kk2):
                    for l in range(ll1,ll2):
                        col = k*nely+l
                        fac = rmin - np.sqrt(((i-k)*(i-k)+(j-l)*(j-l)))
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = torch.maximum(torch.tensor(0),torch.tensor(fac))
                        cc = cc + 1

        #Finalize assemble and covert to csc format
        Idx = torch.concat((jH.unsqueeze(1),iH.unsqueeze(1)),dim=1).T
        H = torch.sparse_coo_tensor(Idx, sH, (n,n))
        H_csc = H.to_sparse_csc()
        # pp = H_csc.to_dense()
        # Hs = torch.sum(pp,dim=1).reshape(-1,1)
        Hs = torch.sparse.sum(H, [1]).to_dense().reshape(-1,1)


        #Boundary Condition's and the Support
        dofs = np.arange(2*(nelx+1)*(nely+1))
        fixed = np.union1d(dofs[0:2 * (nely + 1):2], np.array([2 * (nelx + 1) * (nely + 1) - 1]))
        free = np.setdiff1d(dofs, fixed)
        dofs = torch.from_numpy(dofs)
        fixed = torch.from_numpy(fixed)
        free = torch.from_numpy(free)
        #Solution and RHS vectors
        f = torch.zeros((NDOF,1))
        u = torch.zeros((NDOF,1))
        #Set the Load
        f[1,0] = -1
        #Set the loop counter and gradient vectors
        loop = 0
        change = 1
        dv = torch.ones(n)
        dc = torch.ones(n)
        ce = torch.ones(n)

        Idx_K = torch.concat((jK.unsqueeze(1), iK.unsqueeze(1)), dim=1).T
        while (change>1e-3) and (loop<2e3):
            start = time.time()
            loop +=1
            #Setup and solve FE problem
            ssp = torch.from_numpy((KE.reshape(1,-1).T))
            sK = (ssp * (E_min+(xPhys)**p)*(E_max-E_min)).transpose(1, 0).flatten()
            K = torch.sparse_coo_tensor(Idx_K, sK, (NDOF, NDOF))
            K = K.to_sparse_csc().to_dense()
            # Remove constrained dofs from matrix
            # K = K[free, :][:, free].to_sparse_csc()
            K = K[free, :][:, free]
            # Solve system
            # u[free, 0] = torch.linalg.solve(K, f[free, 0])
            #Transfer to Batch Form
            K_batch = K.unsqueeze(dim=0)
            u_batch = u.unsqueeze(dim=0)
            f_batch = f.unsqueeze(dim=0)
            K_sp_batch = K_batch.to_sparse()
            f_free_batch = f_batch[:,free, :]
            u_batch[:,free,:] = solve(K_sp_batch, f_free_batch)
            # u = u_batch.reshape(-1,1)
            end = time.time()
            print('using {}'.format(end - start))
            # Objective and sensitivity
            part1 = torch.mm(u[edofMat.numpy()].reshape(nelx * nely, 8), KE_T)
            part2 = u[edofMat.numpy()].reshape(nelx * nely, 8)
            ce[:] = torch.sum(part1 * part2, dim=1)
            obj = torch.sum((E_min + xPhys ** p * (E_max - E_min)) * ce)
            dc[:] = (-p * xPhys ** (p - 1) * (E_max - E_min)) * ce
            dv[:] = torch.ones(nely * nelx)
            # Sensitivity filtering:
            if filter_m=='Sensitivity':
                dc[:] = (torch.mv(H_csc , x * dc).reshape(1,-1).T / Hs)[:, 0] / torch.maximum(torch.tensor(0.001), x)
            elif filter_m=='Density':
                dc[:] = torch.sparse.mm(H_csc , (dc.reshape(1,-1).T / Hs))[:, 0]
                dv[:] = torch.sparse.mm(H_csc , (dv.reshape(1,-1).T / Hs))[:, 0]
            # Optimality criteria
            if xsolver=='OC':
                xold1[:] = x
                (x[:], g) = oc(nelx, nely, x, volfrac, dc, dv, g)
            # Filter design variables
            if filter_m == 'Sensitivity':
                xPhys[:] = x
            elif filter_m=='Density':
                xPhys[:] = (torch.sparse.mm(H_csc , x.reshape(1,-1).T) / Hs)[:, 0]
            # Compute the change by the inf. norm
            change = torch.linalg.norm(x.reshape(nelx * nely, 1) - xold1.reshape(nelx * nely, 1), ord=float('inf'))
            end = time.time()
            # print('using {}'.format(end - start))
            # Write iteration history to screen (req. Python 2.6 or newer)
            print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}, time: {4:.3f}".format(loop, obj, x.sum() / n, change, end - start))
        # Plot result
        fig, ax = plt.subplots()
        im = ax.imshow(-xPhys.detach().cpu().numpy().reshape((nelx, nely)).T, cmap='gray', \
                       interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
        plt.show()

        a = 1









    def MPM(self, K, P):
        pass
        # return u_wst, f_wst

    def CG(self, A, x):
        pass
        # return A_inv





if __name__ == "__main__":
    TTT = TopoOpt()
    p = TTT.Opt2D(150,50, 0.5)
        # start = time.time()
        # output1 = grid_sample_3d(image, grid)
        # end = time.time()
        # print('using {}'.format(end - start))