import torch
import numpy as np
import time
import torch.sparse
from matplotlib import colors
import matplotlib.pyplot as plt
# from torch_sparse_solve import solve
import sklearn.metrics.pairwise as sp
import scipy.sparse as scisp
from MMA import mmasub, subsolv
import mayavi.mlab as mlab

# import dolfin as df
"""
This is just a trial file for the implementation of topology optimization in 2D and 3D
With both single load and worst case 

https://github.com/arjendeetman/TopOpt-MMA-Python

3D version reference:
Zuo ZH, Xie YM (2015) A simple and compact Python code for complex 3D topology optimization. Advances in Engineering Software 85: 1-11. 
"""
# torch.set_default_tensor_type('torch.cuda.DoubleTensor')
torch.set_default_tensor_type('torch.DoubleTensor')


# element stiffness matrix
def lk():
    E = 1
    nu = 0.3
    k = np.array(
        [1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8, -1 / 4 + nu / 12, -1 / 8 - nu / 8,
         nu / 6, 1 / 8 - 3 * nu / 8])
    KE = E / (1 - nu ** 2) * np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                                       [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                                       [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                                       [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                                       [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                                       [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                                       [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                                       [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]]);
    return (KE)


# Optimality criterion
def oc(nelx, nely, nelz, x, volfrac, dc, dv, g):
    l1 = 0.0
    l2 = 1e9
    move = 0.2
    # reshape to perform vector operations
    if nelz == 0:
        xnew = torch.zeros(nelx * nely)
    else:
        xnew = torch.zeros(nelx * nely * nelz)
    while (l2 - l1) / (l1 + l2) > 1e-3 and (l1 + l2) > 0:
        lmid = 0.5 * (l2 + l1)
        xnew[:] = torch.maximum(torch.tensor(0.0), torch.maximum(x - move, torch.minimum(torch.tensor(1.0),
                                                                                         torch.minimum(x + move,
                                                                                                       x * torch.sqrt(
                                                                                                           -dc / dv / lmid)))))
        gt = g + torch.sum((dv * (xnew - x)))
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
        # print(l1, l2, l1+l2)
    return (xnew, gt)


def oc_np(nelx, nely, nelz, x, volfrac, dc, dv, g, H, Hs):
    l1 = 0.0
    l2 = 1e9
    move = 0.2
    # reshape to perform vector operations
    if nelz == 0:
        xnew = np.zeros(nelx * nely)
    else:
        xnew = np.zeros(nelx * nely * nelz)
        ppp = np.zeros(nelx * nely * nelz)
    while (l2 - l1) / (l1 + l2) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        pp = -dc / dv
        xnew[:] = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(pp / lmid)))))
        ppp[:] = np.asarray(H * xnew[np.newaxis].T / Hs)[:, 0]
        gt = ppp.sum() - volfrac * nelx * nely * nelz
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
        # print(l1, l2, l1+l2)
    return (xnew, ppp)


def lk_H8(nu):
    A = np.array(
        [[32, 6, -8, 6, -6, 4, 3, -6, -10, 3, -3, -3, -4, -8], [-48, 0, 0, -24, 24, 0, 0, 0, 12, -12, 0, 12, 12, 12]])
    k = 1 / 144 * np.matmul(A.conj().transpose(), np.vstack((1, nu))).reshape(-1)
    K1 = np.array([
        [k[0], k[1], k[1], k[2], k[4], k[4]],
        [k[1], k[0], k[1], k[3], k[5], k[6]],
        [k[1], k[1], k[0], k[3], k[6], k[5]],
        [k[2], k[3], k[3], k[0], k[7], k[7]],
        [k[4], k[5], k[6], k[7], k[0], k[1]],
        [k[4], k[6], k[5], k[7], k[1], k[0]]
    ])
    K2 = np.array([
        [k[8], k[7], k[11], k[5], k[3], k[6]],
        [k[7], k[8], k[11], k[4], k[2], k[4]],
        [k[9], k[9], k[12], k[6], k[3], k[5]],
        [k[5], k[4], k[10], k[8], k[1], k[9]],
        [k[3], k[2], k[4], k[1], k[8], k[11]],
        [k[10], k[3], k[5], k[11], k[9], k[12]]
    ])
    K3 = np.array([
        [k[5], k[6], k[3], k[8], k[11], k[7]],
        [k[6], k[5], k[3], k[9], k[12], k[9]],
        [k[4], k[4], k[2], k[7], k[11], k[8]],
        [k[8], k[9], k[1], k[5], k[10], k[4]],
        [k[11], k[12], k[9], k[10], k[5], k[3]],
        [k[1], k[11], k[8], k[3], k[4], k[2]]
    ])
    K4 = np.array([
        [k[13], k[10], k[10], k[12], k[9], k[9]],
        [k[10], k[13], k[10], k[11], k[8], k[7]],
        [k[10], k[10], k[13], k[11], k[7], k[8]],
        [k[12], k[11], k[11], k[13], k[6], k[6]],
        [k[9], k[8], k[7], k[6], k[13], k[10]],
        [k[9], k[7], k[8], k[6], k[10], k[13]]
    ])
    K5 = np.array([
        [k[0], k[1], k[7], k[2], k[4], k[3]],
        [k[1], k[0], k[7], k[3], k[5], k[10]],
        [k[7], k[7], k[0], k[4], k[10], k[5]],
        [k[2], k[3], k[4], k[0], k[7], k[1]],
        [k[4], k[5], k[10], k[7], k[0], k[7]],
        [k[3], k[10], k[5], k[1], k[7], k[0]]
    ])
    K6 = np.array([
        [k[13], k[10], k[6], k[12], k[9], k[11]],
        [k[10], k[13], k[6], k[11], k[8], k[1]],
        [k[6], k[6], k[13], k[9], k[1], k[8]],
        [k[12], k[11], k[9], k[13], k[6], k[10]],
        [k[9], k[8], k[1], k[6], k[13], k[6]],
        [k[11], k[1], k[8], k[10], k[6], k[13]]
    ])
    Line1 = np.concatenate((K1, K2, K3, K4), axis=1)
    Line2 = np.concatenate((K2.transpose(), K5, K6, K3.transpose()), axis=1)
    Line3 = np.concatenate((K3.transpose(), K6, K5.transpose(), K2.transpose()), axis=1)
    Line4 = np.concatenate((K4, K3, K2, K1.transpose()), axis=1)
    KE = np.concatenate((Line1, Line2, Line3, Line4), axis=0)

    KE = 1 / ((nu + 1) * (1 - 2 * nu)) * KE
    return KE


class TopoOpt():
    def __init__(self, vol):
        # self.device = 'cpu'
        self.volfrac = vol
        self.asd = 1

    def Opt2D(self, nelx, nely, volfrac, p=3, rmin=1.5, filter_m='Density', xsolver='OC'):
        print("Minimum complicance problem with OC")
        print(f"Number of degrees:{str(nelx)} x {str(nely)} = {str(nelx * nely)}")
        print(f"Volume fration: {volfrac}, Penalty p: {p}, Fileter radius: {rmin}")
        print(f"Filter method: {filter_m}")
        print(f"Optimizer: {xsolver}")

        # max and min stiffness
        E_min = torch.tensor(1e-9)
        E_max = torch.tensor(1.0)

        # Degree of freedoms
        NDOF = 2 * (nelx + 1) * (nely + 1)

        n = nelx * nely
        x = volfrac * torch.ones(n)
        xPhys = torch.clone(x)
        dc = torch.zeros((nely, nelx))

        # Initialize the OC
        if xsolver == 'OC':
            xold1 = torch.clone(x)
            g = 0

        KE = lk()
        KE_T = torch.from_numpy(KE)
        edofMat = torch.zeros((n, 8), dtype=torch.int)
        for elx in range(nelx):
            for ely in range(nely):
                el = ely + elx * nely
                n1 = (nely + 1) * elx + ely
                n2 = (nely + 1) * (elx + 1) + ely
                temp = np.array(
                    [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1])
                edofMat[el, :] = torch.from_numpy(temp)

        # Construct the index pointers for the coo format
        iK = torch.kron(edofMat, torch.ones((8, 1)))
        iK = torch.flatten(iK)
        jK = torch.kron(edofMat, torch.ones((1, 8)))
        jK = torch.flatten(jK)

        # Filter: Build and assemble the index + data vectors for the coo matrix format
        nfilter = int(nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1) ** 2))
        iH = torch.zeros(nfilter)
        jH = torch.zeros(nfilter)
        sH = torch.zeros(nfilter)

        cc = 0
        for i in range(nelx):
            for j in range(nely):
                row = i * nely + j
                kk1 = int(np.maximum(i - (np.ceil(rmin) - 1), 0))
                kk2 = int(np.minimum(i + np.ceil(rmin), nelx))
                ll1 = int(np.maximum(j - (np.ceil(rmin) - 1), 0))
                ll2 = int(np.minimum(j + np.ceil(rmin), nely))
                for k in range(kk1, kk2):
                    for l in range(ll1, ll2):
                        col = k * nely + l
                        fac = rmin - np.sqrt(((i - k) * (i - k) + (j - l) * (j - l)))
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = torch.maximum(torch.tensor(0), torch.tensor(fac))
                        cc = cc + 1

        # Finalize assemble and covert to csc format
        Idx = torch.concat((jH.unsqueeze(1), iH.unsqueeze(1)), dim=1).T
        H = torch.sparse_coo_tensor(Idx, sH, (n, n))
        H_csc = H.to_sparse_csc()
        # pp = H_csc.to_dense()
        # Hs = torch.sum(pp,dim=1).reshape(-1,1)
        Hs = torch.sparse.sum(H, [1]).to_dense().reshape(-1, 1)

        # Boundary Condition's and the Support
        dofs = np.arange(2 * (nelx + 1) * (nely + 1))
        fixed = np.union1d(dofs[0:2 * (nely + 1):2], np.array([2 * (nelx + 1) * (nely + 1) - 1]))
        free = np.setdiff1d(dofs, fixed)
        dofs = torch.from_numpy(dofs)
        fixed = torch.from_numpy(fixed)
        free = torch.from_numpy(free)
        # Solution and RHS vectors
        f = torch.zeros((NDOF, 1))
        u = torch.zeros((NDOF, 1))
        # Set the Load
        f[1, 0] = -1
        # Set the loop counter and gradient vectors
        loop = 0
        change = 1
        dv = torch.ones(n)
        dc = torch.ones(n)
        ce = torch.ones(n)

        Idx_K = torch.concat((jK.unsqueeze(1), iK.unsqueeze(1)), dim=1).T
        while (change > 1e-3) and (loop < 2e3):
            start = time.time()
            loop += 1
            # Setup and solve FE problem
            ssp = torch.from_numpy((KE.reshape(1, -1).T))
            sK = (ssp * (E_min + (xPhys) ** p) * (E_max - E_min)).transpose(1, 0).flatten()
            K = torch.sparse_coo_tensor(Idx_K, sK, (NDOF, NDOF))
            K = K.to_sparse_csc().to_dense()
            # Remove constrained dofs from matrix
            # K = K[free, :][:, free].to_sparse_csc()
            K = K[free, :][:, free]
            # Solve system
            # u[free, 0] = torch.linalg.solve(K, f[free, 0])
            # Transfer to Batch Form
            K_batch = K.unsqueeze(dim=0)
            u_batch = u.unsqueeze(dim=0)
            f_batch = f.unsqueeze(dim=0)
            K_sp_batch = K_batch.to_sparse()
            f_free_batch = f_batch[:, free, :]
            u_batch[:, free, :] = solve(K_sp_batch, f_free_batch)
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
            if filter_m == 'Sensitivity':
                dc[:] = (torch.mv(H_csc, x * dc).reshape(1, -1).T / Hs)[:, 0] / torch.maximum(torch.tensor(0.001), x)
            elif filter_m == 'Density':
                dc[:] = torch.sparse.mm(H_csc, (dc.reshape(1, -1).T / Hs))[:, 0]
                dv[:] = torch.sparse.mm(H_csc, (dv.reshape(1, -1).T / Hs))[:, 0]
            # Optimality criteria
            if xsolver == 'OC':
                xold1[:] = x
                (x[:], g) = oc(nelx, nely, 0, x, volfrac, dc, dv, g)
            # Filter design variables
            if filter_m == 'Sensitivity':
                xPhys[:] = x
            elif filter_m == 'Density':
                xPhys[:] = (torch.sparse.mm(H_csc, x.reshape(1, -1).T) / Hs)[:, 0]
            # Compute the change by the inf. norm
            change = torch.linalg.norm(x.reshape(nelx * nely, 1) - xold1.reshape(nelx * nely, 1), ord=float('inf'))
            end = time.time()
            # print('using {}'.format(end - start))
            # Write iteration history to screen (req. Python 2.6 or newer)
            print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}, time: {4:.3f}".format(loop, obj, x.sum() / n,
                                                                                               change, end - start))
        # Plot result
        fig, ax = plt.subplots()
        im = ax.imshow(-xPhys.detach().cpu().numpy().reshape((nelx, nely)).T, cmap='gray', \
                       interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
        plt.show()

        a = 1

    def Opt3D(self, nelx, nely, nelz, volfrac, p=3, rmin=1.5):
        print("Minimum complicance problem with OC")
        print(f"Number of degrees:{str(nelx)} x {str(nely)} x {str(nelz)} = {str(nelx * nely * nelz)}")
        print(f"Volume fration: {volfrac}, Penalty p: {p}, Fileter radius: {rmin}")
        # print(f"Filter method: {filter_m}")
        # print(f"Optimizer: {xsolver}")
        maxloop = 200
        tolx = 0.01
        displayflag = 0
        nu = 0.3
        # max and min stiffness
        E_min = torch.tensor(1e-9)
        E_max = torch.tensor(1.0)

        # User-defined load DOFs
        il = nelx
        jl = 0
        kl = np.arange(nelz + 1)
        il, jl, kl = np.meshgrid(il, jl, kl)
        loadnid = kl * (nelx + 1) * (nely + 1) + il * (nely + 1) + (nely + 1 - jl)
        loaddof = 3 * loadnid.flatten() - 1

        # User-defined support fixed DOFs
        iif = 0
        jf = np.arange(nely + 1)
        kf = np.arange(nelz + 1)
        a, b, c = np.meshgrid(iif, jf, kf)
        fixednid = c * (nelx + 1) * (nely + 1) + a * (nely + 1) + (nely + 1 - b)
        fixeddof = np.concatenate((3 * fixednid.flatten(), 3 * fixednid.flatten() - 1, 3 * fixednid.flatten() - 2),
                                  axis=0) - 1

        # Prepare for the finite element analysis
        nele = nelx * nely * nelz
        ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)

        # F = sparse(loaddof, 1, -1, ndof, 1)
        F = torch.zeros(ndof, 1)
        # F[:500] = 1
        F[loaddof - 1, :] = -1
        U = torch.zeros(ndof, 1)
        freedofs = np.setdiff1d(np.arange(ndof), fixeddof)
        # Verified no problem

        KE = lk_H8(nu)
        KE_T = torch.from_numpy(KE)
        nodegrd = np.arange((nely + 1) * (nelx + 1)).reshape(nely + 1, nelx + 1, order='F')
        nodeids = nodegrd[:-1, :-1].reshape(-1, 1, order='F')
        nodeidz = np.arange(0, (nelz) * (nely + 1) * (nelx + 1), (nely + 1) * (nelx + 1))
        nodeids = np.kron(np.ones(nodeidz.shape), nodeids) + np.kron(np.ones(nodeids.shape), nodeidz)
        edofVec = (3 * (nodeids + 1) + 1).reshape(-1, 1, order='F')

        temp_mtix = np.array(
            [0, 1, 2, 3 * nely + 3, 3 * nely + 4, 3 * nely + 5, 3 * nely + 0, 3 * nely + 1, 3 * nely + 2, -3, -2, -1
                , 3 * (nely + 1) * (nelx + 1) + 0, 3 * (nely + 1) * (nelx + 1) + 1, 3 * (nely + 1) * (nelx + 1) + 2
                , 3 * (nely + 1) * (nelx + 1) + 3 * nely + 3, 3 * (nely + 1) * (nelx + 1) + 3 * nely + 4,
             3 * (nely + 1) * (nelx + 1) + 3 * nely + 5, 3 * (nely + 1) * (nelx + 1) + 3 * nely + 0,
             3 * (nely + 1) * (nelx + 1) + 3 * nely + 1, 3 * (nely + 1) * (nelx + 1) + 3 * nely + 2
                , 3 * (nely + 1) * (nelx + 1) - 3, 3 * (nely + 1) * (nelx + 1) - 2, 3 * (nely + 1) * (nelx + 1) - 1])

        a1 = np.kron(np.ones((1, 24)), edofVec.reshape(-1, 1))
        a2 = np.kron(np.ones((nele, 1)), temp_mtix)
        edoMat = np.kron(np.ones((1, 24)), edofVec.reshape(-1, 1)) + np.kron(np.ones((nele, 1)), temp_mtix)
        edoMat = torch.from_numpy(edoMat)
        iK = torch.kron(edoMat, torch.ones((24, 1)))
        iK = torch.flatten(iK)
        jK = torch.kron(edoMat, torch.ones((1, 24)))
        jK = torch.flatten(jK)

        # Prepare the filter
        nfilter = int(nele * ((2 * (np.ceil(rmin) - 1) + 1) ** 2))
        # iH = torch.ones(nfilter)
        # jH = torch.ones(nfilter)
        # sH = torch.zeros(nfilter)
        iH = []
        jH = []
        sH = []
        k = 0
        for k1 in range(1, nelz + 1):
            for i1 in range(1, nelx + 1):
                for j1 in range(1, nely + 1):
                    e1 = (k1 - 1) * nelx * nely + (i1 - 1) * nely + j1
                    kk1 = int(np.maximum(k1 - (np.ceil(rmin) - 1), 1))
                    kk2 = int(np.minimum(k1 + np.ceil(rmin) - 1, nelz))
                    ii1 = int(np.maximum(i1 - (np.ceil(rmin) - 1), 1))
                    ii2 = int(np.minimum(i1 + np.ceil(rmin) - 1, nelx))
                    jj1 = int(np.maximum(j1 - (np.ceil(rmin) - 1), 1))
                    jj2 = int(np.minimum(j1 + np.ceil(rmin) - 1, nely))
                    for k2 in range(kk1, kk2 + 1):
                        for i2 in range(ii1, ii2 + 1):
                            for j2 in range(jj1, jj2 + 1):
                                e2 = (k2 - 1) * nelx * nely + (i2 - 1) * nely + j2
                                fac = rmin - np.sqrt(((i1 - i2) ** 2 + (j1 - j2) ** 2 + (k1 - k2) ** 2))
                                # if k < nfilter:
                                #     iH[k] = e1
                                #     jH[k] = e2
                                #     sH[k] = torch.maximum(torch.tensor(0),torch.tensor(fac))
                                # else:
                                #     iH = torch.concat(iH, e1)
                                #     jH = torch.concat(jH, e2)
                                #     sH = torch.concat(sH, torch.maximum(torch.tensor(0),torch.tensor(fac)))
                                iH.append(e1)
                                jH.append(e2)
                                sH.append(torch.maximum(torch.tensor(0), torch.tensor(fac)))
                                k = k + 1
        # print(k)
        # Finalize assemble and covert to csc format
        iH = torch.tensor(iH)
        jH = torch.tensor(jH)
        sH = torch.tensor(sH)
        # print(sH[:10])
        Idx = torch.concat((jH.unsqueeze(1) - 1, iH.unsqueeze(1) - 1), dim=1).T
        H = torch.sparse_coo_tensor(Idx, sH, (nely * nelx * nelz, nely * nelx * nelz))
        H_csc = H.to_sparse_csc()
        Hs = torch.sparse.sum(H, [1]).to_dense().reshape(-1, 1)

        x = volfrac * torch.ones(nely * nelx * nelz)
        xPhys = torch.clone(x)
        xold1 = torch.clone(x)
        g = 0
        loop = 0
        change = 1
        Idx_K = torch.concat((jK.unsqueeze(1) - 1, iK.unsqueeze(1) - 1), dim=1).T
        dv = torch.ones(nely * nelx * nelz)
        dc = torch.ones(nely * nelx * nelz)
        ce = torch.ones(nely * nelx * nelz)
        while (change > tolx) and (loop < maxloop):
            start = time.time()
            loop += 1
            # FE-Analysis
            ssp = torch.from_numpy((KE.reshape(1, -1).T))
            sK = (ssp * (E_min + (xPhys) ** p) * (E_max - E_min)).transpose(1, 0).flatten()
            a1 = torch.min(Idx_K)
            K = torch.sparse_coo_tensor(Idx_K, sK, (ndof, ndof))
            K = K.to_dense()
            K = (K + torch.transpose(K, 1, 0)) / 2
            K1 = K.numpy()

            # K = K[freedofs, :][:, freedofs].to_sparse_csc()
            K = K[freedofs, :][:, freedofs]
            # Solve system
            U[freedofs, 0] = torch.linalg.solve(K, F[freedofs, 0])
            # K_batch = K.unsqueeze(dim=0)
            # u_batch = U.unsqueeze(dim=0)
            # f_batch = F.unsqueeze(dim=0)
            # K_sp_batch = K_batch.to_sparse()
            # f_free_batch = f_batch[:,freedofs, :]
            # u_batch[:,freedofs,:] = solve(K_sp_batch, f_free_batch)
            # U = u_batch.reshape(-1, 1)
            # U = something already

            # objective function and sensitivity analysis
            part1 = torch.mm(U[edoMat.numpy() - 1].reshape(nelx * nely * nelz, 24), KE_T)
            part2 = U[edoMat.numpy() - 1].reshape(nelx * nely * nelz, 24)
            ce[:] = torch.sum(part1 * part2, dim=1)
            obj = torch.sum((E_min + xPhys ** p * (E_max - E_min)) * ce)
            dc[:] = (-p * xPhys ** (p - 1) * (E_max - E_min)) * ce
            dv[:] = torch.ones(nely * nelx * nelz)

            # Filtering and modification of sensitivities
            dc[:] = torch.sparse.mm(H_csc, (dc.reshape(1, -1).T / Hs))[:, 0]
            dv[:] = torch.sparse.mm(H_csc, (dv.reshape(1, -1).T / Hs))[:, 0]

            # OC update
            l1 = 0.5
            l2 = 1e9
            move = 0.2
            # reshape to perform vector operations
            # xnew = torch.zeros(nelx * nely * nelz)
            while (l2 - l1) / (l1 + l2) > 1e-3:
                lmid = 0.5 * (l2 + l1)
                xnew = torch.maximum(torch.tensor(0.0), torch.maximum(x - move, torch.minimum(torch.tensor(1.0),
                                                                                              torch.minimum(x + move,
                                                                                                            x * torch.sqrt(
                                                                                                                -dc / dv / lmid)))))
                xPhys[:] = (torch.sparse.mm(H_csc, xnew.reshape(1, -1).T) / Hs)[:, 0]
                if torch.sum(xPhys) > volfrac * nelx * nely * nelz:
                    l1 = lmid
                else:
                    l2 = lmid
                # print(l1, l2)
            x = xnew
            change = torch.linalg.norm(x.reshape(nelx * nely * nelz, 1) - xold1.reshape(nelx * nely * nelz, 1),
                                       ord=float('inf'))
            end = time.time()
            # print('using {}'.format(end - start))
            # Write iteration history to screen (req. Python 2.6 or newer)
            print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}, time: {4:.3f}".format(loop, obj, x.sum() / (
                        nelx * nely * nelz), change, end - start))
            # How to save files?

            a = 1

        return 0

    def Opt3D_Grid(self, x, volfrac, p=3, rmin=1.5, maxloop=200):
        original_shape = x.shape
        a1 = x.clone().detach().cpu().numpy()
        nelx, nely, nelz = a1.shape
        print("Minimum complicance problem with OC")
        print(f"Number of degrees:{str(nelx)} x {str(nely)} x {str(nelz)} = {str(nelx * nely * nelz)}")
        print(f"Volume fration: {volfrac}, Penalty p: {p}, Fileter radius: {rmin}")
        maxloop = maxloop
        tolx = 0.01
        displayflag = 0
        nu = 0.3
        # max and min stiffness
        E_min = torch.tensor(1e-9)
        E_max = torch.tensor(1.0)

        # User-defined load DOFs
        il = nelx
        jl = 0
        kl = np.arange(nelz + 1)
        il, jl, kl = np.meshgrid(il, jl, kl)
        loadnid = kl * (nelx + 1) * (nely + 1) + il * (nely + 1) + (nely + 1 - jl)
        loaddof = 3 * loadnid.flatten() - 1

        # User-defined support fixed DOFs
        iif = 0
        jf = np.arange(nely + 1)
        kf = np.arange(nelz + 1)
        a, b, c = np.meshgrid(iif, jf, kf)
        fixednid = c * (nelx + 1) * (nely + 1) + a * (nely + 1) + (nely + 1 - b)
        fixeddof = np.concatenate((3 * fixednid.flatten(), 3 * fixednid.flatten() - 1, 3 * fixednid.flatten() - 2),
                                  axis=0) - 1

        # Prepare for the finite element analysis
        nele = nelx * nely * nelz
        ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)

        # F = sparse(loaddof, 1, -1, ndof, 1)
        F = torch.zeros(ndof, 1)
        # F[:500] = 1
        F[loaddof - 1, :] = -1
        U = torch.zeros(ndof, 1)
        freedofs = np.setdiff1d(np.arange(ndof), fixeddof)
        # Verified no problem

        KE = lk_H8(nu)
        KE_T = torch.from_numpy(KE).float().cuda()
        nodegrd = np.arange((nely + 1) * (nelx + 1)).reshape(nely + 1, nelx + 1, order='F')
        nodeids = nodegrd[:-1, :-1].reshape(-1, 1, order='F')
        nodeidz = np.arange(0, (nelz) * (nely + 1) * (nelx + 1), (nely + 1) * (nelx + 1))
        nodeids = np.kron(np.ones(nodeidz.shape), nodeids) + np.kron(np.ones(nodeids.shape), nodeidz)
        edofVec = (3 * (nodeids + 1) + 1).reshape(-1, 1, order='F')

        temp_mtix = np.array(
            [0, 1, 2, 3 * nely + 3, 3 * nely + 4, 3 * nely + 5, 3 * nely + 0, 3 * nely + 1, 3 * nely + 2, -3, -2, -1
                , 3 * (nely + 1) * (nelx + 1) + 0, 3 * (nely + 1) * (nelx + 1) + 1, 3 * (nely + 1) * (nelx + 1) + 2
                , 3 * (nely + 1) * (nelx + 1) + 3 * nely + 3, 3 * (nely + 1) * (nelx + 1) + 3 * nely + 4,
             3 * (nely + 1) * (nelx + 1) + 3 * nely + 5, 3 * (nely + 1) * (nelx + 1) + 3 * nely + 0,
             3 * (nely + 1) * (nelx + 1) + 3 * nely + 1, 3 * (nely + 1) * (nelx + 1) + 3 * nely + 2
                , 3 * (nely + 1) * (nelx + 1) - 3, 3 * (nely + 1) * (nelx + 1) - 2, 3 * (nely + 1) * (nelx + 1) - 1])

        a1 = np.kron(np.ones((1, 24)), edofVec.reshape(-1, 1))
        a2 = np.kron(np.ones((nele, 1)), temp_mtix)
        edoMat = np.kron(np.ones((1, 24)), edofVec.reshape(-1, 1)) + np.kron(np.ones((nele, 1)), temp_mtix)
        edoMat = torch.from_numpy(edoMat).cuda()
        iK = torch.kron(edoMat, torch.ones((24, 1)))
        iK = torch.flatten(iK)
        jK = torch.kron(edoMat, torch.ones((1, 24)))
        jK = torch.flatten(jK)

        # Prepare the filter
        nfilter = int(nele * ((2 * (np.ceil(rmin) - 1) + 1) ** 2))
        # iH = torch.ones(nfilter)
        # jH = torch.ones(nfilter)
        # sH = torch.zeros(nfilter)
        iH = []
        jH = []
        sH = []
        k = 0
        for k1 in range(1, nelz + 1):
            for i1 in range(1, nelx + 1):
                for j1 in range(1, nely + 1):
                    e1 = (k1 - 1) * nelx * nely + (i1 - 1) * nely + j1
                    kk1 = int(np.maximum(k1 - (np.ceil(rmin) - 1), 1))
                    kk2 = int(np.minimum(k1 + np.ceil(rmin) - 1, nelz))
                    ii1 = int(np.maximum(i1 - (np.ceil(rmin) - 1), 1))
                    ii2 = int(np.minimum(i1 + np.ceil(rmin) - 1, nelx))
                    jj1 = int(np.maximum(j1 - (np.ceil(rmin) - 1), 1))
                    jj2 = int(np.minimum(j1 + np.ceil(rmin) - 1, nely))
                    for k2 in range(kk1, kk2 + 1):
                        for i2 in range(ii1, ii2 + 1):
                            for j2 in range(jj1, jj2 + 1):
                                e2 = (k2 - 1) * nelx * nely + (i2 - 1) * nely + j2
                                fac = rmin - np.sqrt(((i1 - i2) ** 2 + (j1 - j2) ** 2 + (k1 - k2) ** 2))
                                # if k < nfilter:
                                #     iH[k] = e1
                                #     jH[k] = e2
                                #     sH[k] = torch.maximum(torch.tensor(0),torch.tensor(fac))
                                # else:
                                #     iH = torch.concat(iH, e1)
                                #     jH = torch.concat(jH, e2)
                                #     sH = torch.concat(sH, torch.maximum(torch.tensor(0),torch.tensor(fac)))
                                iH.append(e1)
                                jH.append(e2)
                                sH.append(torch.maximum(torch.tensor(0), torch.tensor(fac)))
                                k = k + 1
        # print(k)
        # Finalize assemble and covert to csc format
        iH = torch.tensor(iH)
        jH = torch.tensor(jH)
        sH = torch.tensor(sH)
        # print(sH[:10])
        Idx = torch.concat((jH.unsqueeze(1) - 1, iH.unsqueeze(1) - 1), dim=1).T
        H = torch.sparse_coo_tensor(Idx, sH, (nely * nelx * nelz, nely * nelx * nelz))
        H_csc = H.to_sparse_csc()
        Hs = torch.sparse.sum(H, [1]).to_dense().reshape(-1, 1)

        # Here we only consider infill level only. The grid has shape of 1xNyxNxxNz
        # x = volfrac*torch.ones((1,nely,nelx,nelz))
        # x = volfrac*torch.ones(nely*nelx*nelz)
        x = x.flatten()
        xPhys = torch.clone(x)
        xold1 = torch.clone(x)

        g = 0
        loop = 0
        change = 1
        Idx_K = torch.concat((jK.unsqueeze(1) - 1, iK.unsqueeze(1) - 1), dim=1).T
        dv = torch.ones(nely * nelx * nelz)
        dc = torch.ones(nely * nelx * nelz)
        ce = torch.ones(nely * nelx * nelz)
        while (change > tolx) and (loop < maxloop):
            start = time.time()
            loop += 1
            # FE-Analysis
            ssp = torch.from_numpy((KE.reshape(1, -1).T)).cuda()
            sK = (ssp * (E_min + (xPhys) ** p) * (E_max - E_min)).transpose(1, 0).flatten()
            a1 = torch.min(Idx_K)
            K = torch.sparse_coo_tensor(Idx_K, sK, (ndof, ndof))
            K = K.to_dense()
            K = (K + torch.transpose(K, 1, 0)) / 2

            # K = K[freedofs, :][:, freedofs].to_sparse_csc()
            K = K[freedofs, :][:, freedofs].float()
            # Solve system
            U[freedofs, 0] = torch.linalg.solve(K, F[freedofs, 0])
            # K_batch = K.unsqueeze(dim=0)
            # u_batch = U.unsqueeze(dim=0)
            # f_batch = F.unsqueeze(dim=0)
            # K_sp_batch = K_batch.to_sparse()
            # f_free_batch = f_batch[:,freedofs, :]
            # u_batch[:,freedofs,:] = solve(K_sp_batch, f_free_batch)
            # U = u_batch.reshape(-1, 1)
            # U = something already

            # objective function and sensitivity analysis
            part1 = torch.mm(U[edoMat.cpu().numpy() - 1].reshape(nelx * nely * nelz, 24), KE_T)
            part2 = U[edoMat.cpu().numpy() - 1].reshape(nelx * nely * nelz, 24)
            ce[:] = torch.sum(part1 * part2, dim=1)
            obj = torch.sum((E_min + xPhys ** p * (E_max - E_min)) * ce)
            dc[:] = (-p * xPhys ** (p - 1) * (E_max - E_min)) * ce
            dv[:] = torch.ones(nely * nelx * nelz)

            # Filtering and modification of sensitivities
            dc[:] = torch.sparse.mm(H_csc, (dc.reshape(1, -1).T / Hs))[:, 0]
            dv[:] = torch.sparse.mm(H_csc, (dv.reshape(1, -1).T / Hs))[:, 0]

            # OC update
            l1 = 0.5
            l2 = 1e9
            move = 0.2
            # reshape to perform vector operations
            # xnew = torch.zeros(nelx * nely * nelz)
            while (l2 - l1) / (l1 + l2) > 1e-3:
                lmid = 0.5 * (l2 + l1)
                xnew = torch.maximum(torch.tensor(0.0), torch.maximum(x - move, torch.minimum(torch.tensor(1.0),
                                                                                              torch.minimum(x + move,
                                                                                                            x * torch.sqrt(
                                                                                                                -dc / dv / lmid)))))
                xPhys[:] = (torch.sparse.mm(H_csc, xnew.reshape(1, -1).T.double()) / Hs)[:, 0]
                if torch.sum(xPhys) > volfrac * nelx * nely * nelz:
                    l1 = lmid
                else:
                    l2 = lmid
                # print(l1, l2)
            x = xnew
            change = torch.linalg.norm(x.reshape(nelx * nely * nelz, 1) - xold1.reshape(nelx * nely * nelz, 1),
                                       ord=float('inf'))
            end = time.time()
            # print('using {}'.format(end - start))
            # Write iteration history to screen (req. Python 2.6 or newer)
            print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}, time: {4:.3f}".format(loop, obj, x.sum() / (
                        nelx * nely * nelz), change, end - start))
        # How to save files?

        # import mayavi.mlab as mlab
        # mlab.clf()
        # mlab.contour3d(T)
        # mlab.show()

        return x.reshape(original_shape)

    def Opt3D_NP(self, nelx, nely, nelz, volfrac, p=3, rmin=1.5, maxloop=200, ft=0):
        print("Minimum complicance problem with OC")
        print(f"Number of degrees:{str(nelx)} x {str(nely)} x {str(nelz)} = {str(nelx * nely * nelz)}")
        print(f"Volume fration: {volfrac}, Penalty p: {p}, Fileter radius: {rmin}")
        print(f"Filter method: {ft}")
        # print(f"Optimizer: {xsolver}")
        maxloop = maxloop
        tolx = 0.005
        displayflag = 0
        # Max and min stiffness
        Emin = 1e-9
        Emax = 1.0
        nu = 0.3

        # dofs
        ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)

        # USER - DEFINED LOAD DOFs
        kl = np.arange(nelz + 1)
        loadnid = kl * (nelx + 1) * (nely + 1) + (nely + 1) * (nelx + 1) - 1  # Node IDs
        loaddof = 3 * loadnid + 1  # DOFs

        # USER - DEFINED SUPPORT FIXED DOFs
        [jf, kf] = np.meshgrid(np.arange(nely + 1), np.arange(nelz + 1))  # Coordinates
        fixednid = (kf) * (nely + 1) * (nelx + 1) + jf  # Node IDs
        fixeddof = np.array([3 * fixednid, 3 * fixednid + 1, 3 * fixednid + 2]).flatten()  # DOFs

        # Allocate design variables (as array), initialize and allocate sens.
        x = volfrac * np.ones(nely * nelx * nelz, dtype=float)
        xold = x.copy()
        xPhys = x.copy()
        g = 0  # must be initialized to use the NGuyen/Paulino OC approach

        # FE: Build the index vectors for the for coo matrix format.
        KE = lk_H8(nu)
        edofMat = np.zeros((nelx * nely * nelz, 24), dtype=int)
        for elz in range(nelz):
            for elx in range(nelx):
                for ely in range(nely):
                    el = ely + (elx * nely) + elz * (nelx * nely)
                    n1 = elz * (nelx + 1) * (nely + 1) + (nely + 1) * elx + ely
                    n2 = elz * (nelx + 1) * (nely + 1) + (nely + 1) * (elx + 1) + ely
                    n3 = (elz + 1) * (nelx + 1) * (nely + 1) + (nely + 1) * elx + ely
                    n4 = (elz + 1) * (nelx + 1) * (nely + 1) + (nely + 1) * (elx + 1) + ely
                    edofMat[el, :] = np.array(
                        [3 * n1 + 3, 3 * n1 + 4, 3 * n1 + 5, 3 * n2 + 3, 3 * n2 + 4, 3 * n2 + 5, \
                         3 * n2, 3 * n2 + 1, 3 * n2 + 2, 3 * n1, 3 * n1 + 1, 3 * n1 + 2, \
                         3 * n3 + 3, 3 * n3 + 4, 3 * n3 + 5, 3 * n4 + 3, 3 * n4 + 4, 3 * n4 + 5, \
                         3 * n4, 3 * n4 + 1, 3 * n4 + 2, 3 * n3, 3 * n3 + 1, 3 * n3 + 2])
        # Construct the index pointers for the coo format
        iK = np.kron(edofMat, np.ones((24, 1))).flatten()
        jK = np.kron(edofMat, np.ones((1, 24))).flatten()

        # Filter: Build (and assemble) the index+data vectors for the coo matrix format
        nfilter = int(nelx * nely * nelz * ((2 * (np.ceil(rmin) - 1) + 1) ** 3))
        iH = np.zeros(nfilter)
        jH = np.zeros(nfilter)
        sH = np.zeros(nfilter)
        cc = 0
        for z in range(nelz):
            for i in range(nelx):
                for j in range(nely):
                    row = i * nely + j + z * (nelx * nely)
                    kk1 = int(np.maximum(i - (np.ceil(rmin) - 1), 0))
                    kk2 = int(np.minimum(i + np.ceil(rmin), nelx))
                    ll1 = int(np.maximum(j - (np.ceil(rmin) - 1), 0))
                    ll2 = int(np.minimum(j + np.ceil(rmin), nely))
                    mm1 = int(np.maximum(z - (np.ceil(rmin) - 1), 0))
                    mm2 = int(np.minimum(z + np.ceil(rmin), nelz))
                    for m in range(mm1, mm2):
                        for k in range(kk1, kk2):
                            for l in range(ll1, ll2):
                                col = k * nely + l + m * (nelx * nely)
                                fac = rmin - np.sqrt((i - k) * (i - k) + (j - l) * (j - l) + (z - m) * (z - m))
                                iH[cc] = row
                                jH[cc] = col
                                sH[cc] = np.maximum(0.0, fac)
                                cc = cc + 1
        # Finalize assembly and convert to csc format
        H = scisp.coo_matrix((sH, (iH, jH)), shape=(nelx * nely * nelz, nelx * nely * nelz)).tocsc()
        Hs = H.sum(1)

        # BC's and support
        dofs = np.arange(3 * (nelx + 1) * (nely + 1) * (nelz + 1))
        free = np.setdiff1d(dofs, fixeddof)
        # Solution and RHS vectors
        f = np.zeros((ndof, 1), dtype=float)
        u = np.zeros((ndof, 1), dtype=float)
        # Set load
        f[loaddof, 0] = -1
        # Initialize plot and plot the initial design
        '''plt.ion() # Ensure that redrawing is possible
        fig, ax = plt.subplots()
        im = ax.imshow(-xPhys.reshape((nelx, nely, nelz)).T, cmap='gray', \
                   interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
        fig.show()'''
        # Set loop counter and gradient vectors
        loop = 0
        change = 1
        dv = np.ones(nely * nelx * nelz, dtype=float)
        dc = np.ones(nely * nelx * nelz, dtype=float)
        ce = np.ones(nely * nelx * nelz, dtype=float)
        while change > tolx and loop < maxloop:
            loop = loop + 1
            # Setup and solve FE problem
            sK = ((KE.flatten()[np.newaxis]).T * (Emin + (xPhys) ** p * (Emax - Emin))).flatten(order='F')
            K = scisp.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
            # Remove constrained dofs from matrix
            K = K[free, :][:, free]
            # Solve system
            u[free, 0] = scisp.linalg.spsolve(K, f[free, 0])
            # Objective and sensitivity
            ce[:] = (np.dot(u[edofMat].reshape(nelx * nely * nelz, 24), KE) * u[edofMat].reshape(nelx * nely * nelz,
                                                                                                 24)).sum(1)
            obj = ((Emin + xPhys ** p * (Emax - Emin)) * ce).sum()
            dc[:] = (-p * xPhys ** (p - 1) * (Emax - Emin)) * ce
            dv[:] = np.ones(nely * nelx * nelz)
            # Sensitivity filtering:
            if ft == 0:
                dc[:] = np.asarray((H * (x * dc))[np.newaxis].T / Hs)[:, 0] / np.maximum(0.001, x)
            elif ft == 1:
                dc[:] = np.asarray(H * (dc[np.newaxis].T / Hs))[:, 0]
                dv[:] = np.asarray(H * (dv[np.newaxis].T / Hs))[:, 0]
            # Optimality criteria
            xold[:] = x
            (x[:], g) = oc_3_np(nelx, nely, nelz, x, volfrac, dc, dv, g)
            # Filter design variables
            if ft == 0:
                xPhys[:] = x
            elif ft == 1:
                xPhys[:] = np.asarray(H * x[np.newaxis].T / Hs)[:, 0]
            # Compute the change by the inf. norm
            change = np.linalg.norm(x.reshape(nelx * nely * nelz, 1) - xold.reshape(nelx * nely * nelz, 1), np.inf)
            # Plot to screen
            '''im.set_array(-xPhys.reshape((nelx, nely, nelz)).T)
            fig.canvas.draw()
            plt.pause(0.01)
            # Write iteration history to screen (req. Python 2.6 or newer)'''
            print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format( \
                loop, obj, (g + volfrac * nelx * nely * nelz) / (nelx * nely * nelz), change))
        '''plt.show()'''
        x1 = -xPhys[0:nelx * nely]
        # xPhys1 = xPhys.reshape((nelx,nely,nelz))
        # xPhys2 = xPhys.reshape((nelx,nely,nelz))

        # xPhys2 = np.where(xPhys1 > 0.8, 1)
        # T = xPhys1

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # scat = ax.scatter(X, Y, Z, c=T,vmin=0, vmax=1,edgecolors='none',marker="s",s=50,depthshade=False)
        # fig.colorbar(scat, shrink=0.5, aspect=5)

        fig, ax = plt.subplots()
        im = ax.imshow(x1.reshape((nelx, nely)).T, cmap='gray', \
                       interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
        plt.show()

        # mlab.clf()
        # mlab.contour3d(T,colormap='binary')
        # mlab.show()
    def MPM(self, K, P):
        pass
        # return u_wst, f_wst

    def CG(self, A, x):
        pass
        # return A_inv


"""
FeniCS based Topology Optimization with single/multi loads:
https://github.com/iitrabhi/topo-fenics
https://arxiv.org/pdf/2012.08208.pdf
"""


class TopoOpt_FeniCS():
    def __init__(self):
        self.device = 'cpu'

        self.asd = 1

    def Opt2D(self, nelx, nely, volfrac, p=3, rmin=1.5):
        print("Minimum complicance problem with OC")
        print(f"Number of degrees:{str(nelx)} x {str(nely)} = {str(nelx * nely)}")
        print(f"Volume fration: {volfrac}, Penalty p: {p}, Fileter radius: {rmin}")

        sigma = lambda _u: 2.0 * mu * df.sym(df.grad(_u)) + lmbda * df.tr(df.sym(df.grad(_u))) * df.Identity(len(_u))
        psi = lambda _u: lmbda / 2 * (df.tr(df.sym(df.grad(_u))) ** 2) + mu * df.tr(
            df.sym(df.grad(_u)) * df.sym(df.grad(_u)))
        xdmf = df.XDMFFile("output_2D/density.xdmf")
        mu, lmbda = df.Constant(0.3846), df.Constant(0.5769)
        # PREPARE FINITE ELEMENT ANALYSIS ----------------------------------
        mesh = df.RectangleMesh(df.Point(0, 0), df.Point(nelx, nely), nelx, nely, "right/left")
        U = df.VectorFunctionSpace(mesh, "P", 1)
        D = df.FunctionSpace(mesh, "DG", 0)
        u, v = df.TrialFunction(U), df.TestFunction(U)
        u_sol, density, density_old, density_new = df.Function(U), df.Function(D, name="density"), df.Function(
            D), df.Function(D)
        density.vector()[:] = volfrac
        V0 = df.assemble(1 * df.TestFunction(D) * df.dx)
        # DEFINE SUPPORT ---------------------------------------------------
        support = df.CompiledSubDomain("near(x[0], 0.0, tol) && on_boundary", tol=1e-14)
        bcs = [df.DirichletBC(U, df.Constant((0.0, 0.0)), support)]
        # DEFINE LOAD ------------------------------------------------------
        load_marker = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        df.CompiledSubDomain("x[0]==l && x[1]<=1", l=nelx).mark(load_marker, 1)
        ds = df.Measure("ds")(subdomain_data=load_marker)
        F = df.dot(v, df.Constant((0.0, -1.0))) * ds(1)
        # SET UP THE VARIATIONAL PROBLEM AND SOLVER ------------------------
        K = df.inner(density ** p * sigma(u), df.grad(v)) * df.dx
        solver = df.LinearVariationalSolver(df.LinearVariationalProblem(K, F, u_sol, bcs))
        # PREPARE DISTANCE MATRICES FOR FILTER -----------------------------
        midpoint = [cell.midpoint().array()[:] for cell in df.cells(mesh)]
        distance_mat = np.maximum(rmin - sp.euclidean_distances(midpoint, midpoint), 0)
        distance_sum = distance_mat.sum(1)
        # START ITERATION --------------------------------------------------
        loop, change = 0, 1
        while change > 0.005 and loop < 2000:
            start = time.time()
            loop = loop + 1
            density_old.assign(density)
            # FE-ANALYSIS --------------------------------------------------
            solver.solve()
            # end = time.time()
            # print('using {}'.format(end - start))
            # OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS ------------------
            objective = density ** p * psi(u_sol)
            sensitivity = df.project(-df.diff(objective, density), D).vector()[:]
            # FILTERING/MODIFICATION OF SENSITIVITIES ----------------------
            sensitivity = np.divide(distance_mat @ np.multiply(density.vector()[:], sensitivity),
                                    np.multiply(density.vector()[:], distance_sum))
            # DESIGN UPDATE BY THE OPTIMALITY CRITERIA METHOD --------------
            l1, l2, move = 0, 100000, 0.2
            while l2 - l1 > 1e-4:
                l_mid = 0.5 * (l2 + l1)
                density_new.vector()[:] = np.maximum(0.001, np.maximum(density.vector()[:] - move, np.minimum(1.0,
                                                                                                              np.minimum(
                                                                                                                  density.vector()[
                                                                                                                  :] + move,
                                                                                                                  density.vector()[
                                                                                                                  :] * np.sqrt(
                                                                                                                      -sensitivity / V0 / l_mid)))))
                current_vol = df.assemble(density_new * df.dx)
                l1, l2 = (l_mid, l2) if current_vol > volfrac * V0.sum() else (l1, l_mid)
            # PRINT RESULTS ------------------------------------------------
            change = max(density_new.vector()[:] - density_old.vector()[:])
            end = time.time()
            print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}, time: {4:.3f}".format(loop,
                                                                                               df.project(objective,
                                                                                                          D).vector().sum(),
                                                                                               current_vol / V0.sum(),
                                                                                               change, end - start))
            density.assign(density_new)
            xdmf.write(density, loop)

        p = density.vector()[:]
        file = df.File('output_2D/density.pvd')
        file << density
        df.plot(density)
        plt.show()
        asd = 1


if __name__ == "__main__":
    TTT = TopoOpt()
    p = TTT.Opt3D_NP(50, 10, 10, 0.5)
    # p = TTT.Opt2D(40, 20, 0.5)
    """
    Cupy to PyTorch
    import cupy
    import torch

    from torch.utils.dlpack import to_dlpack
    from torch.utils.dlpack import from_dlpack

    # Create a PyTorch tensor.
    tx1 = torch.randn(1, 2, 3, 4).cuda()

    # Convert it into a DLPack tensor.
    dx = to_dlpack(tx1)

    # Convert it into a CuPy array.
    cx = cupy.fromDlpack(dx)

    # Convert it back to a PyTorch tensor.
    tx2 = from_dlpack(cx.toDlpack())

    """
    #
    # TTT = TopoOpt_FeniCS()
    # p = TTT.Opt2D(150,50, 0.5)
    # start = time.time()
    # output1 = grid_sample_3d(image, grid)
    # end = time.time()
    # print('using {}'.format(end - start))