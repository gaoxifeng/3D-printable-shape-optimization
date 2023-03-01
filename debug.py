import torch
import numpy as np
import time
import libMG as mg
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib import colors
torch.set_default_dtype(torch.float64)


def filter(r_min, grid):
    L = int(np.floor(r_min))
    ker_size = 2 * L + 1
    Ker = (r_min - 1) * torch.ones((ker_size, ker_size, ker_size))
    for i in range(ker_size):
        for j in range(ker_size):
            for k in range(ker_size):
                R = r_min - np.sqrt((i - L) ** 2 + (j - L) ** 2 + (k - L) ** 2)
                Ker[i, j, k] = torch.max(torch.tensor(R), torch.tensor(0.0))

    Ker[L, L, L] = r_min
    Ker_Sum = F.conv3d(torch.ones((1, 1, *grid.shape)).cuda(), Ker.reshape(1, 1, *Ker.shape).cuda(), stride=1, padding='same')
    Ker_Sum = Ker_Sum.reshape(*grid.shape)

    return Ker.cuda(), Ker_Sum


def filter_density(Ker, grid):
    grid_in = grid.reshape(1, 1, *grid.shape)
    grid_out = F.conv3d(grid_in, Ker.reshape(1, 1, *Ker.shape), stride=1, padding='same')
    grid_filtered = grid_out.reshape(*grid.shape)

    return grid_filtered

def oc_grid(nelx, nely, nelz, x, dc, dv, g, Nonphi1, Nonphi2, Targetphi):
    l1 = 0.0
    l2 = 1e9
    move = 0.2
    # reshape to perform vector operations
    xnew = torch.zeros((nelx, nely, nelz))
    dc = dc * Nonphi1

    while (l2 - l1) / (l1 + l2) > 1e-3 and (l1 + l2) > 0:
        lmid = 0.5 * (l2 + l1)
        ppp=torch.div(-dc,dv)/ lmid
        xnew = torch.maximum(torch.tensor(0.0), torch.maximum(x - move, torch.minimum(torch.tensor(1.0),
                                                                                         torch.minimum(x + move,
                                                                                                 x * torch.sqrt(ppp)))))
        xnew = xnew * Nonphi1 + Targetphi * Nonphi2
        gt = g + torch.sum((dv * (xnew - x)))
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
        # print(l1, l2, l1+l2)
    return (xnew, gt)

class TopoOpt():
    def __init__(self, vol):
        # self.device = 'cpu'
        self.volfrac = vol
        self.asd = 1

    def Opt3D_Grid_MGGPU(self, rho, paras,  p=3, rmin=1.5, maxloop=200):
        a1 = rho.clone().detach().cpu().numpy()
        nelx, nely, nelz = a1.shape

        phiTensor, phiFixedTensor, f, Nonphi1, Nonphi2, Targetphi, lam, mu = paras


        bb = mg.BBox()
        bb.minC = [-1, -1, -1]
        bb.maxC = [1, 1, 1]
        Ker, Ker_S = filter(rmin, rho)

        print("Minimum complicance problem with OC")
        print(f"Number of degrees:{str(nelx)} x {str(nely)} x {str(nelz)} = {str(nelx * nely * nelz)}")
        print(f"Volume fration: {self.volfrac}, Penalty p: {p}, Fileter radius: {rmin}")
        maxloop = maxloop
        tolx = 0.001
        # max and min stiffness
        E_min = torch.tensor(1e-9)
        E_max = torch.tensor(1.0)
        loop = 0
        g=0
        change = 1

        mg.initializeGPU()
        grid = mg.GridGPU(phiTensor, phiFixedTensor, bb)
        grid.coarsen(128)
        print(grid)
        rho_p = torch.clone(rho)
        while change > tolx and loop < maxloop:
            start = time.time()
            loop += 1
            rho1 = (E_min + rho_p ** p * (E_max - E_min))
            uMGPCG = torch.zeros((3, nelx + 1, nely + 1, nelz + 1)).cuda()

            sol = mg.GridSolverGPU(grid)
            sol.setupLinearSystem(lam, mu)

            b = f.cuda()
            if grid.isFree():
                # print("Testing free grid!")
                # in this case, we have to project out the 6 rigid bases
                b = grid.projectOutBases(b)
            else:
                print("Testing fixed grid!")


            sol.setB(b)
            sol.solveMGPCG(rho1, uMGPCG, 1e-8, 1000, True, False)
            dc = grid.sensitivity(uMGPCG)

            obj = -torch.sum((E_min + rho_p ** p * (E_max - E_min)) * dc)
            dc = (p * rho_p ** (p - 1) * (E_max - E_min)) * dc
            dc = filter_density(Ker, dc/Ker_S)
            # dc = filter_density(Ker, Ker_S, dc*rho) / torch.maximum(torch.tensor(0.001), rho)
            dv = torch.ones((nelx, nely, nelz)).cuda()
            dv = filter_density(Ker, dv/Ker_S)

            rho_old = torch.clone(rho)
            (rho, g) = oc_grid(nelx, nely, nelz, rho, dc, dv, g, Nonphi1, Nonphi2, Targetphi)

            change = torch.linalg.norm(rho.reshape(-1,1) - rho_old.reshape(-1,1), ord=float('inf'))
            end = time.time()
            print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}, time: {4:.3f}".format(loop, obj, (g + self.volfrac * nelx * nely * nelz) / (nelx * nely * nelz), change, end - start))
            rho_p = filter_density(Ker, rho) / Ker_S
        mg.finalizeGPU()
        xPhys = rho_old.detach().cpu().numpy()


        fig, ax = plt.subplots()
        im = ax.imshow(-xPhys[:,:,1].T, cmap='gray', \
                       interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
        plt.show()
        #
        import mayavi.mlab as mlab
        mlab.clf()
        mlab.contour3d(xPhys,colormap='binary')
        mlab.show()

        return xPhys

    def Toy_Example(self,rho):
        nelx, nely, nelz = rho.clone().detach().cpu().numpy().shape
        phiTensor = -torch.ones_like(rho).cuda()
        phiFixedTensor=torch.ones((nelx + 1, nely + 1, nelz + 1)).cuda()
        phiFixedTensor[0,:,:]=-1
        f = torch.zeros((3, nelx + 1, nely + 1, nelz + 1)).cuda()
        f[1, -1, 0, :] = -1
        Nonphi1 = torch.ones_like(rho)
        Nonphi2 = torch.zeros_like(rho)
        Targetphi = torch.ones_like(rho)
        lam = 0.3 / 0.52
        mu = 1 / 2.6
        return phiTensor, phiFixedTensor, f, Nonphi1, Nonphi2, Targetphi, lam, mu

    def Bridge_Example(self,rho):
        nelx, nely, nelz = rho.clone().detach().cpu().numpy().shape
        phiTensor = -torch.ones_like(rho).cuda()
        phiTensor[9:29,44:69,:] = 1

        phiFixedTensor=torch.ones((nelx + 1, nely + 1, nelz + 1)).cuda()
        phiFixedTensor[:,0,57:61] = -1
        phiFixedTensor[:, 0, 297:301] = -1

        f = torch.zeros((3, nelx + 1, nely + 1, nelz + 1)).cuda()
        f[1,9:29,44,:] = -2

        Nonphi1 = torch.ones_like(rho)
        Nonphi1[:, 40:44, :] = 0
        Nonphi2 = torch.zeros_like(rho)
        Nonphi2[:, 40:44, :] = 1
        Targetphi = torch.tensor(1.0)


        lam = 0.6
        mu = 0.4
        return phiTensor, phiFixedTensor, f, Nonphi1, Nonphi2, Targetphi, lam, mu




if __name__ == "__main__":
    x = 0.15*torch.ones((40,90,360),dtype=torch.float64).cuda()
    TTT = TopoOpt(0.15)
    paras = TTT.Bridge_Example(x)
    p = TTT.Opt3D_Grid_MGGPU(x, paras, 3, 10, 50)

    # bridge = 0.15 * torch.ones((40,90,360),dtype=torch.float64).cuda()
    # p = TTT.Opt3D_Grid_MGGPU_Bridge(bridge, 0.15, 3, 10, 500)