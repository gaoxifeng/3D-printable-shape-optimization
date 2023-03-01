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
    # only if r_min<2
    Ker = (r_min - 1) * torch.ones((ker_size, ker_size, ker_size))
    for i in range(ker_size):
        for j in range(ker_size):
            for k in range(ker_size):
                R = r_min - np.sqrt((i - L) ** 2 + (j - L) ** 2 + (k - L) ** 2)
                Ker[i, j, k] = torch.max(torch.tensor(R), torch.tensor(0.0))
    Ker[L, L, L] = r_min


    Ker_Sum = F.conv3d(torch.ones((1, 1, *grid.shape)).cuda(), Ker.reshape(1, 1, *Ker.shape).cuda(), stride=1, padding='same')
    Ker_Sum = Ker_Sum.reshape(*grid.shape)

    return Ker, Ker_Sum


def filter_density(Ker, grid):
    grid_in = grid.reshape(1, 1, *grid.shape)
    grid_out = F.conv3d(grid_in, Ker.reshape(1, 1, *Ker.shape), stride=1, padding='same')

    # grid_out2 = F.conv3d(grid_in, Ker.reshape(1, 1, *Ker.shape), stride=1, padding=Ker.shape[0]//2) / Ker_sum
    grid_filtered = grid_out.reshape(*grid.shape)

    return grid_filtered

# def filter_density_dcdv(Ker, grid):
#     grid_in = grid.reshape(1, 1, *grid.shape)
#     grid_out = F.conv3d(grid_in, Ker.reshape(1, 1, *Ker.shape), stride=1, padding='same')
#     # grid_out2 = F.conv3d(grid_in, Ker.reshape(1, 1, *Ker.shape), stride=1, padding=Ker.shape[0]//2) / Ker_sum
#     grid_filtered = grid_out.reshape(*grid.shape)
#     return grid_filtered

def construct_fixed(fixedid,nelx,nely,nelz):
    Fix = torch.ones((nelx + 1, nely + 1, nelz + 1))
    Fix1 = torch.zeros((nelx + 1, nely + 1, nelz + 1))
    Fix_fl = torch.clone(Fix.flatten())
    Fix_fl[fixedid.flatten()] = -1
    nn = (nelx+1)*(nely+1)
    for i in range(nelz + 1):
        temp = Fix_fl[nn * i:nn * (i + 1)]
        Fix1[:, :, i] = temp.reshape(nelx + 1, nely + 1)


    return Fix1

def oc_grid(nelx, nely, nelz, x, volfrac, dc, dv, g):
    l1 = 0.0
    l2 = 1e9
    move = 0.2
    # reshape to perform vector operations
    xnew = torch.zeros((nelx, nely, nelz))


    while (l2 - l1) / (l1 + l2) > 1e-3 and (l1 + l2) > 0:
        lmid = 0.5 * (l2 + l1)
        ppp=torch.div(-dc,dv)/ lmid
        xnew = torch.maximum(torch.tensor(0.0), torch.maximum(x - move, torch.minimum(torch.tensor(1.0),
                                                                                         torch.minimum(x + move,
                                                                                                       x * torch.sqrt(ppp)))))
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

    # def Opt3D_Grid_MG(self, rho, volfrac, p=3, rmin=1.5, maxloop=200):
    #     DTYPE = torch.float64
    #
    #     original_shape = rho.shape
    #     a1 = rho.clone().detach().cpu().numpy()
    #     nelx, nely, nelz = a1.shape
    #
    #
    #     bb = mg.BBox()
    #     bb.minC = [-1, -1, -1]
    #     bb.maxC = [1, 1, 1]
    #     res = [nelx, nely, nelz]
    #
    #
    #     print("Minimum complicance problem with OC")
    #     print(f"Number of degrees:{str(nelx)} x {str(nely)} x {str(nelz)} = {str(nelx * nely * nelz)}")
    #     print(f"Volume fration: {volfrac}, Penalty p: {p}, Fileter radius: {rmin}")
    #     maxloop = maxloop
    #     tolx = 0.01
    #     displayflag = 0
    #     nu = 0.3
    #     # max and min stiffness
    #     E_min = torch.tensor(1e-9)
    #     E_max = torch.tensor(1.0)
    #
    #     # dofs
    #     ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    #
    #     # USER - DEFINED LOAD DOFs
    #     kl = np.arange(nelz + 1)
    #     loadnid = kl * (nelx + 1) * (nely + 1) + (nely + 1) * (nelx + 1) - 1  # Node IDs
    #     loaddof = 3 * loadnid + 1  # DOFs
    #     #[{x,y,z},X,Y,Z]
    #
    #
    #     # BC's and support
    #     dofs = np.arange(3 * (nelx + 1) * (nely + 1) * (nelz + 1))
    #     # Solution and RHS vectors
    #
    #     f = torch.zeros((ndof, 1))
    #     u = torch.zeros((ndof, 1))
    #     # Set load
    #     f[loaddof, 0] = -1
    #
    #     nele = 3 * (nelx + 1) * (nely + 1)
    #     nn = (nelx + 1) * (nely + 1)
    #
    #     a = torch.zeros((3, nelx + 1, nely + 1, nelz + 1))
    #     for i in range(nelz + 1):
    #         temp = f[nele * i:nele * (i + 1), :]
    #         # print(temp)
    #         for j in range(3):
    #             pp = temp[nn * j:nn * (j + 1), :].reshape(nelx + 1, nely + 1)
    #             a[j, :, :, i] = pp
    #     aa = torch.clone(a)
    #     aa[[0, 1, 2], :, :, :] = a[[1, 2, 0], :, :, :]
    #
    #     f = torch.clone(aa)
    #     # u = torch.reshape(u, (3, nelx + 1, nely + 1, nelz + 1))
    #     loop = 0
    #     change = 1111
    #     iter = 0
    #     def phi(pos):
    #         # if phi(pos)<0, then this position is solid
    #
    #         return np.linalg.norm(pos) - 0.5
    #
    #     def phiFixed(pos):
    #         # if phiFixed(pos)<0, then this position is fixed
    #         if iter == 0:
    #             return 1
    #         else:
    #             return np.sum(pos)
    #     grid = mg.Grid(phi, phiFixed, bb, res)
    #     grid.coarsen(128)
    #     print(grid)
    #     while change > tolx and loop < maxloop:
    #         start = time.time()
    #         loop += 1
    #
    #         print(loop)
    #
    #
    #
    #         sol = mg.GridSolver(grid)
    #         sol.setupLinearSystem(100, 100)
    #         rho = rho
    #         b = f
    #         if np.max(grid.vertexType()) == 0:
    #             print("Testing free grid!")
    #             # in this case, we have to project out the 6 rigid bases
    #             b = grid.projectOutBases(b)
    #         else:
    #             print("Testing fixed grid!")
    #         sol.setRho(rho)
    #
    #         print("Solving using direct CG:")
    #         uD = torch.reshape(u, (3, nelx + 1, nely + 1, nelz + 1))
    #         sol.setB(b)
    #         sol.solveDirect(uD, 1e-8)
    #         s = grid.sensitivity(uD)
    #         print("||x||=%f ||s||=%f" % (torch.norm(uD), torch.norm(s)))
    #
    #         print("Solving using MGPCG:")
    #         uMGPCG = torch.zeros(tuple([3, res[0] + 1, res[1] + 1, res[2] + 1]), dtype=DTYPE)
    #         sol.setB(b)
    #         sol.solveMGPCG(uMGPCG, 1e-8, 1000, True, True)
    #         s = grid.sensitivity(uMGPCG)
    #         print("||x||=%f ||s||=%f" % (torch.norm(uMGPCG), torch.norm(s)))
    #
    #         print("Solving using MG:")
    #         uMG = torch.zeros(tuple([3, res[0] + 1, res[1] + 1, res[2] + 1]), dtype=DTYPE)
    #         sol.setB(b)
    #         sol.solveMG(uMG, 1e-8, 1000, True)
    #         s = grid.sensitivity(uMG)
    #         print("||x||=%f ||s||=%f" % (torch.norm(uMG), torch.norm(s)))
    #
    #
    #     return 0

    def Opt3D_Grid_MGGPU(self, rho, volfrac, p=3, rmin=1.5, maxloop=200):
        DTYPE = torch.float64

        original_shape = rho.shape
        a1 = rho.clone().detach().cpu().numpy()
        nelx, nely, nelz = a1.shape


        bb = mg.BBox()
        bb.minC = [-1, -1, -1]
        bb.maxC = [1, 1, 1]
        res = [nelx, nely, nelz]

        Ker, Ker_S = filter(rmin, rho)
        Ker = Ker.cuda()
        Ker_S = Ker_S.cuda()


        print("Minimum complicance problem with OC")
        print(f"Number of degrees:{str(nelx)} x {str(nely)} x {str(nelz)} = {str(nelx * nely * nelz)}")
        print(f"Volume fration: {volfrac}, Penalty p: {p}, Fileter radius: {rmin}")
        maxloop = maxloop
        tolx = 0.001
        # max and min stiffness
        E_min = torch.tensor(1e-9)
        E_max = torch.tensor(1.0)

        # dofs
        ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)

        # USER - DEFINED LOAD DOFs
        kl = np.arange(nelz + 1)
        loadnid = kl * (nelx + 1) * (nely + 1) + (nely + 1) * (nelx + 1) - 1  # Node IDs
        loaddof = 3 * loadnid + 1  # DOFs
        # USER - DEFINED SUPPORT FIXED DOFs
        [jf, kf] = np.meshgrid(np.arange(nely + 1), np.arange(nelz + 1))  # Coordinates
        fixednid = (kf) * (nely + 1) * (nelx + 1) + jf  # Node IDs

        dofs = np.arange(3 * (nelx + 1) * (nely + 1) * (nelz + 1))


        f = torch.zeros((ndof, 1))
        # Set load
        f[loaddof, 0] = -1

        nele = 3 * (nelx + 1) * (nely + 1)
        nn = (nelx + 1) * (nely + 1)
        a = torch.zeros((3, nelx + 1, nely + 1, nelz + 1))
        for i in range(nelz + 1):
            temp = f[nele * i:nele * (i + 1), :]
            # print(temp)
            for j in range(3):
                pp = temp[nn * j:nn * (j + 1), :].reshape(nelx + 1, nely + 1)
                a[j, :, :, i] = pp
        aa = torch.clone(a)
        aa[[0, 1, 2], :, :, :] = a[[1, 2, 0], :, :, :]

        f = torch.clone(aa)
        # u = torch.reshape(u, (3, nelx + 1, nely + 1, nelz + 1))
        loop = 0
        g=0
        change = 1111


        mg.initializeGPU()

        phiTensor = -torch.ones_like(rho).cuda()
        phiFixedTensor = construct_fixed(fixednid,nelx,nely,nelz).cuda()
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
            sol.setupLinearSystem(0.3/0.52, 1/2.6)

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
            # dc = (p * rho ** (p - 1) * (E_max - E_min)) * dc

            dc = (p * rho_p ** (p - 1) * (E_max - E_min)) * dc

            dc = filter_density(Ker, dc/Ker_S)
            # dc = filter_density(Ker, Ker_S, dc*rho) / torch.maximum(torch.tensor(0.001), rho)
            dv = torch.ones((nelx, nely, nelz)).cuda()
            dv = filter_density(Ker, dv/Ker_S)

            rho_old = torch.clone(rho)
            (rho, g) = oc_grid(nelx, nely, nelz, rho, volfrac, dc, dv, g)

            change = torch.linalg.norm(rho.reshape(-1,1) - rho_old.reshape(-1,1), ord=float('inf'))
            end = time.time()
            print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}, time: {4:.3f}".format(loop, obj, (g + volfrac * nelx * nely * nelz) / (nelx * nely * nelz), change, end - start))
            rho_p = filter_density(Ker, rho) / Ker_S
            # rho = filter_density_dcdv(Ker, rho / Ker_S)
            # a=1

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


    def Opt3D_Grid_MGGPU_Bridge(self, rho, volfrac, p=3, rmin=1.5, maxloop=200):
        DTYPE = torch.float64

        original_shape = rho.shape
        a1 = rho.clone().detach().cpu().numpy()
        nelx, nely, nelz = a1.shape


        bb = mg.BBox()
        bb.minC = [-1, -1, -1]
        bb.maxC = [1, 1, 1]
        res = [nelx, nely, nelz]

        Ker, Ker_S = filter(rmin, rho)
        Ker = Ker.cuda()
        Ker_S = Ker_S.cuda()


        print("Minimum complicance problem with OC")
        print(f"Number of degrees:{str(nelx)} x {str(nely)} x {str(nelz)} = {str(nelx * nely * nelz)}")
        print(f"Volume fration: {volfrac}, Penalty p: {p}, Fileter radius: {rmin}")
        maxloop = maxloop
        tolx = 0.002
        # max and min stiffness
        E_min = torch.tensor(1e-9)
        E_max = torch.tensor(1.0)

        # dofs
        ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)

        # # USER - DEFINED LOAD DOFs
        # kl = np.arange(nelz + 1)
        # loadnid = kl * (nelx + 1) * (nely + 1) + (nely + 1) * (nelx + 1) - 1  # Node IDs
        # loaddof = 3 * loadnid + 1  # DOFs
        # # USER - DEFINED SUPPORT FIXED DOFs
        # [jf, kf] = np.meshgrid(np.arange(nely + 1), np.arange(nelz + 1))  # Coordinates
        # fixednid = (kf) * (nely + 1) * (nelx + 1) + jf  # Node IDs
        #
        # dofs = np.arange(3 * (nelx + 1) * (nely + 1) * (nelz + 1))
        #
        #
        # f = torch.zeros((ndof, 1))
        # # Set load
        # f[loaddof, 0] = -1
        #
        # nele = 3 * (nelx + 1) * (nely + 1)
        # nn = (nelx + 1) * (nely + 1)
        # a = torch.zeros((3, nelx + 1, nely + 1, nelz + 1))
        # for i in range(nelz + 1):
        #     temp = f[nele * i:nele * (i + 1), :]
        #     # print(temp)
        #     for j in range(3):
        #         pp = temp[nn * j:nn * (j + 1), :].reshape(nelx + 1, nely + 1)
        #         a[j, :, :, i] = pp
        # aa = torch.clone(a)
        # aa[[0, 1, 2], :, :, :] = a[[1, 2, 0], :, :, :]
        #
        # f = torch.clone(aa)
        f = torch.zeros((3, nelx + 1, nely + 1, nelz + 1))
        f[1,9:29,44,:] = -2
        # u = torch.reshape(u, (3, nelx + 1, nely + 1, nelz + 1))
        loop = 0
        g=0
        change = 1111


        mg.initializeGPU()

        phiTensor = -torch.ones_like(rho).cuda()
        phiTensor[9:29,44:69,:] = 1
        # phiFixedTensor = construct_fixed(fixednid,nelx,nely,nelz).cuda()
        phiFixedTensor = torch.ones((nelx + 1, nely + 1, nelz + 1)).cuda()
        phiFixedTensor[:,0,57:61] = -1
        phiFixedTensor[:, 0, 297:301] = -1
        # phiFixedTensor[:, 40:44, :] = -1

        grid = mg.GridGPU(phiTensor, phiFixedTensor, bb)
        grid.coarsen(128)
        print(grid)
        rho[:, 40:44, :]=1
        rho_p = torch.clone(rho)
        while change > tolx and loop < maxloop:
            start = time.time()
            loop += 1
            rho1 = (E_min + rho_p ** p * (E_max - E_min))
            uMGPCG = torch.zeros((3, nelx + 1, nely + 1, nelz + 1)).cuda()

            sol = mg.GridSolverGPU(grid)
            sol.setupLinearSystem(0.6, 0.4)

            b = f.cuda()
            if grid.isFree():
                # print("Testing free grid!")
                # in this case, we have to project out the 6 rigid bases
                b = grid.projectOutBases(b)
            else:
                print("Testing fixed grid!")
            print(f'Set up time: {time.time()-start}')

            sol.setB(b)
            sol.solveMGPCG(rho1, uMGPCG, 1e-8, 1000, True, False)
            print(f'Solve time: {time.time() - start}')
            dc = grid.sensitivity(uMGPCG)
            dc[:, 40:44, :] = 0
            obj = -torch.sum((E_min + rho_p ** p * (E_max - E_min)) * dc)
            # dc = (p * rho ** (p - 1) * (E_max - E_min)) * dc

            dc = (p * rho_p ** (p - 1) * (E_max - E_min)) * dc

            dc = filter_density(Ker, dc/Ker_S)
            # dc = filter_density(Ker, Ker_S, dc*rho) / torch.maximum(torch.tensor(0.001), rho)
            dv = torch.ones((nelx, nely, nelz)).cuda()
            dv = filter_density(Ker, dv/Ker_S)

            rho_old = torch.clone(rho)
            (rho, g) = oc_grid(nelx, nely, nelz, rho, volfrac, dc, dv, g)
            print(f'update time: {time.time() - start}')
            rho[:, 40:44, :] = 1
            change = torch.linalg.norm(rho.reshape(-1,1) - rho_old.reshape(-1,1), ord=float('inf'))
            end = time.time()
            print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}, time: {4:.3f}".format(loop, obj, (g + volfrac * nelx * nely * nelz) / (nelx * nely * nelz), change, end - start))
            rho_p = filter_density(Ker, rho) / Ker_S
            rho_p[:, 40:44, :] = 1
            # rho = filter_density_dcdv(Ker, rho / Ker_S)
            # a=1

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




if __name__ == "__main__":
    x = 0.3*torch.ones((60,20,4),dtype=torch.float64).cuda()
    # x[0,0,1:5]=1
    TTT = TopoOpt(0.3)
    # p = TTT.Opt3D_NP(60, 20, 4, 0.3, 3, 1.5, 10)
    # p = TTT.Opt3D_Grid_MGGPU(x, 0.3, 3, 1.5, 50)

    bridge = 0.15 * torch.ones((40,90,360),dtype=torch.float64).cuda()
    p = TTT.Opt3D_Grid_MGGPU_Bridge(bridge, 0.15, 3, 10, 500)