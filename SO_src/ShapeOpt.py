import torch, time
import numpy as np
import libMG as mg
from SOLayer import SOLayer
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


class ShapeOpt():
    def __init__(self, s, rmin=1.5, maxloop=200, maxloopLinear=1000, tolx=1e-3, tolLinear=1e-2,
                 outputInterval=1, eps_Heaviside=1e-3, outputDetail=False):
        # self.device = 'cpu'
        self.s = s
        self.eps = eps_Heaviside
        self.rmin = rmin
        self.maxloop = maxloop
        self.maxloopLinear = maxloopLinear
        self.tolx = tolx
        self.tolLinear = tolLinear
        self.outputInterval = outputInterval
        self.outputDetail = outputDetail

    def run(self, sdf, phiTensor, phiFixedTensor, f, rhoMask, lam, mu, volfrac):
        nelx, nely, nelz = sdf.shape
        bb = mg.BBox()
        bb.minC = [0, 0, 0]
        bb.maxC = [nelx, nely, nelz]
        Ker, Ker_S = ShapeOpt.filter(self.rmin, sdf)
        # Prepare the CFL for updating the level set function
        CFL = 1

        print("Minimum shape optimization problem with OC")
        print(f"Number of degrees: {str(nelx)} x {str(nely)} x {str(nelz)} = {str(nelx * nely * nelz)}")
        # print(f"Volume fration: {self.volfrac}, Penalty p: {self.p}")
        # max and min stiffness
        change = self.tolx * 2
        loop = 0

        # initialize torch layer
        mg.initializeGPU()

        SOLayer.reset(phiTensor, phiFixedTensor, f, bb, lam, mu, self.tolLinear, self.maxloopLinear, self.outputDetail)
        #Generate a sdf with holes to indicate some parts with negative holes
        #And use viewer to visualize the generated sdf
        sdf = SOLayer.redistance(sdf)

        while change > self.tolx and loop < self.maxloop:
            start = time.time()
            loop += 1

            # compute filtered volume gradient
            sdf = sdf.detach()
            sdf.requires_grad_()
            H = ShapeOpt.Heaviside(sdf, self.eps, self.s)
            H_filtered = ShapeOpt.filter_density(Ker, H) / Ker_S
            vol = torch.sum(H_filtered)
            print(torch.sum(H_filtered).item() / (nelx * nely * nelz))
            vol.backward()
            gradVol = (self.s * torch.exp(-self.s * sdf) / (1 + torch.exp(-self.s * sdf)) ** 2)
            # gradVol = sdf.grad.detach()

            # compute filtered objective gradient
            sdf = sdf.detach()
            sdf.requires_grad_()
            H = ShapeOpt.Heaviside(sdf, self.eps, self.s)
            H_filtered = ShapeOpt.filter_density(Ker, H) / Ker_S
            obj = SOLayer.apply(H_filtered)
            obj.backward()
            gradObj = sdf.grad.detach()

            # Find Lagrange Lambda and update shape
            H_old = H.clone()
            sdf, vol = ShapeOpt.find_lambda(sdf, self.s, self.eps, gradObj, gradVol, volfrac, CFL, Ker, Ker_S)
            sdf = SOLayer.redistance(sdf)
            H = ShapeOpt.Heaviside(sdf, self.eps, self.s)

            change = torch.linalg.norm(H.reshape(-1, 1) - H_old.reshape(-1, 1), ord=float('inf')).item()
            end = time.time()
            if loop % self.outputInterval == 0:
                print("it.: {0}, obj.: {1:.3f}, vol.: {2:.3f}, ch.: {3:.3f}, time: {4:.3f}, mem: {4:.3f}Gb".
                      format(loop, obj, vol, change, end - start, torch.cuda.memory_allocated(None) / 1024 / 1024 / 1024))
        mg.finalizeGPU()
        return sdf.detach().cpu().numpy()

    def find_lambda(sdf0, s, eps, gradObj, gradVol, volfrac, CFL, Ker, Ker_S):
        nelx, nely, nelz = sdf0.shape
        # C = (nelx + 1) * (nely + 1) * (nelz + 1) / torch.sum(gradObj)
        V = nelx * nely * nelz

        def compute_volume(lam, sdf0):
            V_N = 1 * gradObj + lam * gradVol
            sdf = sdf0 - V_N * CFL/torch.max(V_N)
            # print(lam, torch.norm(sdf))
            H = ShapeOpt.Heaviside(sdf, eps, s)
            H_filtered = ShapeOpt.filter_density(Ker, H) / Ker_S
            return sdf, torch.sum(H_filtered).item() / V

        l1 = -1e9
        l2 = 1e9
        print(f'Max possible volfrac {compute_volume(l1, sdf0)[1]}, Min possible volfrac {compute_volume(l2, sdf0)[1]}.')
        # if volfrac > compute_volume(l1,sdf0)[1]:
        #     print(f'Max possible volfrac {compute_volume(l1,sdf0)[1]} is smaller than {volfrac}.')
        #     exit(-1)
        # reshape to perform vector operations
        while l2-l1>1e-3:
            lmid = 0.5 * (l2 + l1)
            # print(torch.min(sdf0))
            sdf, vol = compute_volume(lmid, sdf0)
            # print(torch.min(sdf))
            if vol > volfrac:
                l1 = lmid
            else:
                l2 = lmid
            # print(lmid, vol)
        return sdf, vol

    def Heaviside(sdf, eps, s):
        # phi_s = s*torch.exp(-s*sdf) / (1+torch.exp(-s*sdf))**2
        Phi_s = 1 / (1 + torch.exp(-s * sdf))
        H = (eps + 1) - Phi_s
        return H

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
        Ker_Sum = F.conv3d(torch.ones((1, 1, *grid.shape)).cuda(), Ker.reshape(1, 1, *Ker.shape).cuda(), stride=1,
                           padding='same')
        Ker_Sum = Ker_Sum.reshape(*grid.shape)
        return Ker.cuda(), Ker_Sum

    def filter_density(Ker, grid):
        grid_in = grid.reshape(1, 1, *grid.shape)
        grid_out = F.conv3d(grid_in, Ker.reshape(1, 1, *Ker.shape), stride=1, padding='same')
        grid_filtered = grid_out.reshape(*grid.shape)
        return grid_filtered
# if __name__ == "__main__":
#     #Testing debug function for find_lambda:
#     SO = ShapeOpt(0)
#     gradO = 10*torch.ones((3,3,3)).cuda()
#     gradV = torch.ones_like(gradO)
#     CFL = 0.01
#     A = torch.randn(3,3,3)
#     Ker, Ker_S = TopoOpt.filter(1.5, A)
#
#     A1 = SO.find_lambda(A, 0.5, eps=1e-3, gradObj=gradO, gradVol=gradV, volfrac=0.5, CFL=CFL, Ker=Ker, Ker_S=Ker_S)

