import torch, time
import numpy as np
import libMG as mg
from SOLayer import SOLayer
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


class ShapeOpt():
    def __init__(self, volfrac, p=3, rmin = 1.5, maxloop=200, maxloopLinear=1000, tolx=1e-3, tolLinear=1e-2,
                 outputInterval=1, eps_Heaviside = 1e-3, outputDetail=False):
        # self.device = 'cpu'
        self.volfrac = volfrac
        self.p = p
        self.eps = eps_Heaviside
        self.rmin = rmin
        self.maxloop = maxloop
        self.maxloopLinear = maxloopLinear
        self.tolx = tolx
        self.tolLinear = tolLinear
        self.outputInterval = outputInterval
        self.outputDetail = outputDetail

    def run(self, sdf, s, phiTensor, phiFixedTensor, f, rhoMask, lam, mu):
        nelx, nely, nelz = sdf.shape
        bb = mg.BBox()
        bb.minC = [-1, -1, -1]
        bb.maxC = [1, 1, 1]
        #Prepare the deltaT for updating the level set function
        deltaT = 2 / torch.max(torch.tensor([nelx,nely,nelz]))

        print("Minimum complicance problem with OC")
        print(f"Number of degrees:{str(nelx)} x {str(nely)} x {str(nelz)} = {str(nelx * nely * nelz)}")
        print(f"Volume fration: {self.volfrac}, Penalty p: {self.p}")
        # max and min stiffness
        change = self.tolx * 2
        loop = 0
        Lag_lambda = 10

        # initialize torch layer
        mg.initializeGPU()
        SOLayer.reset(phiTensor, phiFixedTensor, f, bb, lam, mu, self.tolLinear, self.maxloopLinear, self.outputDetail)
        while change > self.tolx and loop < self.maxloop:
            start = time.time()
            loop += 1

            # compute filtered objective gradient
            sdf = sdf.detach()
            sdf.requires_grad_()
            H = ShapeOpt.Heaviside(sdf, self.eps, s)
            obj = SOLayer.apply(H)
            obj.backward()
            gradObj = sdf.grad.detach()

            #Find Lagrange Lambda
            Lag_lambda, g = ShapeOpt.Find_Lambda(sdf, s, gradObj, Lag_lambda, self.volfrac, deltaT, self.eps, self.rmin, 10)
            #Update and evaluate V_N
            V_N = gradObj + Lag_lambda*(s*torch.exp(-s*sdf) / (1+torch.exp(-s*sdf))**2)
            Ker, Ker_S = ShapeOpt.filter(self.rmin, V_N)
            V_N_filtered = ShapeOpt.filter_density(Ker, V_N / Ker_S)

            #Updata Level Set function Phi_s
            sdf_update = -V_N_filtered*deltaT + sdf

            #Redistance and evaluate kappa
            sdf=SOLayer.redistance(sdf_update)
            H_new = ShapeOpt.Heaviside(sdf, self.eps, s)

            change = torch.linalg.norm(H_new.reshape(-1, 1) - H.reshape(-1, 1), ord=float('inf')).item()
            end = time.time()
            if loop % self.outputInterval == 0:
                print("it.: {0}, obj.: {1:.3f}, vol.: {2:.3f}, ch.: {3:.3f}, time: {4:.3f}, mem: {4:.3f}Gb".format(loop,
                                                                                                                   obj,
                                                                                                                   g,
                                                                                                                   change,
                                                                                                                   end - start,
                                                                                                                   torch.cuda.memory_allocated(
                                                                                                                       None) / 1024 / 1024 / 1024))

        mg.finalizeGPU()
        return sdf.detach().cpu().numpy()


    def Find_Lambda(sdf, s, gradObj, lam_old, volfrac, deltaT, eps, rmin, max_iter):
        lam_l = 0.0
        lam_r = lam_old
        H = ShapeOpt.Heaviside(sdf, eps, s)
        nelx, nely, nelz = H.shape

        # evaluate the volume of H and then use binary search to find lamda
        C = (nelx+1)*(nely+1)*(nelz+1) / torch.sum(gradObj)
        delta = (s*torch.exp(-s*sdf) / (1+torch.exp(-s*sdf))**2) #\partial H / \partial phi
        Ker, Ker_S = ShapeOpt.filter(rmin, sdf)

        for k in range(1, max_iter+1):
            lam = (lam_l+lam_r)/2
            V_N = C*gradObj + lam*delta
            V_N_filtered = ShapeOpt.filter_density(Ker, V_N / Ker_S)
            Update = -V_N_filtered*deltaT
            sdf_update = Update + sdf
            H_Vol_update = ShapeOpt.Vol(ShapeOpt.Heaviside(sdf_update, eps, s))
            g = torch.sum(H_Vol_update)/(nelx*nely*nelz)
            if g - volfrac < 0:
                lam_l = lam
            else:
                lam_r = lam
            if torch.abs(g - volfrac) < 1e-4:
                break
            # print(f'lam left:{lam_l}, lam right:{lam_r}, Vol = {H_Vol_update}, g = {g}')
        return lam, g


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

    def Heaviside(sdf, eps, s):
        # phi_s = s*torch.exp(-s*sdf) / (1+torch.exp(-s*sdf))**2
        Phi_s = 1 / (1 + torch.exp(-s * sdf))
        H = eps + 1 - Phi_s
        return H

    def Vol(H):
        filter = torch.ones(1, 1, 3, 3, 3).cuda() / 27
        H_Vol = F.conv3d(H.reshape(1, 1, *H.shape), filter, stride=1,
                           padding='same')
        return H_Vol

# if __name__ == "__main__":
