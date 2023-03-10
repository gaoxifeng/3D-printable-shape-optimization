import torch,time
import numpy as np
import libMG as mg
from TOLayer import TOLayer
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)

class TopoOpt():
    def __init__(self, *, volfrac, p=3, rmin=5, maxloop=200, maxloopLinear=1000, tolx=1e-3, tolLinear=1e-2, outputInterval=1, outputDetail=False):
        # self.device = 'cpu'
        self.volfrac = volfrac
        self.p = p
        self.rmin = rmin
        self.maxloop = maxloop
        self.maxloopLinear = maxloopLinear
        self.tolx = tolx
        self.tolLinear = tolLinear
        self.outputInterval = outputInterval
        self.outputDetail = outputDetail

    def run(self, rho, phiTensor, phiFixedTensor, f, rhoMask, lam, mu):
        nelx, nely, nelz = rho.shape
        bb = mg.BBox()
        bb.minC = [0,0,0]
        bb.maxC = [nelx,nely,nelz]
        Ker, Ker_S = TopoOpt.filter(self.rmin, rho)

        print("Minimum complicance problem with OC")
        print(f"Number of degrees:{str(nelx)} x {str(nely)} x {str(nelz)} = {str(nelx * nely * nelz)}")
        print(f"Volume fration: {self.volfrac}, Penalty p: {self.p}, Filter radius: {self.rmin}")
        # max and min stiffness
        E_min = torch.tensor(1e-3)
        E_max = torch.tensor(1.0)
        change = self.tolx*2
        loop = 0
        g=0

        #compute filtered volume gradient (this is contant so we can precompute)
        rho = rho.detach()
        rho.requires_grad_()
        rho_filtered = TopoOpt.filter_density(Ker, rho)/Ker_S
        volume = torch.sum(rho_filtered)
        volume.backward()
        gradVolume = rho.grad.detach()
            
        #initialize torch layer
        mg.initializeGPU()
        TOLayer.reset(phiTensor, phiFixedTensor, f, bb, lam, mu, self.tolLinear, self.maxloopLinear, self.outputDetail)
        while change > self.tolx and loop < self.maxloop:
            start = time.time()
            loop += 1

            #compute filtered objective gradient
            rho = rho.detach()
            rho.requires_grad_()
            rho_filtered = TopoOpt.filter_density(Ker, rho)/Ker_S
            obj = TOLayer.apply(E_min + rho_filtered ** self.p * (E_max - E_min))
            obj.backward()
            gradObj = rho.grad.detach()
            
            rho_old = rho.clone()
            rho, g = TopoOpt.oc_grid(rho, gradObj, gradVolume, g, rhoMask)
            change = torch.linalg.norm(rho.reshape(-1,1) - rho_old.reshape(-1,1), ord=float('inf')).item()
            end = time.time()
            if loop%self.outputInterval == 0:
                print("it.: {0}, obj.: {1:.3f}, vol.: {2:.3f}, ch.: {3:.3f}, time: {4:.3f}, mem: {4:.3f}Gb".format(loop, obj, (g + self.volfrac * nelx * nely * nelz) / (nelx * nely * nelz), change, end - start, torch.cuda.memory_allocated(None)/1024/1024/1024))
        
        mg.finalizeGPU()
        return rho_old.detach().cpu().numpy()
    
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
    
    def oc_grid(x, gradObj, gradVolume, g, rhoMask):
        l1 = 0.0
        l2 = 1e9
        move = 0.2
        eta = 0.5
        # reshape to perform vector operations
        while (l2 - l1) / (l1 + l2) > 1e-3 and (l1 + l2) > 0:
            lmid = 0.5 * (l2 + l1)
            Be_eta = (torch.maximum( torch.tensor(0.0), torch.div(-gradObj, gradVolume) / lmid ) ** eta).detach()
            xnew = (torch.maximum(torch.tensor(0.0), torch.maximum(x - move, torch.minimum(torch.tensor(1.0), torch.minimum(x + move, x * Be_eta))))).detach()
            rhoMask(xnew)
            gt = g + torch.sum((gradVolume * (xnew - x))).item()
            if gt > 0:
                l1 = lmid
            else:
                l2 = lmid
        return (xnew, gt)
