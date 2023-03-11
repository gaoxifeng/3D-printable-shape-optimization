import torch, time
import numpy as np
import libMG as mg
from TopoOpt import TopoOpt
from TOLayer import TOLayer
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)

class ShapeOpt():
    def __init__(self, s, rmin=1.5, maxloop=200, maxloopLinear=1000, tolx=1e-3, tolLinear=1e-2, outputInterval=1, eps_Heaviside=1e-3, outputDetail=False):
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

    def round_up(n, blk):
        return (n+blk-1)//blk

    def fill_block(sdf, x, y, z, blk, ratio):
        nelx, nely, nelz = sdf.shape
        x *= blk
        y *= blk
        z *= blk
        rad = blk/2*ratio
        for xx in range(x, min(x+blk, nelx)):
            for yy in range(y, min(y+blk, nely)):
                for zz in range(z, min(z+blk, nelz)):
                    dir = np.array([xx-x-blk/2, yy-y-blk/2, zz-z-blk/2])
                    sdf[xx,yy,zz]=rad-np.linalg.norm(dir)

    def create_bubble(sdf, blk, ratio=.7):
        nelx, nely, nelz = sdf.shape
        for x in range(0, ShapeOpt.round_up(nelx, blk)):
            for y in range(0, ShapeOpt.round_up(nely, blk)):
                for z in range(0, ShapeOpt.round_up(nelz, blk)):
                    ShapeOpt.fill_block(sdf, x, y, z, blk, ratio)
        return sdf

    def run(self, sdf, phiTensor, phiFixedTensor, f, rhoMask, lam, mu):
        nelx, nely, nelz = sdf.shape
        bb = mg.BBox()
        bb.minC = [0, 0, 0]
        bb.maxC = [nelx, nely, nelz]
        Ker, Ker_S = TopoOpt.filter(self.rmin, sdf)

        print("Minimum shape optimization problem with OC")
        print(f"Number of degrees: {str(nelx)} x {str(nely)} x {str(nelz)} = {str(nelx * nely * nelz)}")
        # print(f"Volume fration: {self.volfrac}, Penalty p: {self.p}")
        # max and min stiffness
        change = self.tolx * 2
        loop = 0
        # Prepare the CFL for updating the level set function
        CFL = 1

        # initialize torch layer
        mg.initializeGPU()
        TOLayer.reset(phiTensor, phiFixedTensor, f, bb, lam, mu, self.tolLinear, self.maxloopLinear, self.outputDetail)
        sdf = TOLayer.redistance(sdf)
        while change > self.tolx and loop < self.maxloop:
            start = time.time()
            loop += 1

            # compute filtered volume gradient
            sdf = sdf.detach()
            sdf.requires_grad_()
            H = ShapeOpt.Heaviside(sdf, self.eps, self.s)
            #H_filtered = TopoOpt.filter_density(Ker, H) / Ker_S
            #vol = torch.sum(H_filtered)
            vol = torch.sum(H)
            vol.backward()
            gradVol = sdf.grad.detach()
            
            # use given volume as constraint
            if loop==1:
                volTarget = vol.item()

            # compute filtered objective gradient
            sdf = sdf.detach()
            sdf.requires_grad_()
            H = ShapeOpt.Heaviside(sdf, self.eps, self.s)
            H_filtered = TopoOpt.filter_density(Ker, H) / Ker_S
            obj = TOLayer.apply(H_filtered)
            obj.backward()
            gradObj = sdf.grad.detach()

            # Find Lagrange Lambda and update shape
            H_old = H.clone()
            sdf, vol = ShapeOpt.oc_grid(sdf, self.s, self.eps, gradObj, gradVol, volTarget, CFL, Ker, Ker_S)
            sdf = TOLayer.redistance(sdf)
            H = ShapeOpt.Heaviside(sdf, self.eps, self.s)
            change = torch.linalg.norm(H.reshape(-1, 1) - H_old.reshape(-1, 1), ord=float('inf')).item()
            end = time.time()
            if loop % self.outputInterval == 0:
                print("it.: {0}, obj.: {1:.3f}, vol.: {2:.3f}, ch.: {3:.3f}, time: {4:.3f}, mem: {4:.3f}Gb".
                      format(loop, obj, vol, change, end - start, torch.cuda.memory_allocated(None) / 1024 / 1024 / 1024))
        mg.finalizeGPU()
        return sdf.detach().cpu().numpy()
    
    def Heaviside(sdf, eps, s):
        # phi_s = s*torch.exp(-s*sdf) / (1+torch.exp(-s*sdf))**2
        Phi_s = 1 / (1 + torch.exp(-s * sdf))
        H = (eps + 1) - Phi_s
        return H
    
    def oc_grid(sdf0, s, eps, gradObj, gradVol, volTarget, CFL, Ker, Ker_S):
        def compute_volume(lam, sdf0):
            V_N = (gradObj + lam * gradVol).detach()
            scale = torch.minimum(V_N / (torch.max(torch.abs(V_N)) + 1e-6), torch.tensor(1.0))
            sdf = sdf0 - CFL * scale
            H = ShapeOpt.Heaviside(sdf, eps, s).detach()
            #H_filtered = (TopoOpt.filter_density(Ker, H) / Ker_S).detach()
            #return sdf, torch.sum(H_filtered).item()
            return sdf, torch.sum(H).item()

        l1 = -1e9
        l2 = 1e9
        vol = 0
        while (l2 - l1) > 1e-3 and abs(vol - volTarget) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            sdf, vol = compute_volume(lmid, sdf0)
            if vol > volTarget:
                l1 = lmid
            else:
                l2 = lmid
        return sdf, vol
