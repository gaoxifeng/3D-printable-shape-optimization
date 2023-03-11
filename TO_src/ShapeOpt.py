import skfmm, torch, time, math
import numpy as np
import libMG as mg
from TOLayer import TOLayer
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)
from Viewer import *

class ShapeOpt():
    def __init__(self, *, limit=1.5, tau=.01, eps_Heaviside=1e-4, maxloop=200, maxloopLinear=1000, CFL=.2, tolx=1e-3, tolLinear=1e-2, outputInterval=1, outputDetail=False):
        # self.device = 'cpu'
        self.limit = limit
        self.tau = tau
        self.eps = eps_Heaviside
        self.maxloop = maxloop
        self.maxloopLinear = maxloopLinear
        self.CFL = CFL
        self.tolx = tolx
        self.tolLinear = tolLinear
        self.outputInterval = outputInterval
        self.outputDetail = outputDetail

    def round_up(n, blk):
        return (n+blk-1)//blk

    def fill_block(sdf, x, y, z, blk, ratio, cube):
        nelx, nely, nelz = sdf.shape
        x *= blk
        y *= blk
        z *= blk
        rad = blk/2*ratio
        for xx in range(x, min(x+blk, nelx)):
            for yy in range(y, min(y+blk, nely)):
                for zz in range(z, min(z+blk, nelz)):
                    dir = np.array([xx-x-blk/2, yy-y-blk/2, zz-z-blk/2])
                    if cube:
                        sdf[xx,yy,zz]=np.max(np.abs(dir)-rad)
                    else: sdf[xx,yy,zz]=np.linalg.norm(dir)-rad

    def create_bubble(sdf, blk, ratio=.7, cube=False):
        nelx, nely, nelz = sdf.shape
        for x in range(0, ShapeOpt.round_up(nelx, blk)):
            for y in range(0, ShapeOpt.round_up(nely, blk)):
                for z in range(0, ShapeOpt.round_up(nelz, blk)):
                    ShapeOpt.fill_block(sdf, x, y, z, blk, ratio, cube)
        return sdf

    def run(self, sdf, phiTensor, phiFixedTensor, f, rhoMask, lam, mu):
        nelx, nely, nelz = sdf.shape
        bb = mg.BBox()
        bb.minC = [0, 0, 0]
        bb.maxC = [nelx, nely, nelz]

        print("Minimum shape optimization problem with OC")
        print(f"Number of degrees: {str(nelx)} x {str(nely)} x {str(nelz)} = {str(nelx * nely * nelz)}")
        # print(f"Volume fration: {self.volfrac}, Penalty p: {self.p}")
        # max and min stiffness
        change = self.tolx * 2
        loop = 0

        # initialize torch layer
        mg.initializeGPU()
        TOLayer.reset(phiTensor, phiFixedTensor, f, bb, lam, mu, self.tolLinear, self.maxloopLinear, self.outputDetail)
        sdf = ShapeOpt.clamp(TOLayer.redistance(sdf), self.limit)
        while change > self.tolx and loop < self.maxloop:
            start = time.time()
            loop += 1
            if loop%50==0:
                showSdf(sdf.detach().cpu().numpy())

            # compute filtered volume gradient
            sdf = sdf.detach()
            sdf.requires_grad_()
            H = ShapeOpt.Heaviside(sdf, 0., self.limit)
            vol = torch.sum(H)
            vol.backward()
            gradVol = sdf.grad.detach()
            
            # use given volume as constraint
            if loop==1:
                volTarget = vol.item()

            # compute filtered objective gradient
            sdf = sdf.detach()
            sdf.requires_grad_()
            if not math.isfinite(self.tau):
                obj = ShapeOpt.Dirichlet(sdf)
            else: 
                H = ShapeOpt.Heaviside(sdf, self.eps, self.limit)
                obj = TOLayer.apply(H) + ShapeOpt.Dirichlet(sdf) * self.tau
            obj.backward()
            gradObj = sdf.grad.detach()

            # Find Lagrange Lambda and update shape
            sdf_old = sdf.clone()
            sdf, _ = ShapeOpt.oc_grid(sdf, self.limit, gradObj, gradVol, volTarget, self.CFL)
            change = torch.linalg.norm(sdf.reshape(-1, 1) - sdf_old.reshape(-1, 1), ord=float('inf')).item()
            end = time.time()
            if loop % self.outputInterval == 0:
                print("it.: {0}, obj.: {1:.3f}, vol.: {2:.3f}, ch.: {3:.3f}, time: {4:.3f}, mem: {4:.3f}Gb".
                      format(loop, obj, vol, change, end - start, torch.cuda.memory_allocated(None) / 1024 / 1024 / 1024))
        mg.finalizeGPU()
        return sdf.detach().cpu().numpy()
    
    def Dirichlet(sdf):
        L = sdf[1,:,:]-sdf[0,:,:]
        R = sdf[-1,:,:]-sdf[-2,:,:]
        M = (sdf[2:,:,:]-sdf[:-2,:,:])/2
        DX = torch.cat((L.unsqueeze(0),M,R.unsqueeze(0)),dim=0)
        ret = torch.dot(DX.reshape(-1),DX.reshape(-1))
        
        L = sdf[:,1,:]-sdf[:,0,:]
        R = sdf[:,-1,:]-sdf[:,-2,:]
        M = (sdf[:,2:,:]-sdf[:,:-2,:])/2
        DY = torch.cat((L.unsqueeze(1),M,R.unsqueeze(1)),dim=1)
        ret += torch.dot(DY.reshape(-1),DY.reshape(-1))
        
        L = sdf[:,:,1]-sdf[:,:,0]
        R = sdf[:,:,-1]-sdf[:,:,-2]
        M = (sdf[:,:,2:]-sdf[:,:,:-2])/2
        DZ = torch.cat((L.unsqueeze(2),M,R.unsqueeze(2)),dim=2)
        ret += torch.dot(DZ.reshape(-1),DZ.reshape(-1))
        return ret
    
    def Heaviside(sdf, eps, limit):
        sdf_ratio = sdf / limit #clamp to [-1,1]
        X_s = 0.25 * sdf_ratio ** 3 - 0.75 * sdf_ratio + 0.5
        return eps + X_s
    
    def clamp(sdf, limit):
        return torch.maximum(torch.tensor(-limit), torch.minimum(torch.tensor(limit), sdf)).detach()
    
    def oc_grid(sdf0, limit, gradObj, gradVol, volTarget, CFL):
        l1 = -1e9
        l2 = 1e9
        vol = 0
        gradObj = torch.minimum(gradObj / (torch.max(torch.abs(gradObj)) + 1e-6), torch.tensor(1.0))
        while (l2 - l1) > 1e-6 and abs(vol - volTarget) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            sdf = (sdf0 - CFL * gradObj - lmid * gradVol).detach()
            sdf = ShapeOpt.clamp(sdf, limit)
            vol = torch.sum(ShapeOpt.Heaviside(sdf, 0., limit)).item()
            if vol > volTarget:
                l1 = lmid
            else:
                l2 = lmid
        return sdf, vol
