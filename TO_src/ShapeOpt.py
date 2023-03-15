import torch,time
import numpy as np
import libMG as mg
from TOUtils import *
from TOLayer import TOLayer
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)

class ShapeOpt():
    def __init__(self, *, volfrac, dt=0.1, tau=1e-4, p=4, d=-0.02, maxloop=200, maxloopLinear=1000, tolx=1e-3, tolLinear=1e-2, outputInterval=1, outputDetail=False):
        # self.device = 'cpu'
        self.volfrac = volfrac
        self.dt = dt
        self.tau = tau
        self.p = p
        self.d = d
        self.rmin = rmin
        self.maxloop = maxloop
        self.maxloopLinear = maxloopLinear
        self.tolx = tolx
        self.tolLinear = tolLinear
        self.outputInterval = outputInterval
        self.outputDetail = outputDetail

    def run(self, phiTensor, phiFixedTensor, f, rhoMask, lam, mu):
        nelx, nely, nelz = shape3D(phiTensor)
        nelz = max(nelz,1)
        bb = mg.BBox()
        bb.minC = [0,0,0]
        bb.maxC = shape3D(phiTensor)
        phi = torch.ones(phiFixedTensor.shape).cuda()
        
        print("Level-set shape optimization problem")
        print(f"Number of degrees:{str(nelx)} x {str(nely)} x {str(nelz)} = {str(nelx * nely * nelz)}")
        print(f"Volume fration: {self.volfrac}")
        # max and min stiffness
        E_min = torch.tensor(1e-3)
        E_max = torch.tensor(1.0)
        nvol = self.maxloop//2
        change = self.tolx*2
        loop = 0
        g=0
        
        #initialize torch layer
        mg.initializeGPU()
        TOLayer.reset(phiTensor, phiFixedTensor, f, bb, lam, mu, self.tolLinear, self.maxloopLinear, self.outputDetail)
        TOLayer.setupCurvatureFlow(self.tau, self.dt)
        while change > self.tolx and loop < self.maxloop:
            start = time.time()
            loop += 1
            
            #compute volume
            str = (ShapeOpt.nodeToCell(phi)>0).float().detach()
            vol = torch.sum(str).item() / str.reshape(-1).shape[0]
            if loop == 1:
                volInit = vol
                
            #FE-analysis, calculate sensitivity
            str.requires_grad_()
            obj = TOLayer.apply(str * (E_max - E_min) + E_min)
            #simple replacement of topological derivative
            obj.backward()
            dir = ((-str.grad * str) * (E_max - E_min) + E_min).detach()
            dirNode = ShapeOpt.cellToNode(dir)
            
            #set augmented Lagrangian parameter
            ex = Vmax + (volInit - Vmax) * max(0,1 - loop / nvol)
            lam = torch.sum(dirNode) / dirNode.reshape(-1).shape[0] * exp(self.p * ( (vol - ex) / ex + self.d))
            
            #update level set function
            phi_old = phi.clone()
            C = dir.reshape(-1).shape[0] / torch.sum(torch.abs(dirNode)).item()
            phi = TOLayer.solveCurvatureFlow(C * (gradObjNode - lam) + phi / self.dt)
            phi = torch.minimum(torch.tensor(1.), torch.maximum(torch.tensor(-1.), phi))
            change = torch.linalg.norm(phi.reshape(-1,1) - phi_old.reshape(-1,1), ord=float('inf')).item()
            end = time.time()
            if loop%self.outputInterval == 0:
                print("it.: {0}, obj.: {1:.3f}, vol.: {2:.3f}, ch.: {3:.3f}, time: {4:.3f}, mem: {4:.3f}Gb".format(loop, obj, torch.sum(str) / (nelx * nely * nelz), change, end - start, torch.cuda.memory_allocated(None)/1024/1024/1024))
        
        mg.finalizeGPU()
        return to3DScalar(rho_old).detach().cpu().numpy()
    
    def nodeToCell(phi):
        if dim(phi) == 2:
            cellPhi = (phi[:-1,:-1] + phi[1:,:-1] + phi[:-1,1:] + phi[1:,1:]) / 4.
        else: cellPhi = (phi[:-1,:-1,:-1] + phi[1:,:-1,:-1] + phi[:-1,1:,:-1] + phi[1:,1:,:-1] +
                         phi[:-1,:-1,1:] + phi[1:,:-1,1:] + phi[:-1,1:,1:] + phi[1:,1:,1:]) / 8.
        return cellPhi
    
    def cellToNode(n):
        ret = torch.zeros(tuple([d+1 for d in n.shape]))
        if dim(n)==2:
            ret[:-1,:-1] += n
            ret[ 1:,:-1] += n 
            ret[:-1, 1:] += n 
            ret[ 1:, 1:] += n
            return phi/4
        else:
            phi[:-1,:-1,:-1] += n
            phi[ 1:,:-1,:-1] += n 
            phi[:-1, 1:,:-1] += n 
            phi[ 1:, 1:,:-1] += n
            phi[:-1,:-1, 1:] += n 
            phi[ 1:,:-1, 1:] += n
            phi[:-1, 1:, 1:] += n 
            phi[ 1:, 1:, 1:] += n
            return phi/8
            