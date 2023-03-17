import torch,time,math,os
import numpy as np
import libMG as mg
from Viewer import *
from TOUtils import *
from TOLayer import TOLayer
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)

class LevelSetTopoOpt():
    def __init__(self, *, volfrac, dt=0.1, tau=1e-4, p=4, d=-0.02, maxloop=200, maxloopLinear=1000, tolx=1e-3, tolLinear=1e-2, outputInterval=1, outputDetail=False):
        # self.device = 'cpu'
        self.volfrac = volfrac
        self.dt = dt
        self.tau = tau
        self.p = p
        self.d = d
        self.maxloop = maxloop
        self.maxloopLinear = maxloopLinear
        self.tolx = tolx
        self.tolLinear = tolLinear
        self.outputInterval = outputInterval
        self.outputDetail = outputDetail

    def run(self, phiTensor, phiFixedTensor, f, lam, mu, phi=None, curvatureOnly=False):
        nelx, nely, nelz = shape3D(phiTensor)
        nelz = max(nelz,1)
        bb = mg.BBox()
        bb.minC = [0,0,0]
        bb.maxC = shape3D(phiTensor)
        if phi is not None:
            assert phi.shape == phiFixedTensor.shape
        else: phi = torch.ones(phiFixedTensor.shape).cuda()
        
        print("Level-set minimum complicance problem")
        print(f"Number of degrees:{str(nelx)} x {str(nely)} x {str(nelz)} = {str(nelx * nely * nelz)}")
        print(f"Volume fraction: {self.volfrac}")
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
        TOLayer.setupCurvatureFlow(self.dt, self.tau * nelx * nely * nelz)
        while change > self.tolx and loop < self.maxloop:
            start = time.time()
            loop += 1
            
            #compute volume
            strength = (LevelSetTopoOpt.nodeToCell(phi)>0).double().detach()
            vol = torch.sum(strength).item() / strength.reshape(-1).shape[0]
            if loop == 1:
                volInit = vol
                if self.volfrac is None:
                    self.volfrac = volInit
                
            #FE-analysis, calculate sensitivity
            if curvatureOnly:
                dir = torch.ones(strength.shape).cuda()
                obj = 0
            else:
                strength.requires_grad_()
                obj = TOLayer.apply(strength * (E_max - E_min) + E_min)
                #simple replacement of topological derivative
                obj.backward()
                dir = (-strength.grad.detach() * (strength * (E_max - E_min) + E_min)).detach()
            dirNode = LevelSetTopoOpt.cellToNode(dir)
            
            #set augmented Lagrangian parameter
            ex = self.volfrac + (volInit - self.volfrac) * max(0, 1 - loop / nvol)
            lam = torch.sum(dirNode) / dirNode.reshape(-1).shape[0] * math.exp(self.p * ( (vol - ex) / ex + self.d))
            
            #update level set function
            phi_old = phi.clone()
            C = dir.reshape(-1).shape[0] / torch.sum(torch.abs(dirNode)).item()
            phi = TOLayer.implicitCurvatureFlow(C * (dirNode - lam) + phi / self.dt)
            phi = torch.minimum(torch.tensor(1.), torch.maximum(torch.tensor(-1.), phi))
            change = torch.linalg.norm(phi.reshape(-1,1) - phi_old.reshape(-1,1), ord=float('inf')).item()
            end = time.time()
            if loop%self.outputInterval == 0:
                if not os.path.exists("results"):
                    os.mkdir("results")
                print("it.: {0}, obj.: {1:.3f}, vol.: {2:.3f}, ch.: {3:.3f}, time: {4:.3f}, mem: {5:.3f}Gb".format(loop, obj, torch.sum(strength).item() / (nelx * nely * nelz), change, end - start, torch.cuda.memory_allocated(None)/1024/1024/1024))
                showRhoVTK("results/phi"+str(loop), to3DNodeScalar(phi).detach().cpu().numpy(), False)
                
        mg.finalizeGPU()
        return to3DNodeScalar(phi_old).detach().cpu().numpy()
    
    def nodeToCell(phi):
        if dim(phi) == 2:
            cellPhi = (phi[:-1,:-1] + 
                       phi[ 1:,:-1] + 
                       phi[:-1, 1:] + 
                       phi[ 1:, 1:]) / 4.
        else: 
            cellPhi = (phi[:-1,:-1,:-1] + 
                       phi[ 1:,:-1,:-1] + 
                       phi[:-1, 1:,:-1] + 
                       phi[ 1:, 1:,:-1] +
                       phi[:-1,:-1, 1:] + 
                       phi[ 1:,:-1, 1:] + 
                       phi[:-1, 1:, 1:] + 
                       phi[ 1:, 1:, 1:]) / 8.
        return cellPhi
    
    def cellToNode(n):
        ret = torch.zeros(tuple([d+1 for d in n.shape])).cuda()
        if dim(n)==2:
            ret[:-1,:-1] += n
            ret[ 1:,:-1] += n 
            ret[:-1, 1:] += n 
            ret[ 1:, 1:] += n
            return ret / 4.
        else:
            ret[:-1,:-1,:-1] += n
            ret[ 1:,:-1,:-1] += n 
            ret[:-1, 1:,:-1] += n 
            ret[ 1:, 1:,:-1] += n
            ret[:-1,:-1, 1:] += n 
            ret[ 1:,:-1, 1:] += n
            ret[:-1, 1:, 1:] += n 
            ret[ 1:, 1:, 1:] += n
            return ret / 8.
