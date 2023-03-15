import torch
import numpy as np
import libMG as mg
from TOUtils import *

class TOLayer(torch.autograd.Function):
    grid = None
    sol = None
    b = None
    tol = 1e-2
    maxloop = 1000
    output = True
    dim = None
    
    @staticmethod
    def reset(phiTensor, phiFixedTensor, f, bb, lam, mu, tol=1e-2, maxloop=1000, output=True):
        TOLayer.grid = mg.GridGPU(to3DScalar(phiTensor), to3DNodeScalar(phiFixedTensor), bb)
        TOLayer.grid.coarsen(128)
        TOLayer.dim = dim(phiTensor)
        TOLayer.sol = mg.GridSolverGPU(TOLayer.grid)
        TOLayer.sol.setupLinearElasticity(lam, mu, TOLayer.dim)
        
        TOLayer.b = f.cuda()
        if TOLayer.grid.isFree():
            TOLayer.b = makeSameDimVector(TOLayer.sol.projectOutBases(to3DNodeVector(TOLayer.b)), TOLayer.dim)
        TOLayer.u = torch.zeros(TOLayer.b.shape).cuda()
        TOLayer.tol = tol
        TOLayer.maxloop = maxloop
        TOLayer.output = output
    
    @staticmethod
    def forward(ctx,rho):
        TOLayer.solveK(rho)
        dc = makeSameDimScalar(TOLayer.sol.sensitivity(to3DNodeVector(TOLayer.u)), TOLayer.dim)
        ctx.save_for_backward(dc)
        return -torch.sum(rho * dc)
        
    @staticmethod
    def backward(ctx,coef):
        dc=ctx.saved_tensors[0]
        return dc*coef
    
    @staticmethod
    def solveK(rho):
        if TOLayer.grid.isFree():
            TOLayer.b = makeSameDimVector(TOLayer.sol.projectOutBases(to3DNodeVector(TOLayer.b)), TOLayer.dim)
            TOLayer.u = makeSameDimVector(TOLayer.sol.projectOutBases(to3DNodeVector(TOLayer.u)), TOLayer.dim)
        TOLayer.sol.updateVector(to3DScalar(rho))
        TOLayer.sol.setBVector(to3DNodeVector(TOLayer.b),False)
        TOLayer.u = makeSameDimVector(TOLayer.sol.solveMGPCGVector(to3DNodeVector(TOLayer.u), TOLayer.tol, TOLayer.maxloop, True, TOLayer.output), TOLayer.dim)
        if TOLayer.grid.isFree():
            TOLayer.u = makeSameDimVector(TOLayer.sol.projectOutBases(to3DNodeVector(TOLayer.u)), TOLayer.dim)
        return TOLayer.u
    
    @staticmethod
    def redistance(rho, eps=1e-3, maxIter=1000, output=False):
        return makeSameDimScalar(TOLayer.sol.reinitialize(rho, eps, maxIter, output), TOLayer.dim)
        
    @staticmethod
    def setupCurvatureFlow(dt, tau):
        TOLayer.sol.setupCurvatureFlow(dt, tau, TOLayer.dim)
        
    @staticmethod
    def implicitCurvatureFlow(b, tol=1e-9, maxloop=100, MG=False):
        if MG:
            TOLayer.sol.updateScalar()
        TOLayer.sol.setBScalar(to3DNodeScalar(b),False)
        TOLayer.sol.mulBScalarByNNT()
        return makeSameDimScalar(TOLayer.sol.solveMGPCGScalar(to3DNodeScalar(b), tol, maxloop, MG, TOLayer.output), TOLayer.dim)
        
def debug3D(iter=0, DTYPE=torch.float64):
    bb=mg.BBox()
    bb.minC=[-1,-1,-1]
    bb.maxC=[1,1,1]
    res=[10,10,15]
    
    def interp1D(a,b,alpha):
        return a*(1-alpha)+b*alpha
    def interpC(x,y,z,bb):
        id=[x,y,z]
        for d in range(3):
            id[d]=interp1D(bb.minC[d],bb.maxC[d],(id[d]+0.5)/res[d])
        return np.array(id)
    def interpV(x,y,z,bb):
        id=[x,y,z]
        for d in range(3):
            id[d]=interp1D(bb.minC[d],bb.maxC[d],id[d]/res[d])
        return np.array(id)
    def phi(pos):
        #if phi(pos)<0, then this position is solid
        return np.linalg.norm(pos)-0.5
    def phiFixed(pos):
        #if phiFixed(pos)<0, then this position is fixed
        if iter==0:
            return 1
        else: return np.sum(pos)

    phiTensor=torch.rand(tuple(res),dtype=DTYPE).cuda()
    for z in range(res[2]):
        for y in range(res[1]):
            for x in range(res[0]):
                phiTensor[x,y,z]=phi(interpC(x,y,z,bb))
    
    phiFixedTensor=torch.rand(tuple([res[0]+1,res[1]+1,res[2]+1]),dtype=DTYPE).cuda()
    for z in range(res[2]+1):
        for y in range(res[1]+1):
            for x in range(res[0]+1):
                phiFixedTensor[x,y,z]=phiFixed(interpV(x,y,z,bb))
    f=torch.rand(tuple([3,res[0]+1,res[1]+1,res[2]+1]),dtype=DTYPE).cuda()
    
    eps = 1e-7
    TOLayer.reset(phiTensor, phiFixedTensor, f, bb, 100, 100, 1e-8)
    rho = torch.rand(tuple(res),dtype=DTYPE).cuda()
    
    rho.requires_grad_()
    E = TOLayer.apply(rho)
    E.backward()
    
    drho = torch.ones(tuple(res),dtype=DTYPE).cuda()
    rho2 = rho+drho*eps
    E2 = TOLayer.apply(rho2)
    numeric = (E2-E)/eps
    analytic = torch.sum(rho.grad * drho)
    print('TOLayer: analytic=%f, numeric=%f, error=%f'%(analytic,numeric,analytic-numeric))
        
def debug2D(iter=0, DTYPE=torch.float64):
    bb=mg.BBox()
    bb.minC=[-1,-1,0]
    bb.maxC=[1,1,0]
    res=[10,10]
    
    def interp1D(a,b,alpha):
        return a*(1-alpha)+b*alpha
    def interpC(x,y,bb):
        id=[x,y]
        for d in range(2):
            id[d]=interp1D(bb.minC[d],bb.maxC[d],(id[d]+0.5)/res[d])
        return np.array(id)
    def interpV(x,y,bb):
        id=[x,y]
        for d in range(2):
            id[d]=interp1D(bb.minC[d],bb.maxC[d],id[d]/res[d])
        return np.array(id)
    def phi(pos):
        #if phi(pos)<0, then this position is solid
        return np.linalg.norm(pos)-0.5
    def phiFixed(pos):
        #if phiFixed(pos)<0, then this position is fixed
        if iter==0:
            return 1
        else: return np.sum(pos)

    phiTensor=torch.rand(tuple(res),dtype=DTYPE).cuda()
    for y in range(res[1]):
        for x in range(res[0]):
            phiTensor[x,y]=phi(interpC(x,y,bb))
    
    phiFixedTensor=torch.rand(tuple([res[0]+1,res[1]+1]),dtype=DTYPE).cuda()
    for y in range(res[1]+1):
        for x in range(res[0]+1):
            phiFixedTensor[x,y]=phiFixed(interpV(x,y,bb))
    f=torch.rand(tuple([3,res[0]+1,res[1]+1]),dtype=DTYPE).cuda()
    
    eps = 1e-7
    TOLayer.reset(phiTensor, phiFixedTensor, f, bb, 100, 100, 1e-8)
    rho = torch.rand(tuple(res),dtype=DTYPE).cuda()
    
    rho.requires_grad_()
    E = TOLayer.apply(rho)
    E.backward()
    
    drho = torch.ones(tuple(res),dtype=DTYPE).cuda()
    rho2 = rho+drho*eps
    E2 = TOLayer.apply(rho2)
    numeric = (E2-E)/eps
    analytic = torch.sum(rho.grad * drho)
    print('TOLayer: analytic=%f, numeric=%f, error=%f'%(analytic,numeric,analytic-numeric))
        
if __name__=='__main__':
    torch.set_default_dtype(torch.float64)
    mg.initializeGPU()
    debug3D(0)
    debug3D(1)
    debug2D(1)
    mg.finalizeGPU()