import torch
import numpy as np
import libMG as mg

#Fully based on TOLayer, add one function to reinitialize the SDF
class SOLayer(torch.autograd.Function):
    grid = None
    sol = None
    b = None
    tol = 1e-2
    maxloop = 1000
    output = True

    @staticmethod
    def reset(phiTensor, phiFixedTensor, f, bb, lam, mu, tol=1e-2, maxloop=1000, output=True):
        SOLayer.grid = mg.GridGPU(phiTensor, phiFixedTensor, bb)
        SOLayer.grid.coarsen(128)
        SOLayer.sol = mg.GridSolverGPU(SOLayer.grid)
        SOLayer.sol.setupLinearSystem(lam, mu)

        SOLayer.b = f.cuda()
        if SOLayer.grid.isFree():
            SOLayer.b = SOLayer.sol.projectOutBases(SOLayer.b)
        SOLayer.u = torch.zeros(SOLayer.b.shape).cuda()
        SOLayer.tol = tol
        SOLayer.maxloop = maxloop
        SOLayer.output = output

    @staticmethod
    def forward(ctx, rho):
        SOLayer.solveK(rho)
        dc = SOLayer.sol.sensitivity(SOLayer.u)
        ctx.save_for_backward(dc)
        return -torch.sum(rho * dc)

    @staticmethod
    def backward(ctx, coef):
        dc = ctx.saved_tensors[0]
        return dc * coef

    @staticmethod
    def solveK(rho):
        if SOLayer.grid.isFree():
            SOLayer.b = SOLayer.sol.projectOutBases(TOLayer.b)
            SOLayer.u = SOLayer.sol.projectOutBases(TOLayer.u)
        SOLayer.sol.setRho(rho)
        SOLayer.sol.setB(SOLayer.b)
        SOLayer.u = SOLayer.sol.solveMGPCG(SOLayer.u, SOLayer.tol, SOLayer.maxloop, True, SOLayer.output)
        if SOLayer.grid.isFree():
            SOLayer.u = TOLayer.sol.projectOutBases(SOLayer.u)
        return SOLayer.u

    @staticmethod
    def redistance(rho):
        rho_new = SOLayer.sol.reinitialize(rho, 1e-3, 1000, False)
        return rho_new
