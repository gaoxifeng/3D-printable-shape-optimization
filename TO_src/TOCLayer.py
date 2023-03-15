import torch
import numpy as np
import libMG as mg
from TOUtils import *


class TOCLayer(torch.autograd.Function):
    grid = None
    sol = None
    b = None
    tol = 1e-2
    maxloop = 1000
    output = True
    dim = None

    @staticmethod
    def reset(phiTensor, phiFixedTensor, f, bb, lam, mu, tol=1e-2, maxloop=1000, output=True):
        TOCLayer.grid = mg.GridGPU(phiTensor, phiFixedTensor, bb)
        TOCLayer.grid.coarsen(128)
        TOCLayer.sol = mg.GridSolverGPU(TOCLayer.grid)
        TOCLayer.sol.setupLinearElasticity(lam, mu, 3)
        TOCLayer.dim = dim(phiTensor)

        TOCLayer.b = f.cuda()
        TOCLayer.u = torch.zeros(
            (TOCLayer.b.shape[0], TOCLayer.b.shape[1] + 1, TOCLayer.b.shape[2] + 1, TOCLayer.b.shape[3] + 1)).cuda()
        TOCLayer.tol = tol
        TOCLayer.maxloop = maxloop
        TOCLayer.output = output

    @staticmethod
    def forward(ctx, rho):
        TOCLayer.solveK(rho)
        dcc = TOCLayer.sol.sensitivityCell(TOCLayer.b, TOCLayer.u)
        dc = TOCLayer.sol.sensitivity(TOCLayer.u)
        ctx.save_for_backward(dc + dcc * 2)
        return -torch.sum(rho * dc)

    @staticmethod
    def backward(ctx, coef):
        dc = ctx.saved_tensors[0]
        return dc * coef

    @staticmethod
    def solveK(rho):
        if TOCLayer.grid.isFree():
            # TOCLayer.b = TOCLayer.sol.projectOutBases(TOCLayer.b)   # (projection for b is done inside setBCell)
            TOCLayer.u = TOCLayer.sol.projectOutBases(TOCLayer.u)
        TOCLayer.sol.updateVector(rho)
        TOCLayer.sol.setBCellVector(TOCLayer.b, False)
        TOCLayer.u = TOCLayer.sol.solveMGPCGVector(TOCLayer.u, TOCLayer.tol, TOCLayer.maxloop, True, TOCLayer.output)
        if TOCLayer.grid.isFree():
            TOCLayer.u = TOCLayer.sol.projectOutBases(TOCLayer.u)
        return TOCLayer.u

    @staticmethod
    def nodeToCell(rho):  # u->b
        TOCLayer.sol.nodeToCellVector(TOCLayer.b, TOCLayer.u, rho)

    @staticmethod
    def redistance(rho, eps=1e-3, maxIter=1000, output=False):
        return TOLayer.sol.reinitialize(rho, eps, maxIter, output)

    @staticmethod
    def setupCurvatureFlow(dt, tau):
        TOCLayer.sol.setupCurvatureFlow(dt, tau, TOCLayer.dim)

    @staticmethod
    def implicitCurvatureFlow(b, tol=1e-9, maxloop=100, MG=False):
        if MG:
            TOCLayer.sol.updateScalar()
        TOCLayer.sol.setBScalar(to3DNodeScalar(b), False)
        TOCLayer.sol.mulBScalarByNNT()
        return makeSameDimScalar(TOCLayer.sol.solveMGPCGScalar(to3DNodeScalar(b), tol, maxloop, MG, TOCLayer.output),
                                 TOCLayer.dim)


def debug(iter=0, DTYPE=torch.float64):
    bb = mg.BBox()
    bb.minC = [-1, -1, -1]
    bb.maxC = [1, 1, 1]
    res = [10, 10, 15]

    def interp1D(a, b, alpha):
        return a * (1 - alpha) + b * alpha

    def interpC(x, y, z, bb):
        id = [x, y, z]
        for d in range(3):
            id[d] = interp1D(bb.minC[d], bb.maxC[d], (id[d] + 0.5) / res[d])
        return np.array(id)

    def interpV(x, y, z, bb):
        id = [x, y, z]
        for d in range(3):
            id[d] = interp1D(bb.minC[d], bb.maxC[d], id[d] / res[d])
        return np.array(id)

    def phi(pos):
        # if phi(pos)<0, then this position is solid
        return np.linalg.norm(pos) - 0.5

    def phiFixed(pos):
        # if phiFixed(pos)<0, then this position is fixed
        if iter == 0:
            return 1
        else:
            return np.sum(pos)

    phiTensor = torch.rand(tuple(res), dtype=DTYPE).cuda()
    for z in range(res[2]):
        for y in range(res[1]):
            for x in range(res[0]):
                phiTensor[x, y, z] = phi(interpC(x, y, z, bb))

    phiFixedTensor = torch.rand(tuple([res[0] + 1, res[1] + 1, res[2] + 1]), dtype=DTYPE).cuda()
    for z in range(res[2] + 1):
        for y in range(res[1] + 1):
            for x in range(res[0] + 1):
                phiFixedTensor[x, y, z] = phiFixed(interpV(x, y, z, bb))
    f = torch.rand(tuple([3, res[0], res[1], res[2]]), dtype=DTYPE).cuda()

    eps = 1e-7
    TOCLayer.reset(phiTensor, phiFixedTensor, f, bb, 100, 100, 1e-8)
    rho = torch.rand(tuple(res), dtype=DTYPE).cuda()

    rho.requires_grad_()
    E = TOCLayer.apply(rho)
    E.backward()

    drho = torch.ones(tuple(res), dtype=DTYPE).cuda()
    rho2 = rho + drho * eps
    E2 = TOCLayer.apply(rho2)
    numeric = (E2 - E) / eps
    analytic = torch.sum(rho.grad * drho)
    print('TOCLayer: analytic=%f, numeric=%f, error=%f' % (analytic, numeric, analytic - numeric))


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    mg.initializeGPU()
    debug(0)
    debug(1)
    mg.finalizeGPU()