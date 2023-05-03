import torch.nn

from LevelSetShapeOpt import *
from TOCLayer import TOCLayer
from TOCWorstCase import TOCWorstCase
from TO_src.Viewer import *
from SIMPTopoOpt import SIMPTopoOpt

def weighted_grid(sdf,alpha):
    return 1+alpha*torch.exp(-torch.abs(sdf))

def moving_mask(sdf, sdf0, h, value):
    mask = -2*torch.ones_like(sdf)
    sdfc = torch.clamp(sdf, min=-h, max=h) / h
    Hphic = sdfc ** 3 * .25 - sdfc * .75 + 0.5
    mask[Hphic < value] = 1
    mask[sdf0 < 0] = -1
    return mask
class LevelSetShapeOptWorstCaseApproximation(LevelSetShapeOpt):
    def __init__(self, *, h=1, dt=1.5, tau=1e-6, maxloop=200, maxloopLinear=1000, tolx=1e-3, tolLinear=1e-2,
                 outputInterval=1, outputDetail=False):
        LevelSetShapeOpt.__init__(self, h=h, dt=dt, tau=tau, maxloop=maxloop, maxloopLinear=maxloopLinear, tolx=tolx,
                                  tolLinear=tolLinear, outputInterval=outputInterval, outputDetail=outputDetail)

    def run(self, phiTensor, phiFixedTensor, lam, mu, phi, phi0, type='movingmask'):
        nelx, nely, nelz = shape3D(phiTensor)
        nelz = max(nelz, 1)
        bb = mg.BBox()
        bb.minC = [0, 0, 0]
        bb.maxC = shape3D(phiTensor)

        volTarget = torch.sum(phi0 < 0).item()
        phiFixedTensor0 = torch.clone(phiFixedTensor)
        if type=='mask':
            phiFixedTensor = phiFixedTensor0 * (phi0)
        elif type == 'movingmask':
            phiFixedTensor = phiFixedTensor0* (phi0)
        else:
            pass

        print(f"Optimization with {type}")
        print("Worst case level-set shape optimization problem")
        print(f"Number of degrees:{str(nelx)} x {str(nely)} x {str(nelz)} = {str(nelx * nely * nelz)}")
        print(f"Volume target: {volTarget}, Tau={self.tau}")

        # max and min stiffness
        E_min = torch.tensor(1e-3)
        E_max = torch.tensor(1.0)
        nvol = self.maxloop // 2
        change = self.tolx * 2
        loop = 0
        g = 0

        # initialize torch layer
        mg.initializeGPU()
        if len(phi.shape) == 2:
            f = torch.zeros((2, nelx, nely))
        else:
            f = torch.zeros((3, nelx, nely, nelz))
        # showFMagnitudeCellVTK("TwoForce_finit", f.detach().cpu().numpy())

        TOCLayer.reset(phiTensor, phiFixedTensor, f, bb, lam, mu, self.tolLinear, self.maxloopLinear, self.outputDetail)
        TOCLayer.setupCurvatureFlow(self.dt, self.tau * nelx * nely * nelz)
        phi = TOCLayer.reinitializeNode(phi)
        phi0 = TOCLayer.reinitializeNode(phi0)
        rho0 = LevelSetShapeOpt.compute_density(phi0, self.h)
        showRhoVTK("results/rho_000", to3DNodeScalar(rho0).detach().cpu().numpy(), False)
        while change > self.tolx and loop < self.maxloop:

            start = time.time()
            loop += 1

            # update worst case
            rho = LevelSetShapeOpt.compute_density(phi, self.h)

            if loop==1:
                TOCWorstCase.compute_worst_case(rho * (E_max - E_min) + E_min)
            else:
                TOCWorstCase.update_worst_case(rho * (E_max - E_min) + E_min)
                # pass
            # compute volume
            phi = phi.detach()
            phi.requires_grad_()
            vol = torch.sum(LevelSetShapeOpt.compute_density(phi, self.h))
            vol.backward()
            gradVolume = phi.grad.detach()

            # FE-analysis, calculate sensitivity, add L^2 loss in shape difference
            phi = phi.detach()
            phi.requires_grad_()

            # SD_loss = torch.nn.MSELoss(reduction='sum')((phi-phi0), torch.zeros_like(phi))
            SD_loss = torch.nn.MSELoss(reduction='sum')(LevelSetShapeOpt.compute_density(phi, self.h), rho0)
            SP_loss = TOCLayer.apply(LevelSetShapeOpt.compute_density(phi, self.h) * (E_max - E_min) + E_min)

            b = 1 * SP_loss.item() / (SD_loss.item() + 0.01)
            obj = SP_loss +  b * SD_loss

            obj.backward()
            gradObj = -phi.grad.detach()


            # update level set function / reinitialize
            phi_old = phi.clone()
            phi = TOCLayer.implicitCurvatureFlow(gradObj + phi / self.dt)

            phi = TOCLayer.reinitializeNode(phi)
            change = torch.linalg.norm(phi.reshape(-1, 1) - phi_old.reshape(-1, 1), ord=float('inf')).item()

            end = time.time()
            if loop % self.outputInterval == 0:
                if not os.path.exists("results"):
                    os.mkdir("results")
                print("it.: {0}, SP_loss.: {1:.3f}, SD_loss.: {2:.3f}, vol.: {3:.3f}, ch.: {4:.3f}, time: {5:.3f}, mem: {6:.3f}Gb".format(loop,
                                                                                                                   SP_loss,
                                                                                                                   SD_loss,
                                                                                                                   vol,
                                                                                                                   change,
                                                                                                                   end - start,
                                                                                                                   torch.cuda.memory_allocated(
                                                                                                                       None) / 1024 / 1024 / 1024))
            if loop % 50 == 0:
                showRhoVTK("results/rho" + str(loop), to3DNodeScalar(rho).detach().cpu().numpy(), False)
            if loop % 10 == 0:
                if type == 'movingmask':
                    # Redefine the mask based on current sdf
                    # self.h = self.h - loop/self.maxloop
                    mask = moving_mask(phi_old, phi0, self.h, 0.75)
                    phiFixedTensor = phiFixedTensor0 * mask
                    # phi = torch.clone(phi)
                    TOCLayer.reset(phiTensor, phiFixedTensor, f, bb, lam, mu, self.tolLinear, self.maxloopLinear,
                                   self.outputDetail)
                    TOCLayer.setupCurvatureFlow(self.dt, self.tau * nelx * nely * nelz)
                    TOCWorstCase.compute_worst_case(LevelSetShapeOpt.compute_density(phi, self.h) * (E_max - E_min) + E_min)
                        # phi = TOCLayer.reinitializeNode(phi)
                else:
                    pass
        mg.finalizeGPU()
        return to3DNodeScalar(phi_old).detach().cpu().numpy(), rho.detach().cpu().numpy()
