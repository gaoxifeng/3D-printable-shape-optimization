import torch.nn

from LevelSetShapeOpt import *
from TOCLayer import TOCLayer
from TOCWorstCase import TOCWorstCase
from TO_src.Viewer import *

class LevelSetShapeOptWorstCaseApproximation(LevelSetShapeOpt):
    def __init__(self, *, h=1, dt=1.5, tau=1e-6, maxloop=200, maxloopLinear=1000, tolx=1e-3, tolLinear=1e-2,
                 outputInterval=1, outputDetail=False):
        LevelSetShapeOpt.__init__(self, h=h, dt=dt, tau=tau, maxloop=maxloop, maxloopLinear=maxloopLinear, tolx=tolx,
                                  tolLinear=tolLinear, outputInterval=outputInterval, outputDetail=outputDetail)

    def run(self, phiTensor, phiFixedTensor, lam, mu, phi, phi0):
        nelx, nely, nelz = shape3D(phiTensor)
        nelz = max(nelz, 1)
        bb = mg.BBox()
        bb.minC = [0, 0, 0]
        bb.maxC = shape3D(phiTensor)
        #Now phi is pregenerated, we will write a function to automatically generate it.
        volTarget = torch.sum(phi0 < 0).item()

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
            # import mcubes
            # import trimesh
            # vertices, triangles = mcubes.marching_cubes(rho0.detach().cpu().numpy(), 0.99)
            # mesh = trimesh.Trimesh(vertices, triangles)
            # # _ = mesh.export('test.obj')
            # mesh.show()
            if loop == 1:
                TOCWorstCase.compute_worst_case(rho * (E_max - E_min) + E_min)
                # showFMagnitudeVTK("phi_f", TOCLayer.b.detach().cpu().numpy())
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
            SD_loss = torch.nn.MSELoss(reduction='sum')(LevelSetShapeOpt.compute_density(phi, self.h), rho0)
            obj = TOCLayer.apply(LevelSetShapeOpt.compute_density(phi0, self.h) * (E_max - E_min) + E_min) \
                  + SD_loss
            obj.backward()
            gradObj = -phi.grad.detach()

            # compute Lagrangian multiplier
            # gradObj, vol = LevelSetShapeOpt.find_lam(phi, gradObj, gradVolume, volTarget, self.h, self.dt)

            # update level set function / reinitialize
            phi_old = phi.clone()
            phi = TOCLayer.implicitCurvatureFlow(gradObj + phi / self.dt)
            phi = TOCLayer.reinitializeNode(phi)
            change = torch.linalg.norm(phi.reshape(-1, 1) - phi_old.reshape(-1, 1), ord=float('inf')).item()
            end = time.time()
            if loop % self.outputInterval == 0:
                if not os.path.exists("results"):
                    os.mkdir("results")
                print("it.: {0}, obj.: {1:.3f}, SD_loss.: {2:.3f}, vol.: {3:.3f}, ch.: {4:.3f}, time: {5:.3f}, mem: {6:.3f}Gb".format(loop,
                                                                                                                   obj,
                                                                                                                   SD_loss,
                                                                                                                   vol,
                                                                                                                   change,
                                                                                                                   end - start,
                                                                                                                   torch.cuda.memory_allocated(
                                                                                                                       None) / 1024 / 1024 / 1024))
                showRhoVTK("results/rho" + str(loop), to3DNodeScalar(rho).detach().cpu().numpy(), False)
        showFMagnitudeVTK("phi_f_final", TOCLayer.b.detach().cpu().numpy())
        mg.finalizeGPU()
        return to3DNodeScalar(phi_old).detach().cpu().numpy()
