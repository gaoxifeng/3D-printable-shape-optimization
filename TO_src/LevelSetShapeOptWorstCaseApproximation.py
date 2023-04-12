import torch.nn

from LevelSetShapeOpt import *
from TOCLayer import TOCLayer
from TOCWorstCase import TOCWorstCase
from TO_src.Viewer import *
from SIMPTopoOpt import SIMPTopoOpt

def weighted_grid(sdf,alpha):
    return 1+alpha*torch.exp(-torch.abs(sdf))

def moving_mask(sdf):
    mask = -torch.ones_like(sdf)
    mask[sdf < 5] = 1
    return mask
class LevelSetShapeOptWorstCaseApproximation(LevelSetShapeOpt):
    def __init__(self, *, h=1, dt=1.5, tau=1e-6, maxloop=200, maxloopLinear=1000, tolx=1e-3, tolLinear=1e-2,
                 outputInterval=1, outputDetail=False):
        LevelSetShapeOpt.__init__(self, h=h, dt=dt, tau=tau, maxloop=maxloop, maxloopLinear=maxloopLinear, tolx=tolx,
                                  tolLinear=tolLinear, outputInterval=outputInterval, outputDetail=outputDetail)

    def run(self, phiTensor, phiFixedTensor, lam, mu, phi, phi0, mask0, type='mask'):
        nelx, nely, nelz = shape3D(phiTensor)
        nelz = max(nelz, 1)
        bb = mg.BBox()
        bb.minC = [0, 0, 0]
        bb.maxC = shape3D(phiTensor)
        #Now phi is pregenerated, we will write a function to automatically generate it.
        volTarget = torch.sum(phi0 < 0).item()
        # phi = torch.clone(phi0)
        # Ker, Ker_S = SIMPTopoOpt.filter(2.5, phi)
        if type=='mask':
            phiFixedTensor = phiFixedTensor * (mask0)
        else:
            pass

        print(f"Optimization with {type}")
        print("Worst case level-set shape optimization problem")
        print(f"Number of degrees:{str(nelx)} x {str(nely)} x {str(nelz)} = {str(nelx * nely * nelz)}")
        print(f"Volume target: {volTarget}, Tau={self.tau}")
        # mask = LevelSetTopoOpt.nodeToCell(mask0)
        # showRhoVTK("results/mask_000", to3DNodeScalar(mask).detach().cpu().numpy(), False)
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
            # TOCWorstCase.compute_worst_case(rho * (E_max - E_min) + E_min)

            # if loop % 10 == 1:
            #     torch.save(phi, f"phi{loop}.pt")
            #     torch.save(rho, f"rho{loop}.pt")
            #     showRhoVTK(f'rho_{loop}', to3DNodeScalar(rho).detach().cpu().numpy(), False)
            #     TOCWorstCase.compute_worst_case(rho * (E_max - E_min) + E_min)
            #     # TOCWorstCase.update_worst_case(rho * (E_max - E_min) + E_min)
            #     showFMagnitudeVTK(f"phi_f_{loop}", TOCLayer.b.detach().cpu().numpy())
            #
            # else:
            #     TOCWorstCase.update_worst_case(rho * (E_max - E_min) + E_min)
            #     pass


            # TOCLayer.fix()
            # u = TOCLayer.solveK(rho * (E_max - E_min) + E_min).detach().cpu().numpy()
            # f = TOCLayer.bOut.detach().cpu().numpy()
            #
            # showFMagnitudeCellVTK("TwoForce_u", u)
            # showFMagnitudeCellVTK("TwoForce_f", f)

            # Eng = np.matmul(f.flatten().T , u.flatten())
            # showFMagnitudeCellVTK("Eng_r0", u*f)
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
            SD_loss = torch.nn.MSELoss(reduction='mean')(phi, phi0)
            SP_loss = TOCLayer.apply(LevelSetShapeOpt.compute_density(phi, self.h) * (E_max - E_min) + E_min)

            # c = 0
            # if SD_loss > c*volTarget:
            #     b = 1*SP_loss.item() / (SD_loss.item()+0.01)
            # else:
            #     b = 0
            #     print('Now there is no shape similarity loss')
            b = 1 * SP_loss.item() / (SD_loss.item() + 0.01)
            obj = SP_loss +  b * SD_loss
            # obj = SP_loss
            obj.backward()
            gradObj = -phi.grad.detach()

            # if type=='weighted':
            #     #Generate the weight value for each grid pixel and then update based on the sdf value
            #     #Follow the formula we used in DRiLLS paper
            #     gradObj = weighted_grid(phi,1) * gradObj
            # else:
            #     pass
            # if SD_loss > c*volTarget:
            #     pass
            # else:
                # self.dt = 0.1
            # gradObj, vol = LevelSetShapeOpt.find_lam(phi, gradObj, gradVolume, volTarget, self.h, self.dt)
                # print('Now we apply lam')

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
            # if type == 'movingmask':
            #     # Redefine the mask based on current sdf
            #     if loop % 20 == 0:
            #         mask = moving_mask(phi_old)
            #         phiFixedTensor = phiFixedTensor * mask
            #         # phi = torch.clone(phi)
            #         TOCLayer.reset(phiTensor, phiFixedTensor, f, bb, lam, mu, self.tolLinear, self.maxloopLinear,
            #                        self.outputDetail)
            #         TOCLayer.setupCurvatureFlow(self.dt, self.tau * nelx * nely * nelz)
            #         TOCWorstCase.compute_worst_case(LevelSetShapeOpt.compute_density(phi, self.h) * (E_max - E_min) + E_min)
            #         # phi = TOCLayer.reinitializeNode(phi)
            #     else:
            #         pass
            # else:
            #     pass

        mg.finalizeGPU()
        return to3DNodeScalar(phi_old).detach().cpu().numpy(), rho.detach().cpu().numpy()
