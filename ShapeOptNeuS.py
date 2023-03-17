import torch
from tqdm import tqdm
from ShapeOpt import ShapeOpt
from TOCWorstCase import TOCWorstCase
from TOCLayer import TOCLayer
from TOUtils import *
import libMG as mg
import time
import math
"""
Input is a mesh, core function is run
one batch with batchsize images and NeuS run once
After K iterations, generate a grid to ShapeOpt,
backpropogate the gradient or what?
How to combine the two parts with a general loss term?
"""
torch.set_default_dtype(torch.float64)
class ShapeOptNeuS(ShapeOpt):
    def __init__(self, FieldRender, volfrac, dt=0.1, tau=1e-4, p=4, d=-0.02, maxloop=200, maxloopLinear=1000, tolx=1e-3, tolLinear=1e-2, outputInterval=1, outputDetail=False):
        ShapeOpt.__init__(self, volfrac=volfrac, dt=dt, tau=tau, p=p, d=d, maxloop=maxloop, maxloopLinear=maxloopLinear, tolx=tolx, tolLinear=tolLinear, outputInterval=outputInterval, outputDetail=outputDetail)
        self.Field_R = FieldRender
        a = 1

    def run(self, phiTensor, phiFixedTensor, f, lam, mu, phi=None, curvatureOnly=False, img_batch_size=4, img_resolution=64, res_step=5):
        self.Field_R.train(img_batch_size, img_resolution, res_step)
        self.WorstCaseSO(phiTensor, phiFixedTensor, f, lam, mu, phi, curvatureOnly)


    def WorstCaseSO(self, phiTensor, phiFixedTensor, f, lam, mu, phi=None, curvatureOnly=False):
        nelx, nely, nelz = shape3D(phiTensor)
        nelz = max(nelz,1)
        bb = mg.BBox()
        bb.minC = [0,0,0]
        bb.maxC = shape3D(phiTensor)
        if phi is not None:
            assert phi.shape == phiFixedTensor.shape
        else: phi = torch.ones(phiFixedTensor.shape).cuda()

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
        strength = (ShapeOpt.nodeToCell(phi) > 0).double().detach()
        #initialize torch layer
        mg.initializeGPU()
        f = torch.zeros((3, nelx, nely, nelz))
        TOCLayer.reset(phiTensor.double(), phiFixedTensor.double(), f.double(), bb, lam, mu, self.tolLinear, self.maxloopLinear, self.outputDetail)
        TOCLayer.setupCurvatureFlow(self.dt, self.tau * nelx * nely * nelz)
        TOCWorstCase.compute_worst_case(strength * (E_max - E_min) + E_min, eps=1e-2, maxloop=50)
        while change > self.tolx and loop < self.maxloop:
            start = time.time()
            loop += 1

            #compute volume
            strength = (ShapeOpt.nodeToCell(phi)>0).double().detach()
            vol = torch.sum(strength).item() / strength.reshape(-1).shape[0]
            if loop == 1:
                volInit = vol
                if self.volfrac is None:
                    self.volfrac = volInit

            # FE-analysis, calculate sensitivity
            if curvatureOnly:
                dir = torch.ones(strength.shape).cuda()
                obj = 0
            else:
                strength.requires_grad_()
                obj = TOCLayer.apply(strength * (E_max - E_min) + E_min)
                # simple replacement of topological derivative
                obj.backward()
                dir = (-strength.grad.detach() * (strength * (E_max - E_min) + E_min)).detach()
            dirNode = ShapeOpt.cellToNode(dir)

            # set augmented Lagrangian parameter
            ex = self.volfrac + (volInit - self.volfrac) * max(0, 1 - loop / nvol)
            lam = torch.sum(dirNode) / dirNode.reshape(-1).shape[0] * math.exp(self.p * ((vol - ex) / ex + self.d))

            # update level set function
            phi_old = phi.clone()
            C = dir.reshape(-1).shape[0] / torch.sum(torch.abs(dirNode)).item()
            phi = TOCLayer.implicitCurvatureFlow(C * (dirNode - lam) + phi / self.dt)
            phi = torch.minimum(torch.tensor(1.), torch.maximum(torch.tensor(-1.), phi))
            change = torch.linalg.norm(phi.reshape(-1, 1) - phi_old.reshape(-1, 1), ord=float('inf')).item()
            end = time.time()
            if loop % self.outputInterval == 0:
                print("it.: {0}, obj.: {1:.3f}, vol.: {2:.3f}, ch.: {3:.3f}, time: {4:.3f}, mem: {5:.3f}Gb".format(loop,
                                                                                                                   obj,
                                                                                                                   torch.sum(
                                                                                                                       strength).item() / (
                                                                                                                           nelx * nely * nelz),
                                                                                                                   change,
                                                                                                                   end - start,
                                                                                                                   torch.cuda.memory_allocated(
                                                                                                                       None) / 1024 / 1024 / 1024))

        mg.finalizeGPU()
        return phi