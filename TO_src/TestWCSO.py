"""
Write a script to test How WC SO could modify the surface of a given model
The input is a mesh file and the output is optimized mesh file
"""
from TOCWorstCase import TOCWorseCase
from TOCLayer import TOCLayer
from TOLayer import TOLayer
import libMG as mg
import math
import torch
import numpy as np
from TOLayer import TOLayer
from ShapeOpt import ShapeOpt as SO
from Viewer import *
import trimesh
import mcubes
import time
from TOUtils import *
torch.set_default_dtype(torch.float64)

class TestWCSO():
###WorstCase SO taken any input mesh file as input shape
    def __init__(self, filename, margin=5, lam = 1, mu = 1):
        self.file_name = filename
        self.h = margin
        self.rho_init, self.res, self.volfrac, self.phiTensor\
            ,self.phiFixedTensor, self.rhoMask = self.initialization()
        self.lam = lam
        self.mu = mu




    def WCSO(self,dt=0.1, tau=1e-4, p=4, d=-0.02, maxloop=200, maxloopLinear=1000, tolx=1e-3, tolLinear=1e-2, outputInterval=1, outputDetail=False, curvatureOnly=False):
        nelx, nely, nelz = self.res
        bb = mg.BBox()
        bb.minC = [0,0,0]
        bb.maxC = [nelx,nely,nelz]

        print("Level-set shape optimization problem")
        print(f"Number of degrees:{str(nelx)} x {str(nely)} x {str(nelz)} = {str(nelx * nely * nelz)}")
        print(f"Volume fration: {self.volfrac}")
        # max and min stiffness
        E_min = torch.tensor(1e-3)
        E_max = torch.tensor(1.0)
        change = tolx*2
        nvol = maxloop // 2
        loop = 0
        g=0

        phi = -self.rho_init.clone()
        strength = (SO.nodeToCell(phi) > 0).double().detach()
        #initialize torch layer
        mg.initializeGPU()
        Initial_f = torch.rand((3, *self.rho_init.shape)).cuda()
        showRhoVTK('fWCPhi_init_SO', phi.detach().cpu().numpy())

        TOCLayer.reset(self.phiTensor, self.phiFixedTensor, Initial_f, bb, self.lam, self.mu, tolLinear, maxloopLinear, outputDetail)
        TOCLayer.setupCurvatureFlow(dt, tau * nelx * nely * nelz)
        TOCWorseCase.compute_worst_case(strength * (E_max - E_min) + E_min, eps=1e-2, maxloop=50)
        print_f = TOCLayer.b.detach().cpu().numpy()
        showFMagnitudeCellVTK("fWorstCell_SO", print_f)
        while change > tolx and loop < maxloop:
            start = time.time()
            loop += 1

            #compute volume
            strength = (SO.nodeToCell(phi)>0).double().detach()
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
            dirNode = SO.cellToNode(dir)

            # set augmented Lagrangian parameter
            ex = self.volfrac + (volInit - self.volfrac) * max(0, 1 - loop / nvol)
            lam = torch.sum(dirNode) / dirNode.reshape(-1).shape[0] * math.exp(p * ((vol - ex) / ex + d))

            # update level set function
            phi_old = phi.clone()
            C = dir.reshape(-1).shape[0] / torch.sum(torch.abs(dirNode)).item()
            phi = TOCLayer.implicitCurvatureFlow(C * (dirNode - lam) + phi / dt)
            phi = torch.minimum(torch.tensor(1.), torch.maximum(torch.tensor(-1.), phi))
            change = torch.linalg.norm(phi.reshape(-1, 1) - phi_old.reshape(-1, 1), ord=float('inf')).item()
            end = time.time()
            if loop % outputInterval == 0:
                print("it.: {0}, obj.: {1:.3f}, vol.: {2:.3f}, ch.: {3:.3f}, time: {4:.3f}, mem: {5:.3f}Gb".format(loop,
                                                                                                                   obj,
                                                                                                                   torch.sum(
                                                                                                                       strength).item() / (
                                                                                                                               nelx * nely * nelz),
                                                                                                                   change,
                                                                                                                   end - start,
                                                                                                                   torch.cuda.memory_allocated(
                                                                                                                       None) / 1024 / 1024 / 1024))
            # rho = torch.maximum(rho, torch.tensor(0.001))
            # TOCWorseCase.compute_worst_case(rho)
            # TOCWorseCase.update_worst_case(rho)
        showFMagnitudeCellVTK("fWorstCell_final_SO", TOCLayer.b.detach().cpu().numpy())
        showRhoVTK('fWCPhi_SO',phi.detach().cpu().numpy())
        mg.finalizeGPU()
        return to3DScalar(phi_old).detach().cpu().numpy()


    def initialization(self):
        #Prepare the bisic information for WorstCase TO
        temp_file = trimesh.load(self.file_name)
        Indicator = temp_file.voxelized(1).encoding.dense.copy()
        res = np.array(Indicator.shape)+2*self.h
        #I have to map the BBox to real number of the resolution and get the bounding box
        #Not implemented yet
        Rho = (1e-3)*np.ones(res)
        data = 1*(Indicator == True)
        Rho[self.h:self.h+Indicator.shape[0],self.h:self.h+Indicator.shape[1],self.h:self.h+Indicator.shape[2]] = data
        Rho = np.maximum(Rho, 0.001)
        volfrac = Rho.sum() / (res[0]*res[1]*res[2])
        #Testing code for visualization the generated Rho
        # vertices, triangles = mcubes.marching_cubes(Rho, 0.99)
        # mesh = trimesh.Trimesh(vertices, triangles)
        # # _ = mesh.export('test.obj')
        # mesh.show()

        Rho = self.dumbbell_SO()
        Rho = np.minimum(Rho, 1)
        res = Rho.shape
        # volfrac = Rho.sum() / (res[0] * res[1] * res[2])

        rho = torch.from_numpy(Rho).cuda()
        nelx, nely, nelz = res
        nelx = nelx-1
        nely = nely-1
        nelz = nelz-1
        phiTensor = -torch.ones((nelx, nely, nelz)).cuda()
        phiFixedTensor = torch.ones((nelx + 1, nely + 1, nelz + 1)).cuda()

        def rhoMask(inputRho):
            pass
        return rho, res, None, phiTensor, phiFixedTensor, rhoMask

    def dumbbell_SO(self):
        center = [15, 15, 15]
        width = 10
        X, Y, Z = np.meshgrid(np.arange(0, 61), np.arange(0, 61), np.arange(0, 61))
        dist = np.max([np.abs(X - center[0]), np.abs(Y - center[1]), np.abs(Z - center[2])], axis=0)
        cube = np.ones_like(dist)
        cube[dist <= width / 2] = -2

        center2 = [45, 45, 15]
        dist = np.max([np.abs(X - center2[0]), np.abs(Y - center2[1]), np.abs(Z - center2[2])], axis=0)
        cube2 = np.ones_like(dist)
        cube2[dist <= width / 2] = -2

        Cube = cube + cube2

        for i in range(center[0], center2[0]):
            for j in range(-3, 3):
                for k in range(center[2] - 2, center[2] + 2):
                    # if i+j<=center[0]+center2:
                    Cube[i, i + j, k] = -1
        return Cube.astype(np.float64)

if __name__ == "__main__":
    # file = './data/stanford-bunny.obj'
    file = 'dumbbell.obj'
    TT = TestWCSO(file)
    a = TT.WCSO(maxloop=10)
    torch.save(a, "dumbbell_SO.pt")
    # a = torch.load("dumbbell_SO.pt")
    showRho(a, 0.99)
