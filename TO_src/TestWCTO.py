"""
Write a script to test How WC TO could modify the surface of a given model
The input is a mesh file and the output is optimized mesh file
"""
from TOCWorstCase import TOCWorseCase
from TOCLayer import TOCLayer
import libMG as mg
import torch
import numpy as np
from TopoOpt import TopoOpt as TO
from Viewer import showRho
import trimesh
import mcubes
import time
from TOUtils import *
torch.set_default_dtype(torch.float64)

class TestWCTO():
###WorstCase TO taken any input mesh file as input shape
    def __init__(self, filename, margin=5, lam = 1, mu = 1):
        self.file_name = filename
        self.h = margin
        self.rho_init, self.res, self.vol, self.phiTensor\
            ,self.phiFixedTensor, self.rhoMask = self.initialization()
        self.lam = lam
        self.mu = mu




    def WCTO(self,p=3, rmin=1.5, maxloop=200, maxloopLinear=1000, tolx=1e-3, tolLinear=1e-2, outputInterval=1, outputDetail=False):
        nelx, nely, nelz = self.res
        bb = mg.BBox()
        bb.minC = [0,0,0]
        bb.maxC = [nelx,nely,nelz]
        Ker, Ker_S = TO.filter(rmin, self.rho_init)

        print("Worst Case minimum complicance problem with OC and Random Initialization Loads")
        print(f"Number of degrees:{str(nelx)} x {str(nely)} x {str(nelz)} = {str(nelx * nely * nelz)}")
        print(f"Volume fration: {self.vol}, Penalty p: {p}, Filter radius: {rmin}")
        # max and min stiffness
        E_min = torch.tensor(1e-3)
        E_max = torch.tensor(1.0)
        change = tolx*2
        loop = 0
        g=0

        #compute filtered volume gradient (this is contant so we can precompute)
        rho = self.rho_init.detach()
        rho.requires_grad_()
        rho_filtered = TO.filter_density(Ker, rho)/Ker_S
        volume = torch.sum(rho_filtered)
        volume.backward()
        gradVolume = rho.grad.detach()

        #initialize torch layer
        mg.initializeGPU()
        Initial_f = torch.rand((3, *self.rho_init.shape)).cuda()
        TOCLayer.reset(self.phiTensor, self.phiFixedTensor, Initial_f, bb, self.lam, self.mu, tolLinear, maxloopLinear, outputDetail)
        TOCWorseCase.compute_worst_case(rho)
        while change > tolx and loop < maxloop:
            start = time.time()
            loop += 1

            #Update worstcase f only once in each iteration
            TOCWorseCase.update_worst_case(rho)


            # compute filtered objective gradient
            rho = rho.detach()
            rho.requires_grad_()
            rho_filtered = TO.filter_density(Ker, rho) / Ker_S
            obj = TOCLayer.apply(E_min + rho_filtered ** p * (E_max - E_min))
            obj.backward()
            gradObj = rho.grad.detach()

            rho_old = rho.clone()
            rho, g = TO.oc_grid(rho, gradObj, gradVolume, g, self.rhoMask)
            change = torch.linalg.norm(rho.reshape(-1, 1) - rho_old.reshape(-1, 1), ord=float('inf')).item()
            end = time.time()
            if loop % self.outputInterval == 0:
                print("it.: {0}, obj.: {1:.3f}, vol.: {2:.3f}, ch.: {3:.3f}, time: {4:.3f}, mem: {4:.3f}Gb".format(loop,
                                                                                                                   obj,
                                                                                                                   (
                                                                                                                               g + self.volfrac * nelx * nely * nelz) / (
                                                                                                                               nelx * nely * nelz),
                                                                                                                   change,
                                                                                                                   end - start,
                                                                                                                   torch.cuda.memory_allocated(
                                                                                                                       None) / 1024 / 1024 / 1024))

        mg.finalizeGPU()
        return to3DScalar(rho_old).detach().cpu().numpy()


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
        volfrac = Rho.sum() / (res[0]*res[1]*res[2])
        #Testing code for visualization the generated Rho
        # vertices, triangles = mcubes.marching_cubes(Rho, 0.99)
        # mesh = trimesh.Trimesh(vertices, triangles)
        # # _ = mesh.export('test.obj')
        # mesh.show()
        rho = torch.from_numpy(Rho).cuda()
        # rho = torch.ones_like(rho)
        nelx, nely, nelz = res
        phiTensor = -torch.ones_like(rho).cuda()
        phiFixedTensor = torch.ones((nelx + 1, nely + 1, nelz + 1)).cuda()


        # import pickle
        # file = open('data.txt', 'wb')
        # pickle.dump(rho, file)
        # file.close()
        # file = open('data.txt', 'rb')
        # # dump information to that file
        # aaa = pickle.load(file)
        # # close the file
        # file.close()

        def rhoMask(inputRho):
            pass
        return rho, res, volfrac, phiTensor, phiFixedTensor, rhoMask



if __name__ == "__main__":
    # file = './data/stanford-bunny.obj'
    file = 'TO.obj'
    TT = TestWCTO(file)
    a = TT.WCTO()
    torch.save(a, "rho_WC.pt")
    showRho(a, 0.99)