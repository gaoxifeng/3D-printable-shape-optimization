import libMG as mg
from SOLayer import SOLayer
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ShapeOpt import ShapeOpt
torch.set_default_dtype(torch.float64)


class Flow():
    def __init__(self, size, center, max_iter=50, rmin=1.5):
        self.arr_size = size
        self.center = center
        self.rmin = rmin
        self.max_iter = max_iter

    def reset_params(self, grid):
        nelx, nely, nelz = grid.shape
        V = nelx * nely * nelz
        Ker, Ker_S = ShapeOpt.filter(self.rmin, grid)
        phiTensor = -torch.ones_like(grid).cuda()
        phiFixedTensor = torch.ones((nelx + 1, nely + 1, nelz + 1)).cuda()
        f = torch.zeros((3, nelx + 1, nely + 1, nelz + 1)).cuda()
        bb = mg.BBox()
        bb.minC = [0, 0, 0]
        bb.maxC = [nelx, nely, nelz]
        return Ker.cuda(), Ker_S.cuda(), V, (phiTensor, phiFixedTensor, f, bb, 1, 1, 1e-2, 200, False)

    def create_bin_sphere(arr_size, center, r):
        coords = np.ogrid[:arr_size[0], :arr_size[1], :arr_size[2]]
        distance = np.sqrt(
            (coords[0] - center[0]) ** 2 + (coords[1] - center[1]) ** 2 + (coords[2] - center[2]) ** 2) - r
        return distance

    def create_bin_cube(arr_size, center, r):
        coords = np.ogrid[:arr_size[0], :arr_size[1], :arr_size[2]]
        distance = (np.abs(coords[0] - center[0]) + np.abs(coords[1] - center[1]) + np.abs(
            coords[2] - center[2])) - r
        return distance
        ##Expansion works but shrink not working???
    def circle_to_square(self, r_circle, s=1, CFL=1):
        sphere = Flow.create_bin_sphere(self.arr_size, self.center, r_circle)
        sphere = torch.from_numpy(sphere).cuda()
        cube = Flow.create_bin_cube(self.arr_size, self.center, r_circle * np.pi ** (1 / 3))
        # cube = Flow.create_bin_sphere(self.arr_size, self.center, r_circle*2)
        cube = torch.from_numpy(cube).cuda()

        Ker, Ker_S, V, params = self.reset_params(sphere)
        mg.initializeGPU()
        SOLayer.reset(*params)

        #redistance the inputs and set up the target volume
        cube = SOLayer.redistance(cube)
        H = ShapeOpt.Heaviside(cube, 1e-3, 1)
        H_filtered = ShapeOpt.filter_density(Ker, H) / Ker_S
        Vol_cube = H_filtered.sum() / V

        sdf = SOLayer.redistance(sphere)

        #Find Lambda and then update the sdf
        for i in range(self.max_iter):
            sdf = sdf.detach()
            sdf.requires_grad_()
            H = ShapeOpt.Heaviside(sdf, 1e-3, s)
            H_filtered = ShapeOpt.filter_density(Ker, H) / Ker_S
            vol = torch.sum(H_filtered)
            vol.backward()
            gradVol = sdf.grad.detach()

            sdf = sdf.detach()
            sdf.requires_grad_()
            Obj = torch.norm(cube - sdf)
            Obj.backward()
            gradObj = sdf.grad.detach()

            sdf, vol = ShapeOpt.find_lambda(sdf, s, 1e-3, gradObj, gradVol, Vol_cube, CFL, Ker, Ker_S)
            sdf = SOLayer.redistance(sdf)

        mg.finalizeGPU()
        return sphere.detach().cpu().numpy(), cube.detach().cpu().numpy(), sdf.detach().cpu().numpy()


if __name__ == "__main__":
    size = (65, 65, 65)
    center = (32, 32, 32)
    r = 10
    F = Flow(size, center, max_iter=20)
    init, target, result = F.circle_to_square(r)

    slice = init[32, :, :]
    fig1, ax2 = plt.subplots(layout='constrained')
    CS = ax2.contourf(slice, 10, cmap=plt.cm.bone)
    cbar = fig1.colorbar(CS)
    ax2.set_title(f'Initial sdf at the center')

    slice = target[32, :, :]
    fig1, ax2 = plt.subplots(layout='constrained')
    CS = ax2.contourf(slice, 10, cmap=plt.cm.bone)
    cbar = fig1.colorbar(CS)
    ax2.set_title(f'Target sdf at the center')

    slice = result[32, :, :]
    fig1, ax2 = plt.subplots(layout='constrained')
    CS = ax2.contourf(slice, 10, cmap=plt.cm.bone)
    cbar = fig1.colorbar(CS)
    ax2.set_title(f'Result sdf at the center')
    plt.show()