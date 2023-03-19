from TO_src.LevelSetShapeOptWorstCaseApproximation import LevelSetShapeOptWorstCaseApproximation
import torch,os
import numpy as np

def Dumbell_Example_3D(res=(100, 100, 100)):
    nelx, nely, nelz = res
    phiTensor = -torch.ones(res).cuda()
    phiFixedTensor = torch.ones((nelx + 1, nely + 1, nelz + 1)).cuda()
    phi = torch.ones([nelx+1, nely+1, nelz+1]).cuda()

    center = [30, 30, 30]
    width = 10
    X, Y, Z = np.meshgrid(np.arange(0, 101), np.arange(0, 101), np.arange(0, 101))
    dist = (X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2
    cube = np.ones_like(dist)
    cube[dist<=width**2] = -1

    center2 = [70, 70, 70]
    dist = (X - center2[0])**2 + (Y - center2[1])**2 + (Z - center2[2])**2
    cube2 = np.zeros_like(dist)
    cube2[dist<=width**2] = -2



    Ball = cube + cube2
    phi0 = torch.from_numpy(Ball).double().cuda()

    dist_init = (X - res[0]//2) ** 2 + (Y - res[1]//2) ** 2 + (Z - res[2]//2) ** 2
    R_min = np.maximum(np.sqrt((center2[0]-res[0]//2)**2+(center2[1]-res[1]//2)**2+(center2[2]-res[2]//2)**2), np.sqrt((center[0]-res[0]//2)**2+(center[1]-res[1]//2)**2+(center[2]-res[2]//2)**2))
    Init = np.ones_like(dist_init)
    Init[dist_init<=(R_min+width+3)**2] = -1
    phi = torch.from_numpy(Init).double().cuda()
    lam = 0.3 / 0.52
    mu = 1 / 2.6
    return res, (phiTensor, phiFixedTensor, lam, mu, phi, phi0)

if __name__ == "__main__":
    _, params = Dumbell_Example_3D()
    sol = LevelSetShapeOptWorstCaseApproximation(outputDetail=False, maxloop=500)
    phi = sol.run(*params)
    torch.save(phi,"phi.pt")
    # phi=torch.load("phi.pt")
    from TO_src.Viewer import *
    import mcubes
    import trimesh
    vertices, triangles = mcubes.marching_cubes(phi, 0.01)
    mesh = trimesh.Trimesh(vertices, triangles)
    # _ = mesh.export('test.obj')
    mesh.show()
    # showRhoVTK("phi", phi, False)