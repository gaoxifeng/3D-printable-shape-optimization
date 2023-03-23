from TO_src.LevelSetShapeOptWorstCaseApproximation import LevelSetShapeOptWorstCaseApproximation
import torch,os
import numpy as np
import pysdf
import skimage

def TWOballs_Example_3D(res=(100, 100, 100)):
    nelx, nely, nelz = res
    phiTensor = -torch.ones(res).cuda()
    phiFixedTensor = torch.ones((nelx + 1, nely + 1, nelz + 1)).cuda()
    phi = torch.ones([nelx+1, nely+1, nelz+1]).cuda()

    center = [40, 40, 40]
    width = 10
    X, Y, Z = np.meshgrid(np.arange(0, 101), np.arange(0, 101), np.arange(0, 101))
    dist = (X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2
    cube = np.ones_like(dist)
    cube[dist<=width**2] = -1

    center2 = [60, 60, 40]
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

def Dumbbell_Example_3D(res=(100, 100, 100)):
    nelx, nely, nelz = res
    phiTensor = -torch.ones(res).cuda()
    phiFixedTensor = torch.ones((nelx + 1, nely + 1, nelz + 1)).cuda()
    phi = torch.ones([nelx+1, nely+1, nelz+1]).cuda()

    center = [40, 40, 40]
    width = 10
    X, Y, Z = np.meshgrid(np.arange(0, 101), np.arange(0, 101), np.arange(0, 101))
    dist = (X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2
    cube = np.ones_like(dist)
    cube[dist<=width**2] = -1

    center2 = [60, 60, 40]
    dist = (X - center2[0])**2 + (Y - center2[1])**2 + (Z - center2[2])**2
    cube2 = np.zeros_like(dist)
    cube2[dist<=width**2] = -2



    Ball = cube + cube2
    for i in range(center[0], center2[0]):
        for j in range(-2, 2):
            for k in range(center[2] - 1, center[2] + 1):
                # if i+j<=center[0]+center2:
                Ball[i, i + j, k] = -1
    # import mcubes
    # import trimesh
    # vertices, triangles = mcubes.marching_cubes(Ball, -0.99)
    # mesh = trimesh.Trimesh(vertices, triangles)
    # # _ = mesh.export('test.obj')
    # mesh.show()
    phi0 = torch.from_numpy(Ball).double().cuda()
    center3 = np.array(center)//2+np.array(center2)//2
    dist_init = (X - center3[0]) ** 2 + (Y - center3[1]) ** 2 + (Z - center3[2]) ** 2
    R_min = np.maximum(np.sqrt((center2[0]-center3[0])**2+(center2[1]-center3[1])**2+(center2[2]-center3[2])**2), np.sqrt((center[0]-center3[0])**2+(center[1]-center3[1])**2+(center[2]-center3[2])**2))
    Init = np.ones_like(dist_init)
    Init[dist_init<=(R_min+width+3)**2] = -1
    phi = torch.from_numpy(Init).double().cuda()
    lam = 0.3 / 0.52
    mu = 1 / 2.6
    return res, (phiTensor, phiFixedTensor, lam, mu, phi, phi0)

def Dumbbell_Example_3D_case2(res=(100, 100, 100)):
    nelx, nely, nelz = res
    phiTensor = -torch.ones(res).cuda()
    phiFixedTensor = torch.ones((nelx + 1, nely + 1, nelz + 1)).cuda()
    phi = torch.ones([nelx+1, nely+1, nelz+1]).cuda()

    center = [10, 10, 40]
    width = 5
    X, Y, Z = np.meshgrid(np.arange(0, 101), np.arange(0, 101), np.arange(0, 101))
    dist = (X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2
    cube = np.ones_like(dist)
    cube[dist<=width**2] = -1

    center2 = [90, 90, 40]
    dist = (X - center2[0])**2 + (Y - center2[1])**2 + (Z - center2[2])**2
    cube2 = np.zeros_like(dist)
    cube2[dist<=width**2] = -2



    Ball = cube + cube2
    for i in range(center[0], center2[0]):
        for j in range(-2, 2):
            for k in range(center[2] - 2, center[2] + 2):
                # if i+j<=center[0]+center2:
                Ball[i, i + j, k] = -1
    # import mcubes
    # import trimesh
    # vertices, triangles = mcubes.marching_cubes(Ball, -0.99)
    # mesh = trimesh.Trimesh(vertices, triangles)
    # # _ = mesh.export('test.obj')
    # mesh.show()
    phi0 = torch.from_numpy(Ball).double().cuda()
    center3 = np.array(center)//2+np.array(center2)//2
    dist_init = (X - center3[0]) ** 2 + (Y - center3[1]) ** 2 + (Z - center3[2]) ** 2
    R_min = np.maximum(np.sqrt((center2[0]-center3[0])**2+(center2[1]-center3[1])**2+(center2[2]-center3[2])**2), np.sqrt((center[0]-center3[0])**2+(center[1]-center3[1])**2+(center[2]-center3[2])**2))
    Init = np.ones_like(dist_init)
    Init[dist_init<=(R_min+width+3)**2] = -1
    phi = torch.from_numpy(Init).double().cuda()
    lam = 0.3 / 0.52
    mu = 1 / 2.6
    return res, (phiTensor, phiFixedTensor, lam, mu, phi0, phi0)

# def combine_two_mesh():
#     mesh = trimesh.load('aaa.obj')
#
#     sdf = pysdf.SDF(mesh.vertices, mesh.faces)
#
#     vertices, triangles = mcubes.marching_cubes(sdf, 0.0)
#     mesh = trimesh.Trimesh(vertices, triangles)
#     # _ = mesh.export('test_init.obj')
#     mesh.show()


if __name__ == "__main__":

    from TO_src.Viewer import *
    import mcubes
    import trimesh

    # _, params = Dumbbell_Example_3D()
    # vertices, triangles = mcubes.marching_cubes(params[-1].detach().cpu().numpy(), 0.0)
    # mesh = trimesh.Trimesh(vertices, triangles)
    # _ = mesh.export('test_init.obj')
    # mesh.show()




    # _, params = TWOballs_Example_3D()
    _, params = TWOballs_Example_3D()
    sol = LevelSetShapeOptWorstCaseApproximation(dt=0.001, tau=1e-6,outputDetail=False, maxloop=200)
    phi = sol.run(*params)
    torch.save(phi,"phi.pt")
    # phi=torch.load("phi.pt")
    #
    # vertices, triangles = mcubes.marching_cubes(phi, 0.01)
    # mesh = trimesh.Trimesh(vertices, triangles)
    # _ = mesh.export('test.obj')
    # mesh.show()
    showRhoVTK("phi", phi, False)