from TO_src.LevelSetShapeOptWorstCaseApproximation import LevelSetShapeOptWorstCaseApproximation
import torch,os
import numpy as np
# import mesh_to_sdf as mts
# import skimage

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

def Dumbbell_Example_3D(res=(200, 200, 100), r=2, l=1):
    nelx, nely, nelz = res
    phiTensor = -torch.ones(res).cuda()
    phiFixedTensor = torch.ones((nelx + 1, nely + 1, nelz + 1)).cuda()
    phi = torch.ones([nelx+1, nely+1, nelz+1]).cuda()

    center = [60, 60, 50]
    width = 30
    X, Y, Z = np.meshgrid(np.arange(0, nelx+1), np.arange(0, nely+1), np.arange(0, nelz+1))
    dist = (X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2
    cube = np.ones_like(dist)
    cube[dist<=width**2] = -1

    center2 = [140, 140, 50]
    dist = (X - center2[0])**2 + (Y - center2[1])**2 + (Z - center2[2])**2
    cube2 = np.zeros_like(dist)
    cube2[dist<=width**2] = -2



    Ball = cube + cube2
    for i in range(center[0], center2[0]):
        for j in range(-r, r):
            for k in range(center[2] - r, center[2] + r):
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
    Init[dist_init<=(R_min+width+5)**2] = -1
    Init[:,:,:15] = 1
    Init[:,:,-15:] = 1
    phi = torch.from_numpy(Init).double().cuda()
    lam = 0.3 / 0.52
    mu = 1 / 2.6

    center = [60, 60, 50]
    width = 30
    X, Y, Z = np.meshgrid(np.arange(0, nelx+1), np.arange(0, nely+1), np.arange(0, nelz+1))
    dist = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
    cube = np.ones_like(dist)
    cube[dist <= width ** 2] = -1

    center2 = [140, 140, 50]
    dist = (X - center2[0]) ** 2 + (Y - center2[1]) ** 2 + (Z - center2[2]) ** 2
    cube2 = np.zeros_like(dist)
    cube2[dist <= width ** 2] = -2

    Ball = cube + cube2
    Ball[Ball==1] = -2
    Ball[Ball== -1] = 1
    for i in range(center[0], center2[0]):
        for j in range(-r-l, r+l):
            for k in range(center[2] - r-l, center[2] + r+l):
                # if i+j<=center[0]+center2:
                Ball[i, i + j, k] = 1

    Mask = torch.from_numpy(Ball).double().cuda()
    return res, (phiTensor, phiFixedTensor, lam, mu, phi, phi0, Mask)

def Dumbbell_Example_3D_case2(res=(55, 55, 55), r=2):
    nelx, nely, nelz = res
    phiTensor = -torch.ones(res).cuda()
    phiFixedTensor = torch.ones((nelx + 1, nely + 1, nelz + 1)).cuda()
    phi = torch.ones([nelx+1, nely+1, nelz+1]).cuda()
    # phiFixedTensor[:30,:30,60:]=-1
    # phiFixedTensor[60:, :30,60:] = -1
    # phiFixedTensor[:30,:30,:10]=-1
    # phiFixedTensor[10:40,: ,10:40] = 1

    center = [15, 25, 25]
    width = 10
    X, Y, Z = np.meshgrid(np.arange(0, 56), np.arange(0, 56), np.arange(0, 56))
    dist = (X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2
    cube = np.ones_like(dist)
    cube[dist<=width**2] = -1
    ff = np.zeros((3,56,56,56))
    temp = np.zeros_like(dist)
    temp[dist<=width**2] = 1
    ff[0,:,:,:] = temp

    center2 = [40, 25, 25]
    dist = (X - center2[0])**2 + (Y - center2[1])**2 + (Z - center2[2])**2
    cube2 = np.zeros_like(dist)
    cube2[dist<=width**2] = -2
    temp2 = np.zeros_like(dist)
    temp2[dist<=width**2] = -1
    ff[0,:,:,:] = ff[0,:,:,:]+temp2


    Ball = cube + cube2
    if r==0:
        pass
    else:
        for i in range(center[0], center2[0]):
            for j in range(-r, r):
                for k in range(center[2] - r, center[2] + r):
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
    lam = 1
    mu = 10
    return res, (phiTensor, phiFixedTensor, lam, mu, ff, phi0)

def combine_two_mesh():
    mesh = trimesh.load('data/f16/f16.obj')

    sdf = mts.mesh_to_voxels(mesh, 400, pad=False)
    torch.save(sdf, "data/f16/f16.pt")
    vertices, triangles = mcubes.marching_cubes(sdf, 0.0)
    mesh = trimesh.Trimesh(vertices, triangles)
    # _ = mesh.export('test_init.obj')
    mesh.show()

# write a function here to generate a sdf model which has a mapping mask for evaluating the original sdf and
#     the desired output for an optimization goal for our final target,


if __name__ == "__main__":

    from TO_src.Viewer import *
    import mcubes
    import trimesh

    # _, params = Dumbbell_Example_3D()
    # vertices, triangles = mcubes.marching_cubes(params[-1].detach().cpu().numpy(), 0.0)
    # mesh = trimesh.Trimesh(vertices, triangles)
    # _ = mesh.export('test_init.obj')
    # mesh.show()




    _, params = Dumbbell_Example_3D(r=2, l=2)
    # # _, params = combine_two_mesh()
    sol = LevelSetShapeOptWorstCaseApproximation(h=1, dt=0.1, tau=1e-7,outputDetail=False, maxloop=600)
    phi, rho = sol.run(*params)
    torch.save(phi,"phi.pt")
    torch.save(rho, "rho.pt")


    rho=torch.load("phi.pt")
    #
    vertices, triangles = mcubes.marching_cubes(params[-2].detach().cpu().numpy(), 0.01)
    mesh = trimesh.Trimesh(vertices, triangles)
    _ = mesh.export('test_init.obj')
    # mesh.show()
    # rho = mcubes.smooth(rho)
    # rho = mcubes.smooth(rho)
    vertices, triangles = mcubes.marching_cubes(rho, -0.1)
    mesh = trimesh.Trimesh(vertices, triangles)
    _ = mesh.export('test_out.obj')
    showRhoVTK("phi", phi, False)