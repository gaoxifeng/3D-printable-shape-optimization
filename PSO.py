from TopologyOp import TopoOpt
from Render_Fields_modified import Render_Fields
import torch
import numpy as np

def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles
"""
Step 1: Given any input mesh file obj, use NVidiff to prepare its masked rendering images

Step 2: Use the images to train NeuS network, and take another predefined Grid to optimize the shape

Step 3: Use the alternating way to train the NeuS network and do the TO alternatively.

Step 4: Optimize the final Mesh with some Regularization tools or methods for smooth surface reconstruction.
"""



class ADMM_3DPSO(torch.nn.Module):
    def __init__(self, MeshRender, FieldRender, TOptimizer):
        super().__init__()

        """
        Prepare two different sets of data:
        The first one is generated by the MeshRender and is used for our FieldRender
        The second one is generated by authors and is used for TOptimizer to optimize the shape or infill rate directly
        Combine these two parts then we could formulate our final loss functions and use a joint way to optimize by ADAM.
        """





    def Alterating_train(self, steps, swap_steps):

        return None


    def Generate_mesh(self, SDF):
        #Return a mesh generated from the NeuS SDF
        return None

    def Visualization(self, original_mesh, generated_mesh):
        #Render two meshes together to compare the results of our TO
        #And the ideal case should be that the output of our method could outperform the original mesh in rendering image
        return None