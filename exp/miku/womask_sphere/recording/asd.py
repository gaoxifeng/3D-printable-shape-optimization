# import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
from mesh_to_sdf import mesh_to_voxels

import trimesh
import skimage
import skimage.measure
import mcubes
mesh = trimesh.load('./data/bunny/bunny.obj')

voxels = mesh_to_voxels(mesh, 60, pad=False)
vertices, triangles = mcubes.marching_cubes(voxels, 0)
mcubes.export_obj(vertices, triangles, 'sphere.obj')
# vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
# mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
# mesh.show()