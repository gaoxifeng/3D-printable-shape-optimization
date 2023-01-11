import os
import torch
import numpy as np
from Nvdiff_src import obj
from Nvdiff_src import util
from Nvdiff_src import mesh
from Nvdiff_src import texture
from Nvdiff_src import render
from Nvdiff_src import regularizer
from Nvdiff_src.mesh import Mesh
import nvdiffrast.torch as dr

RADIUS = 3.5


def load_mesh(filename, mtl_override=None):
    name, ext = os.path.splitext(filename)
    if ext == ".obj":
        return obj.load_obj(filename, clear_ks=True, mtl_override=mtl_override)
    assert False, "Invalid mesh file extension"


class Render_Mesh():
    def __init__(self, mesh_dir, out_dir, mtl_override=None):
        # texture files required
        self.ref_mesh = load_mesh(mesh_dir, mtl_override)
        print("Base mesh has %d triangles and %d vertices." % (
        self.ref_mesh.t_pos_idx.shape[0], self.ref_mesh.v_pos.shape[0]))

        self.render_ref_mesh = mesh.compute_tangents(self.ref_mesh)
        # Compute AABB of reference mesh. Used for centering during rendering
        self.ref_mesh_aabb = mesh.aabb(self.render_ref_mesh.eval())

        self.glctx = dr.RasterizeGLContext()
        self.out_dir = 'Result/' + out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        # os.makedirs(os.path.join(self.out_dir, "mesh_Mesh"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "images_Mesh"), exist_ok=True)

    def render(self, mvp, campos, lightpos, resolution, mesh_scale=2.0):
        # lightpos = util.cosine_sample(campos) * RADIUS #This is not used when we generate the images
        params = {'mvp': mvp, 'lightpos': lightpos, 'campos': campos, 'resolution': [resolution, resolution], 'time': 0}
        _opt_ref = mesh.center_by_reference(self.render_ref_mesh.eval(params), self.ref_mesh_aabb, mesh_scale)
        with torch.no_grad():
            light_power = 0
            color_ref = render.render_mesh(self.glctx, _opt_ref, mvp, campos, lightpos, light_power, resolution,
                                           background=None)
        for i in range(color_ref.shape[0]):
            np_result_image = color_ref[i].detach().cpu().numpy()
            util.save_image(self.out_dir + '/images/' + ('train_%06d.png' % i), np_result_image)

        return color_ref


def main(Batch_size, mesh_dir, out_dir, resolution):
    Render = Render_Mesh(mesh_dir=mesh_dir, out_dir=out_dir)
    proj_mtx = util.projection(x=0.4, f=1000.0)
    mvp = np.zeros((Batch_size, 4, 4), dtype=np.float32)
    campos = np.zeros((Batch_size, 3), dtype=np.float32)
    lightpos = np.zeros((Batch_size, 3), dtype=np.float32)

    # We still take the lightpos as input, but we output the same rgb intensity for different views
    for b in range(Batch_size):
        # Random rotation/translation matrix for optimization.
        r_rot = util.random_rotation_translation(0.25)  # (4,4)
        r_mv = np.matmul(util.translate(0, 0, -RADIUS), r_rot)  # (4,4)
        mvp[b] = np.matmul(proj_mtx, r_mv).astype(np.float32)  # (B,4,4)
        campos[b] = np.linalg.inv(r_mv)[:3, 3]  # (B,3)
        lightpos[b] = util.cosine_sample(campos[b])  # (B,3)

    images = Render.render(mvp, campos, lightpos, resolution)

    params = {}


# N C H W

if __name__ == "__main__":
    main(1, 'data/f16/f16.obj', 'F16', 512)



