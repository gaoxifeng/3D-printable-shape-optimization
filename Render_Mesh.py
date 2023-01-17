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
        self.out_dir = 'Result_Nvdiff/' + out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        # os.makedirs(os.path.join(self.out_dir, "mesh_Mesh"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "images_Mesh"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "masks_Mesh"), exist_ok=True)

    def render(self, mvp, campos, lightpos, resolution, iter_i, mesh_scale=2.0):
        # lightpos = util.cosine_sample(campos) * RADIUS #This is not used when we generate the images
        params = {'mvp': mvp, 'lightpos': lightpos, 'campos': campos, 'resolution': [resolution, resolution], 'time': 0}
        _opt_ref = mesh.center_by_reference(self.render_ref_mesh.eval(params), self.ref_mesh_aabb, mesh_scale)
        with torch.no_grad():
            light_power = 0
            color_ref, mask_ref = render.render_mesh(self.glctx, _opt_ref, mvp, campos, lightpos, light_power, resolution,
                                           background=None)
        if iter_i % 1000 == 0:
            for i in range(color_ref.shape[0]):
                np_result_image = color_ref[i].detach().cpu().numpy()
                np_mask_image = mask_ref[i].detach().cpu().numpy()
                util.save_image(self.out_dir + '/images_Mesh/' + ('train_%06d_%03d.png' % (iter_i,i)), np_result_image)
                util.save_image(self.out_dir + '/masks_Mesh/' + ('mask_%06d_%03d.png' % (iter_i,i)), np_mask_image)
        # _opt_ref.material = None
        # obj.write_obj(self.out_dir, _opt_ref)
        return color_ref, mask_ref
    # remove the unused parameters

    def get_camera_rays_at_pixel(self, img, x, y, mv, p):
        """
        Translated from https: // github.com / gaoxifeng / TinyVisualizer / blob / main / TinyVisualizer / Camera3D.cpp
        line 122-138
        """
        #img shape: B x H x W x 3
        H = img.shape[1]
        W = img.shape[2]

        ratioX = (x - W/2)/(W/2)
        ratioY = (y - H/2)/(H/2)

        Rot = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0,0,0,1]])
        mv = Rot @ mv

        dir = np.vstack((ratioX, -ratioY, 0, 1))
        dir = np.linalg.inv(p) @ dir
        dir[:3] = dir[:3]/dir[-1]
        ray = np.zeros((6,1))
        ray[:3,:] = -mv[:3, :3].T @ mv[:3, [-1]]
        ray[3:, :] = mv[:3, :3].T @ dir[:3]
        return ray

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

    images, mask = Render.render(mvp, campos, lightpos, resolution, 1000)

    #verify that rays are recovered from the correct pixels
    pts = np.zeros((1,3))
    dt = np.linspace(0.01, 5, 512)
    for i in range(10):
        x = np.random.randint(0,resolution-1)
        y = np.random.randint(0,resolution-1)
        print(x, y, images[0, x, y]) # print each ray and its rendered image pixel value
        ray = Render.get_camera_rays_at_pixel(images, x, y, r_mv, proj_mtx) # get ray_o and ray_v
        mm = ray[:3, [0]].T + (dt * ray[3:, [0]]).T #generate the points on this ray with mm = ray_o + dt*ray_v
        pts = np.vstack((pts, mm))

    with open(f'Result_Nvdiff/{out_dir}/rays.obj', "w") as f:
        print("    writing %d vertices" % pts.shape[0])
        for i in range(1,pts.shape[0]):
            f.write('v {} {} {} \n'.format(pts[i,0], pts[i,1], pts[i,2]))



# N C H W

if __name__ == "__main__":
    main(5, 'data/f16/f16.obj', 'F16', 512)
    # r > g r>b
    # for x y
    #     rgb = img[x,y]
    # add = figure(rgb)




