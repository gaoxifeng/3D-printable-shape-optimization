import torch
import numpy as np
import os
import logging
import argparse
import cv2 as cv
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from pyhocon import ConfigFactory
from NeuS_src.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from NeuS_src.renderer import NeuSRenderer
from Nvdiff_src import util
from Render_Mesh import Render_Mesh
from torch.utils.tensorboard import SummaryWriter
from Grid_renderer import Grid_renderer
RADIUS = 3.5
"""
Generate a framework to run it first and then try to switch it to traditional format
"""

def Initial_Fields(Field_Shape=[64,64,64], Method='Traditional'):
    if Method == 'Traditional':
        SDF = torch.from_numpy(np.random.random(Field_Shape)).float()
        SDF.requires_grad = True
        CF = torch.from_numpy(np.zeros(Field_Shape)).float()
        CF.requires_grad = True
    elif Method == 'Network':
        SDF = torch.randn(Field_Shape)
        CF = torch.randn(Field_Shape)
    return SDF, CF

def get_camera_rays_at_pixel(H, W, x, y, mv, p):
    """
    Translated from https: // github.com / gaoxifeng / TinyVisualizer / blob / main / TinyVisualizer / Camera3D.cpp
    line 122-138
    """
    #img shape: B x H x W x 3
    # H = img.shape[1]
    # W = img.shape[2]

    ratioX = (x - W/2)/(W/2)
    ratioY = (y - H/2)/(H/2)

    Rot = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    mv = Rot @ mv

    dir = np.vstack((ratioX, -ratioY, 0, 1))
    dir = np.linalg.inv(p) @ dir
    dir[:3] = dir[:3]/dir[-1]
    ray = np.zeros((6,1))
    ray[:3,:] = -mv[:3, :3].T @ mv[:3, [-1]]
    ray[3:, :] = mv[:3, :3].T @ dir[:3]
    return ray


def near_far_from_sphere(rays_o, rays_d):
    a = torch.sum(rays_d**2, dim=-1, keepdim=True)
    b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
    mid = 0.5 * (-b) / a
    near = mid - 1.0
    far = mid + 1.0
    return near, far

class Render_Fields(torch.nn.Module):
    def __init__(self, mesh_dir, out_dir,conf_path, Field_Shape=[1,4,4, 8, 16], Method='Traditional', mode='train', case='CASE_NAME'):
        super().__init__()

        self.method = Method

        if Method == 'Traditional':
            self.out_dir = 'Result_Grid/' + out_dir
            self.base_exp_dir = 'Result_Grid/' + out_dir + 'exp'
            os.makedirs(self.base_exp_dir, exist_ok=True)
            self.end_iter = 200000
            self.batch_size = 2048
            self.mask_weight = 0.01
            self.igr_weight = 0.1
            self.anneal_end = 0

            # FD = torch.from_numpy(np.random.random(Field_Shape)).float().cuda()
            FD = torch.zeros(Field_Shape).cuda()
            self.grid_renderer = Grid_renderer(FD,
                                          n_samples=64,
                                          n_importance = 64,
                                          n_outside = 0,
                                          up_sample_steps = 4,
                                          perturb = 1.0,
                                          variance= 0.2)
            self.optimizer = torch.optim.Adam(self.grid_renderer.parameters(), lr=0.001)



        elif Method == 'Network':
            self.out_dir = 'Result_NeuS/' + out_dir
            self.device = torch.device('cuda')

            # Configuration
            self.conf_path = conf_path
            f = open(self.conf_path)
            conf_text = f.read()
            conf_text = conf_text.replace('CASE_NAME', case)
            f.close()

            self.conf = ConfigFactory.parse_string(conf_text)
            self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
            self.base_exp_dir = self.conf['general.base_exp_dir']
            os.makedirs(self.base_exp_dir, exist_ok=True)


            # Training parameters
            self.end_iter = self.conf.get_int('train.end_iter')
            self.save_freq = self.conf.get_int('train.save_freq')
            self.report_freq = self.conf.get_int('train.report_freq')
            self.val_freq = self.conf.get_int('train.val_freq')
            self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
            self.batch_size = self.conf.get_int('train.batch_size')
            self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
            self.learning_rate = self.conf.get_float('train.learning_rate')
            self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
            self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
            self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
            self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

            # Weights
            self.igr_weight = self.conf.get_float('train.igr_weight')
            self.mask_weight = self.conf.get_float('train.mask_weight')
            self.mode = mode
            self.model_list = []
            self.writer = None

            # Networks
            params_to_train = []
            self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
            self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
            self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
            self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
            params_to_train += list(self.nerf_outside.parameters())
            params_to_train += list(self.sdf_network.parameters())
            params_to_train += list(self.deviation_network.parameters())
            params_to_train += list(self.color_network.parameters())

            self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

            self.renderer = NeuSRenderer(self.nerf_outside,
                                         self.sdf_network,
                                         self.deviation_network,
                                         self.color_network,
                                         **self.conf['model.neus_renderer'])

        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "mesh_Field"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "images_Field"), exist_ok=True)
        self.iter_step = 0

        self.render_mesh = Render_Mesh(mesh_dir=mesh_dir, out_dir=out_dir)



        # self.SDF = torch.nn.Parameter(SDF)
        # self.CF = torch.nn.Parameter(CF)

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))
    def gen_rays_training(self,img, mask, mv, p):
        B, H, W, _ = img.shape
        idx_image = torch.randint(low=0, high=B, size=[self.batch_size]).cpu()
        pixels_x = torch.randint(low=0, high=H, size=[self.batch_size]).cpu()
        pixels_y = torch.randint(low=0, high=W, size=[self.batch_size]).cpu()

        true_rgb = img[(idx_image,pixels_x, pixels_y)]
        masks  = mask[(idx_image,pixels_x, pixels_y)].mean(dim=1).unsqueeze(1)
        rays = []
        for i in range(self.batch_size):
            rays.append(get_camera_rays_at_pixel(H, W, pixels_x[i], pixels_y[i], mv[idx_image[i]], p))
        rays = np.array(rays)
        rays = torch.from_numpy(rays).float().squeeze(2).cuda()
        rays_o = rays[:,:3]
        rays_v = rays[:,3:]
        return rays_o, rays_v, true_rgb, masks

    def gen_rays_validating(self, H, W, mv, p, img_N):
        tx = torch.linspace(0, W - 1, W)
        ty = torch.linspace(0, H - 1, H)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        ppp = torch.stack([pixels_x, pixels_y], dim=-1).reshape(-1,2).detach().cpu()
        rays = []
        for i in range(ppp.shape[0]):
            rays.append(get_camera_rays_at_pixel(H, W, ppp[i,0], ppp[i,1], mv[img_N], p))
        rays = np.array(rays)
        rays = torch.from_numpy(rays).float().squeeze(2).cuda()
        rays_o = rays[:,:3]
        rays_v = rays[:,3:]
        return rays_o, rays_v

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])
    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor
    def train(self, img_batch_size, resolution):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        if self.method == 'Network':
            self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        # image_perm = self.get_image_perm()
        Batch_size = img_batch_size
        H = resolution
        W = resolution
        for iter_i in tqdm(range(res_step)):
            # print(iter_i)
            proj_mtx = util.projection(x=0.4, f=1000.0)
            mvp = np.zeros((Batch_size, 4, 4), dtype=np.float32)
            campos = np.zeros((Batch_size, 3), dtype=np.float32)
            lightpos = np.zeros((Batch_size, 3), dtype=np.float32)
            r_mv = np.zeros((Batch_size, 4, 4),dtype=np.float32)
            # We still take the lightpos as input, but we output the same rgb intensity for different views
            for b in range(Batch_size):
                # Random rotation/translation matrix for optimization.
                r_rot = util.random_rotation_translation(0.25)  # (4,4)
                r_mv[b] = np.matmul(util.translate(0, 0, -RADIUS), r_rot)  # (4,4)
                mvp[b] = np.matmul(proj_mtx, r_mv[b]).astype(np.float32)  # (B,4,4)
                campos[b] = np.linalg.inv(r_mv[b])[:3, 3]  # (B,3)
                lightpos[b] = util.cosine_sample(campos[b])  # (B,3)

            img_rgb, mask_rgb = self.render_mesh.render(mvp, campos, lightpos, resolution, iter_i)


            #####Generate the rays_o, rays_v, masks from different images here
            rays_o, rays_v, true_rgb, mask = self.gen_rays_training(img_rgb, mask_rgb, r_mv, proj_mtx)

            # with torch.no_grad():
            #     rays_o -= rays_o.min(1, keepdim=True)[0]
            #     rays_o /= rays_o.max(1, keepdim=True)[0]
            #     rays_o = (rays_o - 0.5) / 0.5
            #
            #     rays_v -= rays_v.min(1, keepdim=True)[0]
            #     rays_v /= rays_v.max(1, keepdim=True)[0]
            #     rays_v = (rays_v - 0.5) / 0.5

            del img_rgb, mask_rgb
            near, far = near_far_from_sphere(rays_o, rays_v)



            # true_rgb[:,[0]] = torch.ones_like(true_rgb[:,[0]])
            # true_rgb[:, [1]] = torch.zeros_like(true_rgb[:, [1]])
            # true_rgb[:, [2]] = torch.zeros_like(true_rgb[:, [2]])
            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5

            if self.method == 'Traditional':
                render_out = self.grid_renderer.render(rays_o, rays_v, near, far,
                                                  background_rgb=None,
                                                  cos_anneal_ratio=self.get_cos_anneal_ratio())
            elif self.method == 'Network':
                render_out = self.renderer.render(rays_o, rays_v, near, far,
                                                  background_rgb=None,
                                                  cos_anneal_ratio=self.get_cos_anneal_ratio())

            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']

            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            # psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)



            if self.method == 'Traditional':
                loss = color_fine_loss + \
                       eikonal_loss * self.igr_weight + \
                       mask_loss * self.mask_weight
            else:
                loss = color_fine_loss + \
                       eikonal_loss * self.igr_weight + \
                       mask_loss * self.mask_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            # self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            # self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            # self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            # self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)
            # if self.method == 'Traditional':
            #     render_out = self.grid_renderer.render(rays_o, rays_v, near, far,
            #                                       background_rgb=None,
            #                                       cos_anneal_ratio=self.get_cos_anneal_ratio())

            if iter_i%2500==0:
                self.render_image(H, W, r_mv, proj_mtx, 1, iter_i)
                # print(f'{loss.item()},\n')
            if self.iter_step % 200 == 0:
                print('iter:{:8>d} loss = {}'.format(self.iter_step, loss))

            if self.method == 'Network':
                if self.iter_step % self.report_freq == 0:
                    print(self.base_exp_dir)
                    print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

                if self.iter_step % self.save_freq == 0:
                    self.save_checkpoint()

                # if self.iter_step % self.val_freq == 0:
                #     self.validate_image()

                # if self.iter_step % self.val_mesh_freq == 0:
                #     self.validate_mesh()

                self.update_learning_rate()


    def render_image(self, H, W, mv, p, img_N, iter_i):
        """
        render image with a input campos
        """
        # ???cam_o, ray_v, rgb, 1???
        # Batch * 10
        # Batch * 3 predicted rgb
        rays_o, rays_v = self.gen_rays_validating(H, W, mv, p, img_N)
        rays_o = rays_o.split(self.batch_size)
        rays_d = rays_v.split(self.batch_size)

        # out_rgb_fine = []
        # for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
        #     near, far = near_far_from_sphere(rays_o_batch, rays_d_batch)
        #     background_rgb = None
        #
        #     render_out = self.renderer.render(rays_o_batch,
        #                                       rays_d_batch,
        #                                       near,
        #                                       far,
        #                                       background_rgb=background_rgb,
        #                                       cos_anneal_ratio=self.get_cos_anneal_ratio())
        #
        #     out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
        #     del render_out
        # img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        # loc = self.out_dir+ '/images_Field/'+('train_%06d_%03d.png' % (iter_i, img_N))
        # util.save_image(loc, img_fine)

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = None

            if self.method == 'Traditional':
                render_out = self.grid_renderer.render(rays_o_batch, rays_d_batch, near, far,
                                                  background_rgb=None,
                                                  cos_anneal_ratio=self.get_cos_anneal_ratio())

                def feasible(key):
                    return (key in render_out) and (render_out[key] is not None)

                if feasible('color_fine'):
                    out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
                del render_out
            elif self.method == 'Network':
                render_out = self.renderer.render(rays_o_batch, rays_d_batch, near, far,
                                                  background_rgb=None,
                                                  cos_anneal_ratio=self.get_cos_anneal_ratio())
            # render_out = self.renderer.render(rays_o_batch,
            #                                   rays_d_batch,
            #                                   near,
            #                                   far,
            #                                   cos_anneal_ratio=self.get_cos_anneal_ratio(),
            #                                   background_rgb=background_rgb)

                def feasible(key):
                    return (key in render_out) and (render_out[key] is not None)

                if feasible('color_fine'):
                    out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
                if feasible('gradients') and feasible('weights'):
                    n_samples = self.renderer.n_samples + self.renderer.n_importance
                    normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                    if feasible('inside_sphere'):
                        normals = normals * render_out['inside_sphere'][..., None]
                    normals = normals.sum(dim=1).detach().cpu().numpy()
                    out_normal_fine.append(normals)
                del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        # normal_img = None
        # if len(out_normal_fine) > 0:
        #     normal_img = np.concatenate(out_normal_fine, axis=0)
        #     rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
        #     normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
        #                   .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        # os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)
        loc = self.out_dir + '/images_Field/' + ('train_%06d_%03d.png' % (iter_i, img_N))
        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(loc, cv.cvtColor(img_fine[..., i], cv.COLOR_RGB2BGR))





if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/thin_structure.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='bmvs_bear')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    args.conf = './confs/wmask_f16.conf'
    args.field_shape = [1, 4, 64,64,64]
    # args.method = 'Network'
    args.method = 'Traditional'
    args.mode = 'train'
    args.case = 'f16'
    args.mesh_dir = 'data/f16/f16.obj'
    args.out_dir = 'F16'
    runner = Render_Fields(args.mesh_dir, args.out_dir, args.conf, args.field_shape, args.method, args.mode, args.case)
    runner.train(4, 512)



    # Field_Shape = [10,60, 60, 60]
    # SS, CF = Initial_Fields(Field_Shape)
    # Render_F=Render_Fields('F16',Field_Shape)
    # mvp = torch.randn([10,4,4])
    #
    #
    # Image_Ref = torch.randn([10,60,60,60])
    # optimizer_Adam = optim.Adam(Render_F.parameters(), lr=0.001)
    # SDF, CF = Render_F.ptsd()
    # print(2*SDF[0]+CF[0]-Image_Ref[0])
    # for i in range(3000):
    #     SSP = torch.clone(SS)
    #     Image_Field = Render_F(mvp, 2, 3)
    #     print(i)
    #     optimizer_Adam.zero_grad()
    #     Loss = torch.nn.L1Loss()(Image_Ref, Image_Field)
    #     print(Loss)
    #     Loss.backward()
    #     optimizer_Adam.step()
    # SDF, CF = Render_F.ptsd()
    # print(2*SDF[0]+CF[0]-Image_Ref[0])

    # Render_F = Render_Fields('F16', Field_Shape)
    # cam_pos = torch.randn(1,3)
    # p = Render_F.render_image(cam_pos, 512, 1)
    # rays_o, rays_v = gen_rays_from_params(cam_pos, 512,4)
    # print(rays_v.shape)
