import torch
from tqdm import tqdm
from TO_src.LevelSetShapeOptWorstCase import LevelSetShapeOptWorstCase
from TO_src.TOUtils import *
import libMG as mg
import time
import math
import os
from Viewer import *
torch.set_default_dtype(torch.float64)



class ShapeOptNeuS():
    def __init__(self, FieldRender, grid_resolution, lam=1, mu=1, Alternate_Iteration=500, mode='LSSO'):
        super().__init__()
        self.Field_R = FieldRender
        self.res_step = Alternate_Iteration
        self.mode = mode
        self.grid_res = grid_resolution
        self.lam = lam
        self.mu = mu
        self.phiTensor, self.phiFixedTensor = self.gen_paras()
        self.OptSolver = LevelSetShapeOptWorstCase(outputDetail=False, maxloop=1)
        self.scale = self.grid_res[0] / 2.02



    def run(self, img_batch_size=4, img_resolution=256):
        bound_min = -1.01
        bound_max = 1.01
        # self.load_state(f'ckpt_3_iteration.pth')
        # phi = ShapeOptNeuS.extract_fields(bound_min, bound_max, self.grid_res,
        #                                   lambda pts: -self.Field_R.sdf_network.sdf(pts))
        # showRhoVTK(f"phi_11", phi.detach().cpu().numpy(), False)
        # phi = phi*self.scale
        # params = (self.phiTensor, self.phiFixedTensor, self.lam, self.mu, phi)
        # phi_wc = self.OptSolver.run(*params)
        # showRhoVTK(f"phi_12", phi_wc, False)
        # showRhoVTK(f"phi_13", phi_wc/self.scale, False)
# neus + wcso
# zhi zuo yi ci wc
        self.Field_R.train(img_batch_size, img_resolution, 2000)
        phi = ShapeOptNeuS.extract_fields(bound_min, bound_max, self.grid_res,
                                          lambda pts: -self.Field_R.sdf_network.sdf(pts))
        phi = phi * self.scale
        showRhoVTK(f"phi_input_init", phi.detach().cpu().numpy(), False)
        params = (self.phiTensor, self.phiFixedTensor, self.lam, self.mu, phi)
        phi_wc = self.OptSolver.run(*params)
        showRhoVTK(f"phi_output_init", phi_wc, False)
        for i in range(20):
            phi, img_para = self.Field_R.train_sdf(phi_wc/self.scale, self.grid_res, img_batch_size, img_resolution, self.res_step)
            # phi = ShapeOptNeuS.extract_fields(bound_min, bound_max, self.grid_res, lambda pts: -self.Field_R.sdf_network.sdf(pts))
            phi = phi*self.scale
            showRhoVTK(f"phi_input_{i}", phi.detach().cpu().numpy(), False)
            params = (self.phiTensor, self.phiFixedTensor, self.lam, self.mu, phi)
            self.save_state(i, img_para)
            phi_wc = self.OptSolver.run(*params)
            showRhoVTK(f"phi_output_{i}", phi_wc, False)
        mg.finalizeGPU()

    def extract_fields(bound_min, bound_max, resolution, query_func):
        N = 64
        X = torch.linspace(bound_min, bound_max, resolution[0]+1).split(N)
        Y = torch.linspace(bound_min, bound_max, resolution[1]+1).split(N)
        Z = torch.linspace(bound_min, bound_max, resolution[2]+1).split(N)

        # map the bound_min and bound_max to 0,nelxyz

        u = torch.zeros((resolution[0]+1, resolution[1]+1, resolution[2]+1))
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs)
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                        val = query_func(pts).reshape(len(xs), len(ys), len(zs))
                        u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
        return u

    def gen_paras(self):
        nelx, nely, nelz = self.grid_res
        phiTensor = -torch.ones((nelx, nely, nelz)).cuda()
        phiFixedTensor = torch.ones((nelx + 1, nely + 1, nelz + 1)).cuda()
        return phiTensor, phiFixedTensor

    def save_state(self,i, img_para):
        self.Field_R.save_checkpoint(i)
        # self.Field_R.render_image_Neus(*img_para, i)
        pass

    def load_state(self, network_checkpoint_name):
        self.Field_R.load_checkpoint(checkpoint_name=network_checkpoint_name)
        pass