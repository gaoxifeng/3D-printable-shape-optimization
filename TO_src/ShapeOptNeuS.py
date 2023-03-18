import torch
from tqdm import tqdm
from TO_src.LevelSetShapeOptWorstCase import LevelSetShapeOptWorstCase
from TO_src.TOUtils import *
import libMG as mg
import time
import math
import os
torch.set_default_dtype(torch.float64)



class ShapeOptNeuS():
    def __init__(self, FieldRender, grid_resolution, lam=1, mu=1, Alternate_Iteration=50, mode='LSSO'):
        super().__init__()
        self.Field_R = FieldRender
        self.res_step = Alternate_Iteration
        self.mode = mode
        self.grid_res = grid_resolution
        self.lam = lam
        self.mu = mu
        self.phiTensor, self.phiFixedTensor = self.gen_paras()
        self.OptSolver = LevelSetShapeOptWorstCase(outputDetail=False)



    def run(self, img_batch_size=4, img_resolution=128):
        bound_min = [0,0,0]
        bound_max = self.grid_res
        # self.load_state('ckpt_0_iteration.pth')
        for i in range(5):
            self.Field_R.train(img_batch_size, img_resolution, self.res_step)
            phi = ShapeOptNeuS.extract_fields(bound_min, bound_max, self.grid_res, lambda pts: -self.Field_R.sdf_network.sdf(pts))
            params = (self.phiTensor, self.phiFixedTensor, self.lam, self.mu, phi)
            self.save_state(i)
            self.OptSolver.run(*params)

    def extract_fields(bound_min, bound_max, resolution, query_func):
        N = 64
        X = torch.linspace(bound_min[0], bound_max[0], resolution[0]+1).split(N)
        Y = torch.linspace(bound_min[1], bound_max[1], resolution[1]+1).split(N)
        Z = torch.linspace(bound_min[2], bound_max[2], resolution[2]+1).split(N)

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

    def save_state(self,i):
        self.Field_R.save_checkpoint(i)
        pass

    def load_state(self, network_checkpoint_name):
        self.Field_R.load_checkpoint(checkpoint_name=network_checkpoint_name)
        pass