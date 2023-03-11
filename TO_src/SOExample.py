from ShapeOpt import ShapeOpt
import torch, os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def Support_Example(res=(128, 128, 64), volfrac=0.15, r=5):
    nelx, nely, nelz = res
    phiTensor = -torch.ones(res).cuda()
    phiFixedTensor = torch.ones((nelx + 1, nely + 1, nelz + 1)).cuda()

    n=nelx//16
    phiFixedTensor[        :n,        :n,0] = -1
    phiFixedTensor[nelx+1-n: ,        :n,0] = -1
    phiFixedTensor[nelx+1-n: ,nely+1-n: ,0] = -1
    phiFixedTensor[        :n,nely+1-n: ,0] = -1
    f = torch.zeros((3, nelx + 1, nely + 1, nelz + 1)).cuda()
    f[2, nelx//2-n:nelx//2+1+n, nely//2-n:nely//2+1+n, 0] = -1
    lam = 0.3 / 0.52
    mu = 1 / 2.6
    def rhoMask(inputRho):
        pass
    sdf = ShapeOpt.create_bubble(np.zeros(res,dtype=np.float64), min(res)//2, .4, cube=True)
    sdf = torch.from_numpy(sdf).cuda()
    return res, (sdf, phiTensor, phiFixedTensor, f, rhoMask, lam, mu)

if __name__ == "__main__":
    _, params = Support_Example(r=5)
    sol = ShapeOpt(outputDetail=False)
    if not os.path.exists("sdf.pt"):
        sdf = sol.run(*params)
        torch.save(sdf,"sdf.pt")
    else: sdf=torch.load("sdf.pt")
    from Viewer import *
    showSdf(sdf)