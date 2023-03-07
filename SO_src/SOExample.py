from ShapeOpt import ShapeOpt
import torch, os
import torch.nn.functional as F

def Toy_Example(res=(60, 40, 4), volfrac=0.3):
    nelx, nely, nelz = res
    rho = torch.ones(res).cuda() * volfrac
    phiTensor = -torch.ones_like(rho).cuda()
    phiFixedTensor = torch.ones((nelx + 1, nely + 1, nelz + 1)).cuda()
    phiFixedTensor[0, :, :] = -1
    f = torch.zeros((3, nelx + 1, nely + 1, nelz + 1)).cuda()
    f[1, -1, 0, :] = -1
    lam = 0.3 / 0.52
    mu = 1 / 2.6
    volfrac = 0.3  #parameter s in Phi_s (f(x))
    rho = GenerateInitSDF(res)
    def rhoMask(inputRho):
        pass

    return res, (rho, phiTensor, phiFixedTensor, f, rhoMask, lam, mu, volfrac)

def GenerateInitSDF(res=(60, 40, 10), step_size=3, length=2):
    nelx, nely, nelz = res
    sdf = torch.ones(res).cuda()
    for i in range(1, nelx-length, step_size):
        for j in range(1, nely-length, step_size):
            for k in range(1, nelz-length, step_size):
                sdf[i:i+length,j:j+length,k:k+length] = torch.tensor(-1).cuda()
    return sdf







if __name__ == "__main__":
    _, params = Toy_Example(volfrac=0.3)
    # H zai 2 dao 5 ge gezijian bian
    # s = log 2 or log 5
    # sdf * 2
    # s / 2

    sol = ShapeOpt(s=0.5, rmin=2, outputDetail=False, maxloop=10)
    # if not os.path.exists("rho.pt"):
    rho = sol.run(*params)
    torch.save(rho, "rho.pt")
    # else:
        # rho = torch.load("rho.pt")
    from Viewer import *

    showRhoVTK("rho", rho)
    showRho(rho, 0.00)