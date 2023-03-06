from ShapeOpt import ShapeOpt
import torch, os


def Toy_Example(res=(180, 60, 4), volfrac=0.3):
    nelx, nely, nelz = res
    rho = torch.ones(res).cuda() * volfrac
    phiTensor = -torch.ones_like(rho).cuda()
    phiFixedTensor = torch.ones((nelx + 1, nely + 1, nelz + 1)).cuda()
    phiFixedTensor[0, :, :] = -1
    f = torch.zeros((3, nelx + 1, nely + 1, nelz + 1)).cuda()
    f[1, -1, 0, :] = -1
    lam = 0.3 / 0.52
    mu = 1 / 2.6
    s = 0.5  #parameter s in Phi_s (f(x))

    def rhoMask(inputRho):
        pass

    return res, volfrac, (rho, s, phiTensor, phiFixedTensor, f, rhoMask, lam, mu)


if __name__ == "__main__":
    _, volfrac, params = Toy_Example()
    sol = ShapeOpt(volfrac, rmin=2, outputDetail=False, maxloop=10)
    # if not os.path.exists("rho.pt"):
    rho = sol.run(*params)
    torch.save(rho, "rho.pt")
    # else:
        # rho = torch.load("rho.pt")
    from Viewer import *

    showRhoVTK("rho", rho)
    showRho(rho, 0.99)