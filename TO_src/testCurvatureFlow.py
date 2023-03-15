from TOExample import *
from ShapeOpt import *
from Viewer import *

def Toy_Example_2D(res=(360,120), volfrac=0.4):
    nelx, nely = res
    phiTensor = -torch.ones((nelx, nely)).cuda()
    phiFixedTensor = torch.ones((nelx + 1, nely + 1)).cuda()
    phiFixedTensor[0,:] = -1
    f = torch.zeros((2, nelx + 1, nely + 1)).cuda()
    f[1, -1, 0] = -1
    lam = 0.3 / 0.52
    mu = 1 / 2.6
    return res, volfrac, (phiTensor, phiFixedTensor, f, lam, mu)

_, volfrac, params = Toy_Example_2D()
sol = ShapeOpt(volfrac=volfrac, tau=1e-2, outputDetail=False)
phi = sol.run(*params)
showRhoVTK("phi", phi, False)