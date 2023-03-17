from LevelSetShapeOpt import *
from Viewer import *

def Cube_Example_2D(res=(180,180)):
    nelx, nely = res
    phiTensor = -torch.ones((nelx, nely)).cuda()
    phiFixedTensor = torch.ones((nelx + 1, nely + 1)).cuda()
    phiFixedTensor[0,:] = -1
    f = torch.zeros((2, nelx + 1, nely + 1)).cuda()
    phi = torch.ones_like(phiFixedTensor).cuda()
    for x in range(nelx//4,nelx*3//4):
        for y in range(nely//4,nely*3//4):
            phi[x,y]=-1
    lam = 0.3 / 0.52
    mu = 1 / 2.6
    return res, (phiTensor, phiFixedTensor, f, lam, mu, phi)

if __name__ == "__main__":
    _, params = Cube_Example_2D()
    sol = LevelSetShapeOpt(outputDetail=False)
    if not os.path.exists("phi.pt"):
        phi = sol.run(*params, curvatureOnly=True)
        torch.save(phi,"phi.pt")
    else: phi=torch.load("phi.pt")
    from Viewer import *
    showRhoVTK("phi", phi, False)