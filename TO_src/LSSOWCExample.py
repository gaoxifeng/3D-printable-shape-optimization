from LevelSetShapeOptWorstCase import LevelSetShapeOptWorstCase
import torch,os

def Dumbell_Example_2D(res=(360,120)):
    nelx, nely = res
    phiTensor = -torch.ones(res).cuda()
    phiFixedTensor = torch.ones((nelx + 1, nely + 1)).cuda()
    phi = torch.ones([nelx+1, nely+1]).cuda()
    for x in range(res[0]):
        for y in range(res[1]):
            if (x-90)*(x-90)+(y-60)*(y-60)<30*30:
                phi[x,y]=-1
            if (x-270)*(x-270)+(y-60)*(y-60)<30*30:
                phi[x,y]=-1
            if x>90 and x<270 and y>50 and y<70:
                phi[x,y]=-1
    lam = 0.3 / 0.52
    mu = 1 / 2.6
    return res, (phiTensor, phiFixedTensor, lam, mu, phi)

if __name__ == "__main__":
    _, params = Dumbell_Example_2D()
    sol = LevelSetShapeOptWorstCase(outputDetail=False)
    if not os.path.exists("phi.pt"):
        phi = sol.run(*params)
        torch.save(phi,"phi.pt")
    else: phi=torch.load("phi.pt")
    from Viewer import *
    showRhoVTK("phi", phi, False)