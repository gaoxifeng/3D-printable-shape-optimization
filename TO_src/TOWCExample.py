from TopoOptWorstCase import TopoOptWorstCase
import torch,os

def Dumbell_Example_2D(res=(360,120), volfrac=None):
    nelx, nely = res
    rho = torch.zeros(res).cuda()
    phiTensor = -torch.ones_like(rho).cuda()
    phiFixedTensor = torch.ones((nelx + 1, nely + 1)).cuda()
    for x in range(res[0]):
        for y in range(res[1]):
            if (x-90)*(x-90)+(y-60)*(y-60)<30*30:
                rho[x,y]=1
            if (x-270)*(x-270)+(y-60)*(y-60)<30*30:
                rho[x,y]=1
            if x>90 and x<270 and y>50 and y<70:
                rho[x,y]=1
    lam = 0.3 / 0.52
    mu = 1 / 2.6
    def rhoMask(inputRho):
        pass
    return res, volfrac, (rho, phiTensor, phiFixedTensor, rhoMask, lam, mu)

if __name__ == "__main__":
    _, volfrac, params = Dumbell_Example_2D()
    from TOUtils import *
    #showRhoVTK("rho",to3DScalar(params[0]).detach().cpu().numpy(),False)
    #exit(-1)
    sol = TopoOptWorstCase(volfrac=volfrac, rmin=2, outputDetail=False)
    if not os.path.exists("rho.pt"):
        rho = sol.run(*params)
        torch.save(rho,"rho.pt")
    else: rho=torch.load("rho.pt")
    from Viewer import *
    showRhoVTK("rho",rho)
    showRho(rho, 0.99)