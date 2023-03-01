from TO_OC import TopoOpt
import torch,os

def Toy_Example(res=(180,60,4), volfrac=0.3):
    nelx, nely, nelz = res
    rho = torch.ones(res).cuda()*volfrac
    phiTensor = -torch.ones_like(rho).cuda()
    phiFixedTensor = torch.ones((nelx + 1, nely + 1, nelz + 1)).cuda()
    phiFixedTensor[0,:,:] = -1
    f = torch.zeros((3, nelx + 1, nely + 1, nelz + 1)).cuda()
    f[1, -1, 0, :] = -1
    lam = 0.3 / 0.52
    mu = 1 / 2.6
    def rhoMask(inputRho):
        pass
    return res, volfrac, (rho, phiTensor, phiFixedTensor, f, rhoMask, lam, mu)

def Support_Example(res=(128,128,64), volfrac=0.15):
    nelx, nely, nelz = res
    rho = torch.ones(res).cuda()*volfrac
    phiTensor = -torch.ones_like(rho).cuda()
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
    return res, volfrac, (rho, phiTensor, phiFixedTensor, f, rhoMask, lam, mu)

if __name__ == "__main__":
    _, volfrac, params = Support_Example()
    sol = TopoOpt(volfrac, outputDetail=False)
    if not os.path.exists("rho.pt"):
        rho = sol.run(*params)
        torch.save(rho,"rho.pt")
    else: rho=torch.load("rho.pt")
    TopoOpt.show(rho, 0.99)