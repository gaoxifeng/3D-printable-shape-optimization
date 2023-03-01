from TO_OC import TopoOpt
import torch,os

def Toy_Example(res=(180,60,32), volfrac=0.3):
    nelx, nely, nelz = res
    rho = torch.ones(res).cuda()*volfrac
    phiTensor = -torch.ones_like(rho).cuda()
    phiFixedTensor=torch.ones((nelx + 1, nely + 1, nelz + 1)).cuda()
    phiFixedTensor[0,:,:]=-1
    f = torch.zeros((3, nelx + 1, nely + 1, nelz + 1)).cuda()
    f[1, -1, 0, :] = -1
    lam = 0.3 / 0.52
    mu = 1 / 2.6
    def rhoMask(inputRho):
        pass
    return rho, phiTensor, phiFixedTensor, f, rhoMask, lam, mu

if __name__ == "__main__":
    params = Toy_Example()
    sol = TopoOpt(0.3, outputDetail=False)
    if not os.path.exists("rho.pt"):
        rho = sol.run(*params)
        torch.save(rho,"rho.pt")
    else: rho=torch.load("rho.pt")
    TopoOpt.show(rho)
    
    #x = 0.15*torch.ones((40,90,360),dtype=torch.float64).cuda()
    #TTT = TopoOpt(0.15)
    #paras = TTT.Bridge_Example(x)
    #p = TTT.Opt3D_Grid_MGGPU(x, paras, 3, 10, 50)

    # bridge = 0.15 * torch.ones((40,90,360),dtype=torch.float64).cuda()
    # p = TTT.Opt3D_Grid_MGGPU_Bridge(bridge, 0.15, 3, 10, 500)