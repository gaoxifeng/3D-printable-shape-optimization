from TO_OC import TopoOpt
import torch

def Toy_Example(res=(180,60,4), volfrac=0.3):
    nelx, nely, nelz = res
    rho = torch.ones(res).cuda()*volfrac
    phiTensor = -torch.ones_like(rho).cuda()
    phiFixedTensor=torch.ones((nelx + 1, nely + 1, nelz + 1)).cuda()
    phiFixedTensor[0,:,:]=-1
    f = torch.zeros((3, nelx + 1, nely + 1, nelz + 1)).cuda()
    f[1, -1, 0, :] = -1
    
    Nonphi1 = torch.ones_like(rho)
    Nonphi2 = torch.zeros_like(rho)
    Targetphi = torch.ones_like(rho)
    lam = 0.3 / 0.52
    mu = 1 / 2.6
    return rho, phiTensor, phiFixedTensor, f, Nonphi1, Nonphi2, Targetphi, lam, mu

def Bridge_Example(res, volfrac):
    nelx, nely, nelz = rho.clone().detach().cpu().numpy().shape
    phiTensor = -torch.ones_like(rho).cuda()
    phiTensor[9:29,44:69,:] = 1

    phiFixedTensor=torch.ones((nelx + 1, nely + 1, nelz + 1)).cuda()
    phiFixedTensor[:,0,57:61] = -1
    phiFixedTensor[:, 0, 297:301] = -1

    f = torch.zeros((3, nelx + 1, nely + 1, nelz + 1)).cuda()
    f[1,9:29,44,:] = -2

    Nonphi1 = torch.ones_like(rho)
    Nonphi1[:, 40:44, :] = 0
    Nonphi2 = torch.zeros_like(rho)
    Nonphi2[:, 40:44, :] = 1
    Targetphi = torch.tensor(1.0)
    lam = 0.6
    mu = 0.4
    return phiTensor, phiFixedTensor, f, Nonphi1, Nonphi2, Targetphi, lam, mu

if __name__ == "__main__":
    params = Toy_Example()
    sol = TopoOpt(0.3, outputDetail=False)
    rho = sol.run(*params)
    TopoOpt.show(rho)
    
    #x = 0.15*torch.ones((40,90,360),dtype=torch.float64).cuda()
    #TTT = TopoOpt(0.15)
    #paras = TTT.Bridge_Example(x)
    #p = TTT.Opt3D_Grid_MGGPU(x, paras, 3, 10, 50)

    # bridge = 0.15 * torch.ones((40,90,360),dtype=torch.float64).cuda()
    # p = TTT.Opt3D_Grid_MGGPU_Bridge(bridge, 0.15, 3, 10, 500)