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

def Toy_Example(res=(180,60,4), volfrac=0.3):
    nelx, nely, nelz = res
    phiTensor = -torch.ones((nelx, nely, nelz)).cuda()
    phiFixedTensor = torch.ones((nelx + 1, nely + 1, nelz + 1)).cuda()
    phiFixedTensor[0,:,:] = -1
    f = torch.zeros((3, nelx + 1, nely + 1, nelz + 1)).cuda()
    f[1, -1, 0, :] = -1
    lam = 0.3 / 0.52
    mu = 1 / 2.6
    return res, volfrac, (phiTensor, phiFixedTensor, f, lam, mu)

def Support_Example(res=(128,128,64), volfrac=0.15):
    nelx, nely, nelz = res
    phiTensor = -torch.ones((nelx, nely, nelz)).cuda()
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
    return res, volfrac, (phiTensor, phiFixedTensor, f, lam, mu)

def Bridge_Example(res=(40,90,360), volfrac=0.1):
    nelx, nely, nelz = res
    supp = nelz//6
    nz = nelz // 90
    load = nelx//4
    non = nely//2
    ny = nely//20
    phiTensor = -torch.ones((nelx, nely, nelz)).cuda()
    #Non-filled space above the Load surface
    phiTensor[load:nelx-load, non:non+non//2, :] = 1
    phiFixedTensor=torch.ones((nelx + 1, nely + 1, nelz + 1)).cuda()
    #Left and Right support
    phiFixedTensor[:,0,supp:supp+nz] = -1
    phiFixedTensor[:, 0, nelz+1-supp-nz:nelz+1-supp] = -1
    f = torch.zeros((3, nelx + 1, nely + 1, nelz + 1)).cuda()
    #Load on non-design surface
    f[1,load:nelx+1-load,non,:] = -1
    # print(f[1])
    lam = 0.6
    mu = 0.4
    return res, volfrac, (phiTensor, phiFixedTensor, f, lam, mu)

if __name__ == "__main__":
    _, volfrac, params = Bridge_Example()
    sol = ShapeOpt(volfrac=volfrac, outputDetail=False)
    if not os.path.exists("phi.pt"):
        phi = sol.run(*params)
        torch.save(phi,"phi.pt")
    else: phi=torch.load("phi.pt")
    from Viewer import *
    showRhoVTK("phi", phi, False)