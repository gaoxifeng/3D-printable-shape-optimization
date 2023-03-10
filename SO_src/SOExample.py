from ShapeOpt import ShapeOpt
import torch, os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
def create_bubble_square(arr_size, r):
    coords = np.ogrid[:arr_size[0], :arr_size[1], :arr_size[2]]

    init_upper = np.ceil(np.array(arr_size)/(3*r))
    init = np.random.randint((0,0,0),init_upper)
    centers = []
    for i in range(init[0],arr_size[0], 3*r):
        for j in range(init[1], arr_size[1], 3 * r):
            for k in range(init[2], arr_size[2], 3 * r):
                centers.append((i,j,k))
    distance = np.zeros(arr_size)

    for i in range(len(centers)):
        temp = np.sqrt(
            (coords[0] - centers[i][0]) ** 2 + (coords[1] - centers[i][1]) ** 2 + (coords[2] - centers[i][2]) ** 2) - r
        temp[(temp > 0)] = 0
        distance +=temp
    distance[(distance < 0)] = 1
    distance[(distance == 0)] = -1

    return distance

def Toy_Example(res=(180,60,20), volfrac=0.3, r = 5):
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
    sdf = create_bubble_square(res, r)


    sdf = torch.from_numpy(sdf).cuda()
    return res, (sdf, phiTensor, phiFixedTensor, f, rhoMask, lam, mu, volfrac)




def Support_Example(res=(128, 128, 64), volfrac=0.15, r=5):
    nelx, nely, nelz = res
    rho = torch.ones(res).cuda() * volfrac
    phiTensor = -torch.ones_like(rho).cuda()
    phiFixedTensor = torch.ones((nelx + 1, nely + 1, nelz + 1)).cuda()

    n = nelx // 16
    phiFixedTensor[:n, :n, 0] = -1
    phiFixedTensor[nelx + 1 - n:, :n, 0] = -1
    phiFixedTensor[nelx + 1 - n:, nely + 1 - n:, 0] = -1
    phiFixedTensor[:n, nely + 1 - n:, 0] = -1
    f = torch.zeros((3, nelx + 1, nely + 1, nelz + 1)).cuda()
    f[2, nelx // 2 - n:nelx // 2 + 1 + n, nely // 2 - n:nely // 2 + 1 + n, 0] = -1
    lam = 0.3 / 0.52
    mu = 1 / 2.6

    def rhoMask(inputRho):
        pass

    sdf = create_bubble_square(res, r)

    sdf = torch.from_numpy(sdf).cuda()
    return res, (sdf, phiTensor, phiFixedTensor, f, rhoMask, lam, mu, volfrac)



def Bridge_Example(res=(40,90,360), volfrac=0.1, r = 5):
    nelx, nely, nelz = res
    rho = torch.ones(res).cuda()*volfrac
    # rho = torch.load("rho_bg.pt")
    # rho = torch.from_numpy(rho).cuda()
    supp = nelz//6
    nz = nelz // 90
    load = nelx//4
    non = nely//2
    ny = nely//20

    phiTensor = -torch.ones_like(rho).cuda()
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
    def rhoMask(inputRho):
        inputRho[:, non-ny:non, :] = 0
        inputRho[:, non-ny+1:non-1, :] = -1
        inputRho[[0,-1], non - ny:non, :] = 0
        inputRho[:, non - ny:non, [-1,0]] = 0
        inputRho[load:nelx-load, non:non+non//2, :] = 1

    sdf = create_bubble_square(res, r)
    sdf = torch.from_numpy(sdf).cuda()
    sdf[:, non-ny:non, :] = 0
    sdf[:, non - ny+1:non-1, :] = -1
    sdf[[0, -1], non - ny:non, :] = 0
    sdf[:, non - ny:non, [-1, 0]] = 0
    sdf[load:nelx-load, non:non+non//2, :] = 1
    return res, (sdf, phiTensor, phiFixedTensor, f, rhoMask, lam, mu, volfrac)




if __name__ == "__main__":
    _, params = Bridge_Example(r=5)
    sol = ShapeOpt(s=1, rmin=3, outputDetail=False, maxloop=200)
    # # if not os.path.exists("rho.pt"):
    rho = sol.run(*params)
    torch.save(rho, "rho.pt")
    # else:
    rho = torch.load("rho.pt")
    from Viewer import *

    # showRhoVTK("rho", rho)
    showRho(rho,-1e-5)