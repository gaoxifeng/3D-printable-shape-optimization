from ShapeOpt import *
from Viewer import *

res = [128,128]
phiTensor = -torch.ones(tuple(res)).cuda()
phiFixedTensor = torch.ones(tuple([r+1 for r in res])).cuda()
f = torch.zeros(tuple([2]+[r+1 for r in res])).cuda()
phi = -torch.ones(tuple([r+1 for r in res])).cuda()
for x in range(32,96):
    for y in range(32,96):
        phi[x,y] = 1

sol = ShapeOpt(volfrac=None, tau=0.1)
phiNew = sol.run(phiTensor, phiFixedTensor, f, 1, 1, phi, curvatureOnly = True)
showRhoVTK('curvatureFlow', phiNew, False)