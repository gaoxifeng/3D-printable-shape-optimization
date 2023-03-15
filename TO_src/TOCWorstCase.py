import torch,time
import numpy as np
import libMG as mg
from TOCLayer import TOCLayer

class TOCWorseCase:
    def update_worst_case(rho):
        #inverse power method
        TOCLayer.u = TOCLayer.solveK(rho)
        TOCLayer.nodeToCell(rho)
        TOCLayer.b = (TOCLayer.b / torch.norm(TOCLayer.b.reshape(-1))).detach()
        return TOCLayer.b
        
    def compute_worst_case(rho, eps=1e-2, maxloop=100, outputInterval=1):
        TOCLayer.b = torch.rand(TOCLayer.b.shape).cuda()
        change = eps*2
        loop = 0
        #inverse power loop
        while change > eps and loop < maxloop:
            start = time.time()
            loop += 1
            
            b_old = TOCLayer.b.clone()
            TOCWorseCase.update_worst_case(rho)
            change = torch.linalg.norm(TOCLayer.b.reshape(-1,1) - b_old.reshape(-1,1), ord='fro').item()
            
            end = time.time()
            if loop%outputInterval == 0:
                print("it.: {0}, ch.: {1:.3f}, time: {2:.3f}, mem: {3:.3f}Gb".format(loop, change, end - start, torch.cuda.memory_allocated(None)/1024/1024/1024))
                
        return TOCLayer.b
      
def debug(iter=0, DTYPE=torch.float64):
    bb=mg.BBox()
    bb.minC=[-1,-1,-1]
    bb.maxC=[1,1,1]
    res=[64,64,64]
    
    def interp1D(a,b,alpha):
        return a*(1-alpha)+b*alpha
    def interpC(x,y,z,bb):
        id=[x,y,z]
        for d in range(3):
            id[d]=interp1D(bb.minC[d],bb.maxC[d],(id[d]+0.5)/res[d])
        return np.array(id)
    def interpV(x,y,z,bb):
        id=[x,y,z]
        for d in range(3):
            id[d]=interp1D(bb.minC[d],bb.maxC[d],id[d]/res[d])
        return np.array(id)
    def phi(pos):
        #if phi(pos)<0, then this position is solid
        return -1
    def phiFixed(pos):
        #if phiFixed(pos)<0, then this position is fixed
        if iter==0:
            return 1
        else: return np.sum(pos)

    rho = torch.rand(tuple(res),dtype=DTYPE).cuda()
    phiTensor=torch.rand(tuple(res),dtype=DTYPE).cuda()
    for z in range(res[2]):
        for y in range(res[1]):
            for x in range(res[0]):
                pos=interpC(x,y,z,bb)
                phiTensor[x,y,z]=phi(pos)
                posl=pos-np.array([0,-0.4,0])
                posr=pos-np.array([0, 0.4,0])
                rho[x,y,z]=1 if np.linalg.norm(posl)<0.5 or np.linalg.norm(posr)<0.5 else 0.01
    phiFixedTensor=torch.rand(tuple([res[0]+1,res[1]+1,res[2]+1]),dtype=DTYPE).cuda()
    for z in range(res[2]+1):
        for y in range(res[1]+1):
            for x in range(res[0]+1):
                phiFixedTensor[x,y,z]=phiFixed(interpV(x,y,z,bb))
    f=torch.rand(tuple([3,res[0],res[1],res[2]]),dtype=DTYPE).cuda()

    TOCLayer.reset(phiTensor, phiFixedTensor, f, bb, 100, 100, 1e-8, output=False)
    return TOCWorseCase.compute_worst_case(rho), rho.detach().cpu().numpy()
    
if __name__=='__main__':
    torch.set_default_dtype(torch.float64)
    mg.initializeGPU()
    f, rho = debug(0)
    from Viewer import *
    showFMagnitudeCellVTK("fWorstCell",f)
    showFMagnitudeCellVTK("fWorstCellX",f,lambda vec:vec[0])
    showFMagnitudeCellVTK("fWorstCellY",f,lambda vec:vec[1])
    showFMagnitudeCellVTK("fWorstCellZ",f,lambda vec:vec[2])
    showRhoVTK("rho",rho)
    debug(1)
    mg.finalizeGPU()