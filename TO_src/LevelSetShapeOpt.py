from LevelSetTopoOpt import *

class LevelSetShapeOpt():
    def __init__(self, *, h=.5, dt=.5, tau=0, maxloop=200, maxloopLinear=1000, tolx=1e-3, tolLinear=1e-2, outputInterval=1, outputDetail=False):
        # self.device = 'cpu'
        self.h = h
        self.dt = dt
        self.tau = tau
        self.maxloop = maxloop
        self.maxloopLinear = maxloopLinear
        self.tolx = tolx
        self.tolLinear = tolLinear
        self.outputInterval = outputInterval
        self.outputDetail = outputDetail

    def run(self, phiTensor, phiFixedTensor, f, lam, mu, phi=None, curvatureOnly=False):
        nelx, nely, nelz = shape3D(phiTensor)
        nelz = max(nelz,1)
        bb = mg.BBox()
        bb.minC = [0,0,0]
        bb.maxC = shape3D(phiTensor)
        if phi is not None:
            assert phi.shape == phiFixedTensor.shape
        else: phi = torch.ones(phiFixedTensor.shape).cuda()
        
        print("Level-set shape optimization problem")
        print(f"Number of degrees:{str(nelx)} x {str(nely)} x {str(nelz)} = {str(nelx * nely * nelz)}")
        # max and min stiffness
        E_min = torch.tensor(1e-3)
        E_max = torch.tensor(1.0)
        nvol = self.maxloop//2
        change = self.tolx*2
        loop = 0
        g=0
        
        #initialize torch layer
        mg.initializeGPU()
        TOLayer.reset(phiTensor, phiFixedTensor, f, bb, lam, mu, self.tolLinear, self.maxloopLinear, self.outputDetail)
        TOLayer.setupCurvatureFlow(self.dt, self.tau * nelx * nely * nelz)
        phi = TOLayer.reinitializeNode(phi)
        volTarget = torch.sum(LevelSetShapeOpt.computeDensity(phi, self.h)).item()
        print(f"Volume target: {volTarget}")
        while change > self.tolx and loop < self.maxloop:
            start = time.time()
            loop += 1
                
            phi = phi.detach()
            phi.requires_grad_()
            vol = torch.sum(LevelSetShapeOpt.computeDensity(phi, self.h))
            vol.backward()
            gradVolume = phi.grad.detach()
                
            #FE-analysis, calculate sensitivity
            if curvatureOnly:
                gradObj = -torch.ones_like(phi)
                obj = 0
            else:
                phi = phi.detach()
                phi.requires_grad_()
                rho = LevelSetShapeOpt.computeDensity(phi, self.h)
                obj = TOLayer.apply(rho * (E_max - E_min) + E_min)
                obj.backward()
                gradObj = -phi.grad.detach()
            
            #compute Lagrangian multiplier
            gradObj, vol = LevelSetShapeOpt.find_lam(phi, gradObj, gradVolume, volTarget, self.h, self.dt)
            
            #update level set function / reinitialize
            phi_old = phi.clone()
            phi = TOLayer.implicitCurvatureFlow(gradObj + phi / self.dt)
            phi = TOLayer.reinitializeNode(phi)
            change = torch.linalg.norm(phi.reshape(-1,1) - phi_old.reshape(-1,1), ord=float('inf')).item()
            end = time.time()
            if loop%self.outputInterval == 0:
                if not os.path.exists("results"):
                    os.mkdir("results")
                print("it.: {0}, obj.: {1:.3f}, vol.: {2:.3f}, ch.: {3:.3f}, time: {4:.3f}, mem: {5:.3f}Gb".format(loop, obj, vol, change, end - start, torch.cuda.memory_allocated(None)/1024/1024/1024))
                showRhoVTK("results/phi"+str(loop), to3DNodeScalar(phi).detach().cpu().numpy(), False)
                
        mg.finalizeGPU()
        return to3DNodeScalar(phi_old).detach().cpu().numpy()
    
    def computeDensity(phi, h):
        phic = torch.clamp(phi, min=-h, max=h) / h
        Hphic = phic**3 * .25 - phic * .75 + 0.5
        return LevelSetTopoOpt.nodeToCell(Hphic)
    
    def find_lam(phi0, gradObj0, gradVolume0, volTarget, h, dt):
        l1 = -1e9
        l2 = 1e9
        vol = 0.
        while (l2 - l1) > 1e-3 and abs(vol - volTarget) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            gradObj = gradObj0 + lmid * gradVolume0
            gradObj *= min(1, 1/torch.max(torch.abs(gradObj)))
            rho = LevelSetShapeOpt.computeDensity(phi0 + gradObj * dt, h)
            vol = torch.sum(rho).item()
            if vol < volTarget:
                l1 = lmid
            else:
                l2 = lmid
        return (gradObj.detach(), vol)
    
    def initialize_phi(phiFixedTensor, blockRes, f=None, evenOdd=1, scale=.55):
        phi = -torch.ones_like(phiFixedTensor).cuda()
        
        #determine resolution
        res = list(phi.shape)
        blockSize = min([(s-1)//blockRes for s in res])
        blockRes = [s//blockSize for s in res]
        blockSize = [float(s)/r for s,r in zip(res,blockRes)]
        if len(res)==2:
            res+=[1]
            blockRes+=[0]
            blockSize+=[2]
            
        #drill holes
        for x in range(blockRes[0]+1):
            for y in range(blockRes[1]+1):
                for z in range(blockRes[2]+1):
                    #only drill hole add even places
                    if (x+y+z)%2==evenOdd:
                        continue
                    #determine the parameter of the hole
                    ctr = [c*bs for c,bs in zip([x,y,z],blockSize)]
                    size = [bs*scale for bs in blockSize]
                    left = [int(max(c-s,0)) for c,s in zip(ctr,size)]
                    right = [int(min(c+s+1,r)) for c,s,r in zip(ctr,size,res)]
                    #do not drill hole when there is an external force
                    hasForce = False
                    if f is not None:
                        for xx in range(left[0],right[0]):
                            for yy in range(left[1],right[1]):
                                for zz in range(left[2],right[2]):
                                    hasForce = hasForce or torch.norm(to3DNodeVector(f)[:,xx,yy,zz])!=0.
                    if hasForce:
                        continue
                    #drill a hole
                    for xx in range(left[0],right[0]):
                        for yy in range(left[1],right[1]):
                            for zz in range(left[2],right[2]):
                                if sum([(p-c)**2/s**2 for p,c,s in zip([xx,yy,zz],ctr,size)])<1.:
                                    to3DNodeScalar(phi)[xx,yy,zz]=1.
                                
        return phi