from LevelSetTopoOpt import *

class LevelSetShapeOpt():
    def __init__(self, *, h=0.5, dt=0.5, tau=1e-4, maxloop=200, maxloopLinear=1000, tolx=1e-3, tolLinear=1e-2, outputInterval=1, outputDetail=False):
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
        volTarget = torch.sum(LevelSetShapeOpt.computeDensity(phi, self.h)).item()
        
        print("Level-set shape optimization problem")
        print(f"Number of degrees:{str(nelx)} x {str(nely)} x {str(nelz)} = {str(nelx * nely * nelz)}")
        print(f"Volume target: {volTarget}")
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
        phi = TOLayer.reinitializeCell(phi[:-1,:-1])
        if not os.path.exists("results"):
            os.mkdir("results")
        showRhoVTK("results/phiInit", to3DScalar(phi).detach().cpu().numpy(), False)
        while change > self.tolx and loop < self.maxloop:
            start = time.time()
            loop += 1
            
            #compute volume gradient
            phi = phi.detach()
            phi.requires_grad_()
            rho = LevelSetShapeOpt.computeDensity(phi, self.h)
            vol = torch.sum(rho)
            vol.backward()
            gradVolume = phi.grad.detach()
            
            #FE-analysis, calculate sensitivity
            if curvatureOnly:
                gradObj = -gradVolume
                obj = 0
            else:
                phi = phi.detach()
                phi.requires_grad_()
                rho = LevelSetShapeOpt.computeDensity(phi, self.h)
                obj = TOLayer.apply(rho * (E_max - E_min) + E_min)
                obj.backward()
                gradObj = -phi.grad.detach()
            
            #compute Lagrangian multiplier
            gradObj *= min(1, 1/torch.max(torch.abs(gradObj)))
            lam, phi = LevelSetShapeOpt.find_lam(phi, gradObj, gradVolume, volTarget, self.h, self.dt)
            
            #update level set function / reinitialize
            phi_old = phi.clone()
            C = gradObj.reshape(-1).shape[0] / torch.sum(torch.abs(gradObj)).item()
            phi = TOLayer.implicitCurvatureFlow(phi / self.dt)
            phi = TOLayer.reinitializeNode(phi)
            change = torch.linalg.norm(phi.reshape(-1,1) - phi_old.reshape(-1,1), ord=float('inf')).item()
            end = time.time()
            if loop%self.outputInterval == 0:
                print("it.: {0}, obj.: {1:.3f}, vol.: {2:.3f}, ch.: {3:.3f}, time: {4:.3f}, mem: {5:.3f}Gb".format(loop, obj, vol, change, end - start, torch.cuda.memory_allocated(None)/1024/1024/1024))
                showRhoVTK("results/phi"+str(loop), to3DScalar(phi).detach().cpu().numpy(), False)
                
        mg.finalizeGPU()
        return to3DScalar(phi_old).detach().cpu().numpy()
    
    def computeDensity(phi, h):
        phic = torch.clamp(phi, min=-h, max=h) / h
        Hphic = phic**3 * .25 - phic * .75 + 0.5
        return LevelSetTopoOpt.nodeToCell(Hphic)
    
    def find_lam(phi0, gradObj, gradVolume, volTarget, h, dt):
        l1 = -1e9
        l2 = 1e9
        vol = 0.
        while (l2 - l1) > 1e-3 and abs(vol - volTarget) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            phi = phi0 + (gradObj - lmid) * dt
            rho = LevelSetShapeOpt.computeDensity(phi, h)
            vol = torch.sum(rho).item()
            if vol < volTarget:
                l1 = lmid
            else:
                l2 = lmid
        return (lmid, phi)