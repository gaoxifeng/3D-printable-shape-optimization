from TopoOpt import *
from TOCLayer import TOCLayer
from TOCWorstCase import TOCWorseCase

class TopoOptWorstCase(TopoOpt):
    def __init__(self, *, p=3, rmin=5, maxloop=200, maxloopLinear=1000, tolx=1e-3, tolLinear=1e-2, outputInterval=1, outputDetail=False):
        TopoOpt.__init__(self, p=p, rmin=rmin, maxloop=maxloop, maxloopLinear=maxloopLinear, tolx=tolx, tolLinear=tolLinear, outputInterval=outputInterval, outputDetail=outputDetail)

    def run(self, rho, phiTensor, phiFixedTensor, rhoMask, lam, mu):
        nelx, nely, nelz = shape3D(rho)
        nelz = max(nelz,1)
        bb = mg.BBox()
        bb.minC = [0,0,0]
        bb.maxC = shape3D(rho)
        Ker, Ker_S = TopoOpt.filter(self.rmin, rho)
        volfrac = torch.sum(rho).item() / rho.reshape(-1).shape[0]

        print("Worst case minimum complicance problem with OC")
        print(f"Number of degrees:{str(nelx)} x {str(nely)} x {str(nelz)} = {str(nelx * nely * nelz)}")
        print(f"Volume fration: {volfrac}, Penalty p: {self.p}, Filter radius: {self.rmin}")
        # max and min stiffness
        E_min = torch.tensor(1e-3)
        E_max = torch.tensor(1.0)
        change = self.tolx*2
        loop = 0
        g=0

        #compute filtered volume gradient (this is contant so we can precompute)
        rho = rho.detach()
        rho.requires_grad_()
        rho_filtered = TopoOpt.filter_density(Ker, rho)/Ker_S
        volume = torch.sum(rho_filtered)
        volume.backward()
        gradVolume = rho.grad.detach()
            
        #initialize torch layer
        mg.initializeGPU()
        if len(rho.shape)==2:
            f = torch.zeros((2, nelx, nely))
        else: f = torch.zeros((3, nelx, nely, nelz))
        TOCLayer.reset(phiTensor, phiFixedTensor, f, bb, lam, mu, self.tolLinear, self.maxloopLinear, self.outputDetail)
        while change > self.tolx and loop < self.maxloop:
            start = time.time()
            loop += 1

            #update worst case
            TOCLayer.free()
            rho_filtered = (TopoOpt.filter_density(Ker, rho)/Ker_S).detach()
            TOCWorseCase.compute_worst_case(E_min + rho_filtered ** self.p * (E_max - E_min))
            TOCLayer.fix()
        
            #compute filtered objective gradient
            rho = rho.detach()
            rho.requires_grad_()
            rho_filtered = TopoOpt.filter_density(Ker, rho)/Ker_S
            obj = TOCLayer.apply(E_min + rho_filtered ** self.p * (E_max - E_min))
            obj.backward()
            gradObj = (rho.grad / torch.max(torch.abs(rho.grad))).detach()
            
            rho_old = rho.clone()
            rho, g = TopoOptWorstCase.oc_grid(rho, gradObj, gradVolume, g, rhoMask)
            change = torch.linalg.norm(rho.reshape(-1,1) - rho_old.reshape(-1,1), ord=float('inf')).item()
            end = time.time()
            if loop%self.outputInterval == 0:
                print("it.: {0}, obj.: {1:.3f}, vol.: {2:.3f}, ch.: {3:.3f}, time: {4:.3f}, mem: {4:.3f}Gb".format(loop, obj, (g + volfrac * nelx * nely * nelz) / (nelx * nely * nelz), change, end - start, torch.cuda.memory_allocated(None)/1024/1024/1024))
                showRhoVTK("rho"+str(loop), to3DScalar(rho).detach().cpu().numpy(), False)
        
        mg.finalizeGPU()
        return to3DScalar(rho_old).detach().cpu().numpy()

    def oc_grid(x, gradObj, gradVolume, g, rhoMask):
        l1 = -1e9
        l2 = 1e9
        move = 0.2
        eta = 0.5
        # reshape to perform vector operations
        while (l2 - l1) > 1e-6:
            lmid = 0.5 * (l2 + l1)
            #Be_eta = (torch.maximum( torch.tensor(0.0), torch.div(-gradObj, gradVolume) / lmid ) ** eta).detach()
            #xnew = (torch.maximum(torch.tensor(0.0), torch.maximum(x - move, torch.minimum(torch.tensor(1.0), torch.minimum(x + move, x * Be_eta))))).detach()
            xnew = (torch.maximum(torch.tensor(0.0), torch.maximum(x - move, torch.minimum(torch.tensor(1.0), torch.minimum(x + move, x - gradObj - lmid * gradVolume))))).detach()
            rhoMask(xnew)
            gt = g + torch.sum((gradVolume * (xnew - x))).item()
            if gt > 0:
                l1 = lmid
            else:
                l2 = lmid
        return (xnew, gt)