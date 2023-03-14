import torch

def dim(t):
    return len(t.shape)

def shape3D(t):
    if dim(t)==2:
        return [t.shape[0],t.shape[1],0]
    else: return t.shape
    
def to3DScalar(t):
    if dim(t)==2:
        return t.unsqueeze(2)
    else: return t
    
def to3DNodeScalar(t):
    if dim(t)==2:
        x, y = t.shape
        return t.unsqueeze(2).expand(x,y,2)
    else: return t
    
def to3DNodeVector(t):
    if dim(t)==3:
        d, x, y = t.shape
        if d == 2:
            t = torch.nn.functional.pad(t, (0,0,0,0,0,1), "constant", 0)
        return t.unsqueeze(3).expand(3, x, y, 2)
    else: return t
    
def makeSameDimScalar(b, dim):
    if dim==2:
        return b[:,:,0]
    else: return b
    
def makeSameDimVector(b, dim):
    if dim==2:
        return b[:2,:,:,0]
    else: return b