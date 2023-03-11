from sympy import *
import numpy as np
from ShapeFunction3D import *

def getMatrix(r,c):
    return [Matrix([Symbol('U'+str(ic*2+ir)) for ir in range(r)]) for ic in range(c)]

def prepareU():
    Uss = getMatrix(3,8)
    xyz = symbols(['x','y','z'])
    return Uss,xyz

def energy():
    Uss,xyz = prepareU()
    u = interp3D(Uss,xyz)
    F = Matrix([[diff(u[i,0],xyz[j]) for i in range(3)] for j in range(3)])
    eps = (F+F.T)/2
    
    k = Symbol('k')
    v = Symbol('v')
    mu = k/2/(1+v)
    lam = k*v/(1+v)/(1-2*v)
    return mu*trace(MatMul(eps.T,eps)) + lam/2*trace(eps)**2
    
def stiffness():
    Uss,xy = prepareU()
    uss = flatten(Uss)
    e = inte(energy())
    return Matrix([[diff(diff(e,uss[i]),uss[j]) for i in range(24)] for j in range(24)])

if __name__=='__main__':
    print(stiffness())