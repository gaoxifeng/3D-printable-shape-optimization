from sympy import *
import numpy as np
from ShapeFunction2D import *

def getMatrix(r,c):
    return [Matrix([Symbol('U'+str(ic*2+ir)) for ir in range(r)]) for ic in range(c)]

def prepareU():
    Uss = getMatrix(2,4)
    xy = symbols(['x','y'])
    return Uss,xy

def energy():
    Uss,xy = prepareU()
    u = interp2D(Uss,xy)
    F = Matrix([[diff(u[i,0],xy[j]) for i in range(2)] for j in range(2)])
    eps = (F+F.T)/2
    
    k = Symbol('k')
    v = Symbol('v')
    #mu = k/2/(1+v)
    #lam = k*v/(1+v)/(1-2*v)
    mu = k/2/(1+v)
    lam = k*v/(1+v)/(1-v)
    return mu*trace(MatMul(eps.T,eps)) + lam/2*trace(eps)**2
    
def stiffness():
    Uss,xy = prepareU()
    uss = flatten(Uss)
    e = inte(energy())
    return Matrix([[diff(diff(e,uss[i]),uss[j]) for i in range(8)] for j in range(8)])

if __name__=='__main__':
    print(stiffness())