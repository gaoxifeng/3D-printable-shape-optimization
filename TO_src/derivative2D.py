from sympy import *
import numpy as np
vss = symbols(['a','b','c','d'])
xy = symbols(['x','y'])
def interp1D(vss,x):
    return vss[0]*(1-x)+vss[1]*x
def interp2D(vss,xy):
    return interp1D([interp1D(vss[:2],xy[0]),interp1D(vss[2:],xy[0])],xy[1])
vxy = interp2D(vss,xy)
def coeff(expr,v,vss):
    for i in vss:
        if i==v:
            expr = expr.subs(i,1)
        else: expr = expr.subs(i,0)
    return expr
N = [coeff(vxy,v,vss) for v in vss]
Nx = [diff(n,xy[0]) for n in N]
Ny = [diff(n,xy[1]) for n in N]

def NNe(i,j):
    a = N[i]*N[j]
    ax = integrate(a,xy[0]).subs(xy[0],1)-integrate(a,xy[0]).subs(xy[0],0)
    axy = integrate(ax,xy[1]).subs(xy[1],1)-integrate(ax,xy[1]).subs(xy[1],0)
    return axy

def difNNe(i,j):
    a = Nx[i]*Nx[j]
    ax = integrate(a,xy[0]).subs(xy[0],1)-integrate(a,xy[0]).subs(xy[0],0)
    axy1 = integrate(ax,xy[1]).subs(xy[1],1)-integrate(ax,xy[1]).subs(xy[1],0)
    
    a = Ny[i]*Ny[j]
    ax = integrate(a,xy[0]).subs(xy[0],1)-integrate(a,xy[0]).subs(xy[0],0)
    axy2 = integrate(ax,xy[1]).subs(xy[1],1)-integrate(ax,xy[1]).subs(xy[1],0)
    return axy1 + axy2

def NN():
    NNMat = eye(4)
    for i in range(4):
        for j in range(4):
            NNMat[i,j]=NNe(i,j)
    return NNMat

def difNN():
    difNNMat = eye(4)
    for i in range(4):
        for j in range(4):
            difNNMat[i,j]=difNNe(i,j)
    return difNNMat

print(NN())
print(difNN())