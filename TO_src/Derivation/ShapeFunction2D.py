from sympy import *
import numpy as np
def interp1D(vss,x):
    return vss[0]*(1-x)+vss[1]*x
def interp2D(vss,xy):
    return interp1D([interp1D(vss[:2],xy[0]),interp1D(vss[2:],xy[0])],xy[1])
def coeff(expr,v,vss):
    for i in vss:
        if i==v:
            expr = expr.subs(i,1)
        else: expr = expr.subs(i,0)
    return expr

def prepare():
    vss = symbols(['a','b','c','d'])
    xy = symbols(['x','y'])
    vxy = interp2D(vss,xy)
    N = [coeff(vxy,v,vss) for v in vss]
    Nx = [diff(n,xy[0]) for n in N]
    Ny = [diff(n,xy[1]) for n in N]
    return vss,xy,vxy,N,Nx,Ny

def inte(a):
    vss,xy,vxy,N,Nx,Ny = prepare()
    ax = integrate(a,xy[0]).subs(xy[0],1)-integrate(a,xy[0]).subs(xy[0],0)
    axy = integrate(ax,xy[1]).subs(xy[1],1)-integrate(ax,xy[1]).subs(xy[1],0)
    return axy

def difNNe(i,j):
    vss,xy,vxy,N,Nx,Ny = prepare()
    return inte(Nx[i]*Nx[j]) + inte(Ny[i]*Ny[j])

def NN():
    vss,xy,vxy,N,Nx,Ny = prepare()
    NNMat = eye(4)
    for i in range(4):
        for j in range(4):
            NNMat[i,j]=inte(N[i]*N[j])
    return NNMat

def difNN():
    vss,xy,vxy,N,Nx,Ny = prepare()
    difNNMat = eye(4)
    for i in range(4):
        for j in range(4):
            difNNMat[i,j]=difNNe(i,j)
    return difNNMat

if __name__=='__main__':
    print(NN())
    print(difNN())