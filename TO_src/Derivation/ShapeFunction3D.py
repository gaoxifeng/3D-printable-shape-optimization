from sympy import *
import numpy as np
def interp1D(vss,x):
    return vss[0]*(1-x)+vss[1]*x
def interp2D(vss,xy):
    return interp1D([interp1D(vss[:2],xy[0]),interp1D(vss[2:],xy[0])],xy[1])
def interp3D(vss,xyz):
    return interp1D([interp2D(vss[:4],xyz[:2]),interp2D(vss[4:],xyz[:2])],xyz[2])
def coeff(expr,v,vss):
    for i in vss:
        if i==v:
            expr = expr.subs(i,1)
        else: expr = expr.subs(i,0)
    return expr

def prepare():
    vss = symbols(['a','b','c','d',
                   'e','f','g','h'])
    xyz = symbols(['x','y','z'])
    vxyz = interp3D(vss,xyz)
    N = [coeff(vxyz,v,vss) for v in vss]
    Nx = [diff(n,xyz[0]) for n in N]
    Ny = [diff(n,xyz[1]) for n in N]
    Nz = [diff(n,xyz[1]) for n in N]
    return vss,xyz,N,Nx,Ny,Nz

def inte(a):
    vss,xyz,N,Nx,Ny,Nz = prepare()
    ax = integrate(a,xyz[0]).subs(xyz[0],1)-integrate(a,xyz[0]).subs(xyz[0],0)
    axy = integrate(ax,xyz[1]).subs(xyz[1],1)-integrate(ax,xyz[1]).subs(xyz[1],0)
    axyz = integrate(axy,xyz[2]).subs(xyz[2],1)-integrate(axy,xyz[2]).subs(xyz[2],0)
    return axyz

def difNNe(i,j):
    vss,xyz,N,Nx,Ny,Nz = prepare()
    return inte(Nx[i]*Nx[j])+inte(Ny[i]*Ny[j])+inte(Nz[i]*Nz[j])

def NN():
    vss,xyz,N,Nx,Ny,Nz = prepare()
    NNMat = eye(8)
    for i in range(8):
        for j in range(8):
            NNMat[i,j]=inte(N[i]*N[j])
    return NNMat

def difNN():
    vss,xyz,N,Nx,Ny,Nz = prepare()
    difNNMat = eye(8)
    for i in range(8):
        for j in range(8):
            difNNMat[i,j]=difNNe(i,j)
    return difNNMat

if __name__=='__main__':
    print(NN())
    print(difNN())