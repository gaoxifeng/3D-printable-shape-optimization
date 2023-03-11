from sympy import *
import numpy as np
vss = symbols(['a','b','c','d',
               'e','f','g','h'])
xyz = symbols(['x','y','z'])
def interp1D(vss,x):
    return vss[0]*(1-x)+vss[1]*x
def interp2D(vss,xy):
    return interp1D([interp1D(vss[:2],xy[0]),interp1D(vss[2:],xy[0])],xy[1])
def interp3D(vss,xyz):
    return interp1D([interp2D(vss[:4],xyz[:2]),interp2D(vss[4:],xyz[:2])],xyz[2])
vxyz = interp3D(vss,xyz)
def coeff(expr,v,vss):
    for i in vss:
        if i==v:
            expr = expr.subs(i,1)
        else: expr = expr.subs(i,0)
    return expr
N = [coeff(vxyz,v,vss) for v in vss]
Nx = [diff(n,xyz[0]) for n in N]
Ny = [diff(n,xyz[1]) for n in N]
Nz = [diff(n,xyz[1]) for n in N]

def NNe(a,i,j):
    ax = integrate(a,xyz[0]).subs(xyz[0],1)-integrate(a,xyz[0]).subs(xyz[0],0)
    axy = integrate(ax,xyz[1]).subs(xyz[1],1)-integrate(ax,xyz[1]).subs(xyz[1],0)
    axyz = integrate(axy,xyz[2]).subs(xyz[2],1)-integrate(axy,xyz[2]).subs(xyz[2],0)
    return axyz

def difNNe(i,j):
    return NNe(Nx[i]*Nx[j],i,j)+NNe(Ny[i]*Ny[j],i,j)+NNe(Nz[i]*Nz[j],i,j)

def NN():
    NNMat = eye(8)
    for i in range(8):
        for j in range(8):
            NNMat[i,j]=NNe(N[i]*N[j],i,j)
    return NNMat

def difNN():
    difNNMat = eye(8)
    for i in range(8):
        for j in range(8):
            difNNMat[i,j]=difNNe(i,j)
    return difNNMat

print(NN())
print(difNN())