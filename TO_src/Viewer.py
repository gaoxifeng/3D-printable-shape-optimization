import numpy as np

def showRho(rho, iso=0.5, addLayer=True, smooth=False):
    #add a layer of zero
    if addLayer:
        rhoLayered = np.zeros((rho.shape[0]+2,rho.shape[1]+2,rho.shape[2]+2))
        rhoLayered[1:rho.shape[0]+1,1:rho.shape[1]+1,1:rho.shape[2]+1] = rho
        rho = rhoLayered
    #show
    import mcubes,trimesh
    if smooth:
        rho = mcubes.smooth(rho)
    vertices, triangles = mcubes.marching_cubes(rho, iso)
    mesh = trimesh.Trimesh(vertices, triangles)
    mesh.show()
    
def showRhoVTK(name, rho, addLayer=True):
    from pyevtk.hl import gridToVTK
    #add a layer of zero
    if addLayer:
        rhoLayered = np.zeros((rho.shape[0]+2,rho.shape[1]+2,rho.shape[2]+2))
        rhoLayered[1:rho.shape[0]+1,1:rho.shape[1]+1,1:rho.shape[2]+1] = rho
        rho = rhoLayered
        
    nx, ny, nz = rho.shape
    x = np.zeros((nx + 1, ny + 1, nz + 1))
    y = np.zeros((nx + 1, ny + 1, nz + 1))
    z = np.zeros((nx + 1, ny + 1, nz + 1))
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                x[i,j,k] = i
                y[i,j,k] = j
                z[i,j,k] = k
    gridToVTK(name,x,y,z,cellData={"magnitude":rho})
    
def showFMagnitudeVTK(name, f, op=np.linalg.norm):
    from pyevtk.hl import gridToVTK
    nx, ny, nz = (f.shape[1]-1, f.shape[2]-1, f.shape[3]-1)
    x = np.zeros((nx + 1, ny + 1, nz + 1))
    y = np.zeros((nx + 1, ny + 1, nz + 1))
    z = np.zeros((nx + 1, ny + 1, nz + 1))
    a = np.zeros((nx + 1, ny + 1, nz + 1))
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                x[i,j,k] = i
                y[i,j,k] = j
                z[i,j,k] = k
                a[i,j,k] = op(f[:,i,j,k])
    gridToVTK(name,x,y,z,pointData={"magnitude":a})
          

def showFMagnitudeCellVTK(name, f, op=np.linalg.norm):
    from pyevtk.hl import gridToVTK
    nx, ny, nz = (f.shape[1]-1, f.shape[2]-1, f.shape[3]-1)
    x = np.zeros((nx + 1, ny + 1, nz + 1))
    y = np.zeros((nx + 1, ny + 1, nz + 1))
    z = np.zeros((nx + 1, ny + 1, nz + 1))
    a = np.zeros((nx, ny, nz))
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                x[i,j,k] = i
                y[i,j,k] = j
                z[i,j,k] = k
                if k<nz and j<ny and i<nx:
                    a[i,j,k] = op(f[:,i,j,k])
    gridToVTK(name,x,y,z,cellData={"magnitude":a})
          