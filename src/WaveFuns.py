import numpy as np

def GaussianWave (bcs, center, radius, k = None, intvec = None):
    dim = len(bcs)
    if k is None:
        k = [0]*dim
    if intvec is None:
        intvec = [1]
    coords = np.meshgrid(*[[i for i in range(tlen)] for tlen in bcs], indexing="ij")
    distmat = 0
    try:
        if radius == np.Infinity:
            for i in range(dim):
                distmat += 1j*(coords[i].flatten()-center[i])*k[i]
        else:
            raise Exception()
    except:
        try:
            radius[0]
        except:
            radius = [radius]*dim
        for i in range(dim):
            distmat += -(coords[i].flatten()-center[i])**2/(2*radius[i]**2) + 1j*(coords[i].flatten()-center[i])*k[i]
    exponvec = np.exp(distmat)
    totvec = np.kron(np.reshape(exponvec, [np.size(exponvec), 1]), np.reshape(intvec, [np.size(intvec), 1]))
    return totvec / np.linalg.norm(totvec)