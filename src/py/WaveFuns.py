import numpy as np

def GaussianWave (bcs, center, radius, k = None, intvec = None):
    """
    Generate a Gaussian wave packet.

    Parameters:
    bcs: list of ints, dimensionality of the lattice.
    center: tuple of ints, position of the center of the packet.
    radius: radius of the wave packet.
    k: Defualt [0]*dim. If provided, should be a tuple of numbers, giving the wave vector k.
    intvec: Default [1]. If provided, will tensor product the spatial wave function with the internal wave function.
    """
    dim = len(bcs)
    if k is None:
        k = [0]*dim
    if intvec is None:
        intvec = [1]
    coords = np.meshgrid(*[[i for i in range(abs(tlen))] for tlen in bcs], indexing="ij")
    distmat = np.zeros((np.size(coords[0]),), dtype=complex)
    xs = []
    for i in range(dim):
        if bcs[i] > 0:
            xs.append(coords[i].flatten()-center[i])
        else:
            xs.append(np.fmod(coords[i].flatten()-center[i], abs(bcs[i])))
    try:
        if radius == np.inf:
            for i in range(dim):
                distmat += 1j*xs[i]*k[i]
        else:
            raise Exception()
    except:
        try:
            radius[0]
        except:
            radius = [radius]*dim
        for i in range(dim):
            if radius[i] == 0: # Zero radius is a delta function
                distmat[np.array(xs[i])!=0] += -np.inf
            else:
                if bcs[i] > 0:
                    distmat += -xs[i]**2/(2*radius[i]**2) + 1j*xs[i]*k[i]
                else:
                    distmat += -xs[i]**2/(2*radius[i]**2) + 1j*xs[i]*k[i]
    exponvec = np.exp(distmat)
    totvec = np.kron(np.reshape(exponvec, [np.size(exponvec), 1]), np.reshape(intvec, [np.size(intvec), 1]))
    return totvec / np.linalg.norm(totvec)