import scipy.interpolate
import scipy.optimize
from .HamClass import HamModel
from .EvoDat import EvoDat1D, EvoDat2D
from .WaveFuns import GaussianWave
from .Config import *

import numpy as np
import scipy
from scipy.ndimage import gaussian_filter as gaussfilt
from matplotlib import pyplot as plt
from math import floor, ceil

import pickle

def plot_pointG_1D_tslope (model:HamModel, L, T, dT=0.1, force_evolve = False, ip = None, iprad = 1, ik = 0, start_t=5, addn="", precision = 0, plot_ax = None):
    """
    For a 1D chain, do time evolution to extract G(x,x,t) at a point x.
    Try to compare |G(x,x,t)| to the theoretical prediction ~ t^{-a}exp(-i E_s t), where a is 1/2 in the bulk and 3/2 on the boundary.
    Specifically, generates a plot of log|G(x,x,t)|-Im(E_s)t vs log(t) [which should be linear], and compares it to the theoretical prediction.

    Parameters:
    model: HamModel
        The Hamiltonian model to be used.
    L: int
        The size of the 1D chain.
    T: float
        The time to evolve to.
    dT: float
        The time step to use in the evolution.
    force_evolve: bool
        Whether to force the evolution to be redone (i.e. not to read from existing files).
    ip: int
        The point x at which to extract G(x,x,t). If None, the middle of the chain is used.
    iprad: int
        The radius of the initial Gaussian wave packet to use.
    ik: float
        The wave vector of the initial Gaussian wave packet to use.
    start_t: float
        The time at which to start the plot.
    addn: str
        Additional name to add to the file name.
    precision: int
        The precision to use in Mathematica evolution. If given, uses Mathematica to do time evolution. If not given or zero, uses RK45 in python.
    plot_ax: plt.Axes
        If given, the plot will be drawn on this axis. If not given, a new figure will be created.
    """

    Ham = model([L])
    Hamname = model.name+(("_"+addn) if addn!="" else "")
    FName = "{}_FG_L{}_T{}".format(Hamname, L, T)
    if ip is None:
        ip = int(L/2)
    else:
        FName += "_ip{}".format(ip)
        if ip < 0:
            ip = L+ip

    which_boundary = 0
    decay_exp = 1/2
    if ip < L/4:
        which_boundary = -1
        decay_exp = 3/2
    elif ip > 3*L/4:
        which_boundary = 1
        decay_exp = 3/2

    if model.int_dim == 1:
        intvec = [1]
    else:
        intvec = np.random.randn(model.int_dim) + 1j*np.random.randn(model.int_dim)
        intvec = intvec / np.linalg.norm(intvec)

    init = GaussianWave([L], (ip,), iprad, k = (ik,), intvec = intvec)

    try:
        if force_evolve:
            raise Exception()
        evodat = EvoDat1D.read(FName)
    except Exception:
        evodat = EvoDat1D.from_evolve(Ham, T, dT, init, FName, precision=precision, return_idnorm=False)
        evodat.save()

    if plot_ax is None:
        evodat.plot_profile()

    dats = evodat.res[ip,0,:]
    amps = evodat.norms + np.log(np.abs(dats))

    init = evodat.res[:,0,0]

    spf = model.SPFMMA()

    for row in spf:
        # Find the relevant saddle point
        if row[3]:
            z = row[0]
            E = row[1]
            Hpp = row[4]

            if which_boundary == -1:
                xmin = max(0, ip-10*iprad)
                xmax = min(L-1, ip+10*iprad)
                zl = model.GFMMA(E, *[(x+1,ip+1) for x in range(xmin, xmax+1)])
                tamp = np.sum(zl * (init[xmin:xmax+1]))
            elif which_boundary == 1:
                xmin = max(0, ip-10*iprad)
                xmax = min(L-1, ip+10*iprad)
                zl = model.GFMMA(E, *[(x-L,ip-L) for x in range(xmin, xmax+1)])
                tamp = np.sum(zl * (init[xmin:xmax+1]))
            else:
                zl = z ** (ip - np.arange(L))
                tamp = np.sum(zl*init)

            if which_boundary == 0:
                thissp = np.log(tamp/z*np.sqrt(1/(2*np.pi))*Hpp/1j)
            else:
                thissp = np.log(tamp/z*np.sqrt(1/(8*np.pi))*Hpp)

            break
    
    # Deduct the exponential growth rate profile
    amps = amps - np.imag(E)*(evodat.times - evodat.times[0])
    sti = int(start_t / (evodat.times[1]-evodat.times[0]))
    times = evodat.times[sti:]
    amps = amps[sti:]

    if plot_ax is None:
        plt.rcParams.update(**PLT_PARAMS)
        plt.figure(figsize=SINGLE_FIGSIZE, layout="constrained")
        ax = plt.gca()
    else:
        ax = plot_ax

    ax.plot(times, amps, label="Numerical", color="C0")
    ax.plot(times, np.real(thissp)-decay_exp*np.log(times), label="Theory", color="C1", linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$log|G(t)|-\mathrm{Im}E_s t$")
    ax.legend()

    if plot_ax is None:
        plt.savefig(FName+"_texp.pdf")
        plt.close()

########## UPDATED METHOD ##############
def plot_pointG_1D_WF (model:HamModel, L, T, which_edge, depth, dT=0.1, force_evolve = False, ip = 0, iprad = 1, ik = 0, addn="", plot_ax = None, takets = [], precision = 0, takerange = 20, fit = False, potential = None):

    Ham = model([L])
    Hamname = model.name+(("_"+addn) if addn!="" else "")
    FName = "{}_WF_{}{}i{}_L{}T{}".format(Hamname, 'L' if which_edge<=0 else 'R', depth, ip, L, T)
    
    if which_edge > 0:
        ip = L-1-ip
        left = L-depth
        right = L
    else:
        left = 0
        right = depth

    if model.int_dim == 1:
        intvec = [1]
    else:
        intvec = np.random.randn(model.int_dim) + 1j*np.random.randn(model.int_dim)
        intvec = intvec / np.linalg.norm(intvec)

    if potential is None:
        potential = [0] * L*model.int_dim
    potential = np.diag(potential)

    init = GaussianWave([L], (ip,), iprad, k = (ik,), intvec = intvec)

    try:
        if force_evolve:
            raise Exception()
        evodat = EvoDat1D.read(FName)
    except Exception:
        evodat = EvoDat1D.from_evolve(Ham.time_evolve, T, dT, init, FName, precision=precision, potential=potential, return_idnorm = True)
        evodat.save()

    if plot_ax is None:
        evodat.plot_profile()
        plt.rcParams.update(**PLT_PARAMS)
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=DOUBLE_FIGSIZE)
    else:
        ax1. ax2 = plot_ax

    ax1.set_xlabel("$x$")
    ax1.set_ylabel(r"$\log|psi|$")
    ax2.set_xlabel("$x$")
    ax2.set_ylabel(r"Phase$(\psi)$")

    if len(takets) == 0:
        takets = [T]
    
    itakets = [ceil(((t-evodat.times[0])/(evodat.times[1]-evodat.times[0])))-1 for t in takets]

    for i, taketi in enumerate(itakets):
        dat = evodat.res[left:right, 0, taketi]
        amp = np.log(np.abs(dat)) + evodat.idnorm[left:right, 0, taketi]
        phs = np.unwrap(np.angle(dat) - np.angle(evodat.res[ip,0,taketi]))
        xs = np.arange(left, right)
        ax1.plot(xs, amp, label = "t = {:.1f}".format(evodat.times[taketi]))
        ax2.plot(xs, phs, label = "t = {:.1f}".format(evodat.times[taketi]))
        ax1.set_title(f"Wave function at t = {takets[i]}")
        plt.savefig(FName+f"_t{takets[i]}_rng{left}:{right}wf.jpg")
        plt.close()

def plot_pointG_1D_vec (model:HamModel, L, T, dT=0.1, force_evolve= False, output=print, ip = None, iprad = 1, ik = 0, addn="", start_t = 5, error_log = False, precision = 0):

    Ham = model([L])
    B = Ham.int_dim
    if B == 1:
        print("plot_vec won't do anything for single-band Hamiltonians!")
        return
    
    Hamname = model.name+(("_"+addn) if addn!="" else "")

    amps = []
    phas = []

    FName0 = f"{Hamname}_FG_L{L}T{T}"
    if ip is None:
        ip = int(L/2)
    else:
        FName0 += f"ip{ip}"
        if ip < 0:
            ip = L+ip

    for i in range(B):
        FName = f"{FName0}_B{i}"

        intvec = [0]*B
        intvec[i] = 1

        init = GaussianWave([L], (ip,), iprad, k = (ik,), intvec = intvec)

        try:
            if force_evolve:
                raise Exception()
            evodat = EvoDat1D.read(FName)
        except Exception:
            evodat = EvoDat1D.from_evolve(Ham.time_evolve, T, dT, init, FName, precision=precision, return_idnorm=False)
            evodat.save()

        dats = evodat.res[ip,:,:]
        amp = np.log(np.abs(dats)) + evodat.norms
        pha = np.angle(dats)
        amps.append(amp)
        phas.append(pha)

    amps = np.transpose(np.stack(amps), [1,0,2])
    phas = np.transpose(np.stack(phas), [1,0,2])

    # Calculate G_{ij}(t) / G_{00}(t)
    amps = amps - amps[0,0,:]
    phas = phas - phas[0,0,:]
    Gratios = np.exp(amps+1j*phas)

    spf = model.SPFMMA()

    Gmat = None
    E = None
    found = False

    for row in spf:
        if row[3]:

            if found:
                if np.imag(E-row[1]) < 0.1:
                    print("****WARNING**** CLOSE EIGENVALUES")
                    print(f"E1 = {E}, E2 = {row[1]}")
                    break

            E = row[1]
            vR = row[5]
            vL = row[6]
            if len(vR) > 1 or len(vL) > 1:
                raise Exception("Eigenvalue is degenerate! This function currently does not support it.")
            vR = np.array(vR[0])
            vL = np.array(vL[0])
            Gmat = vL[np.newaxis,:] * vR[:,np.newaxis] / (vL@vR)

            found = True

    sti = int(start_t / (evodat.times[1]-evodat.times[0]))
    times = evodat.times[sti:]
    Gratios = Gratios[:,:,sti:]

    names = [
        [
        [rf"$\mathrm{{Re}}[G_\{{{i+1},{j+1}}}(t) / G_{{1,1}}(t)]$", rf"$\mathrm{{Im}}[G_\{{{i+1},{j+1}}}(t) / G_{{1,1}}(t)]$"]
         for j in range(B)
        ] for i in range(B)
        ]

    theonames = [
        [
        [rf"$\mathrm{{Re}}[v^L_i(z_s) v^R_j(z_s)]$", rf"$\mathrm{{Im}}[v^L_i(z_s) v^R_j(z_s)]$"]
         for j in range(B)
        ] for i in range(B)
        ]

    lw = 1

    plt.rcParams.update({"font.size":8, "lines.linewidth":lw, "mathtext.fontset":"cm"})

    # Generate the plots
    
    if not error_log:
        plt.figure(figsize=(3.375, 2.4), layout="constrained")
        ax1 = plt.gca()
    else:
        _, (ax1,ax2) = plt.subplots(1, 2, figsize=(6, 2.4), layout="constrained")

    # Plot real part and imag part

    num = 0

    for i in range(B):
        for j in range(B):
            if i == 0 and j == 0:
                continue

            ax1.plot(times, np.real(Gratios[i,j,:]), color=f"C{num}", label=names[i][j][0])
            ax1.plot(times, [np.real(Gmat[i,j]/Gmat[0,0])]*len(times), color=f"C{num}", linestyle="--", label=theonames[i][j][0])
            if error_log:
                ax2.plot(times, np.abs(np.real(Gratios[i,j,:]-Gmat[i,j]/Gmat[0,0])), color=f"C{num}", label=names[i][j][0])

            num += 1
            
            ax1.plot(times, np.imag(Gratios[i,j,:]), color=f"C{num}", label=names[i][j][1])
            ax1.plot(times, [np.imag(Gmat[i,j]/Gmat[0,0])]*len(times), color=f"C{num}", linestyle="--", label=theonames[i][j][1])
            if error_log:
                ax2.plot(times, np.abs(np.imag(Gratios[i,j,:]-Gmat[i,j]/Gmat[0,0])), color=f"C{num}", label=names[i][j][1])

            num += 1

    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$G_{{i,j}}/G_{{1,1}}$")
    ax1.set_title(r"$G_{{i,j}}/G_{{1,1}}$")
    ax1.legend()

    if error_log:
        ax2.set_xlabel(r"$t$")
        ax2.set_ylabel("Error")
        ax2.set_title("Numerical v.s. Theory Error")
        ax2.legend()

    plt.savefig(FName0+"_VEC.pdf")
    plt.close()

def plot_and_compare_2D_Edge (model2d:HamModel, L, W, T, Ns = 10, edge="x-", k = 0, ipdepth = 0, kspan = 0.1, dT=0.2, force_evolve = False, addname = ""):

    tkdT = 0.25

    FName = "{}_L{}W{}_{}{}_T{}_{}".format(model2d.name, L, W, edge, ipdepth, T, addname)
    if k != 0:
        FName += "_k{}".format(k)
    if edge == "x-":
        ip = (int(L/2), 0+ipdepth)
        ik = (k,0)
        para_edge = 0
        perp_edge = 0
    elif edge == "x+":
        ip = (int(L/2), W-1-ipdepth)
        ik = (k,0)
        para_edge = 0
        perp_edge = 1
    elif edge == "y-":
        ip = (0+ipdepth, int(W/2))
        ik = (0,k)
        para_edge = 1
        perp_edge = 0
    elif edge == "y+":
        ip = (L-1-ipdepth, int(W/2))
        ik = (0,k)
        para_edge = 1
        perp_edge = 1
    else:
        raise Exception("ip must be one of: 'x-', 'x+', 'y-', 'y+'.")
    
    x0, y0 = ip
    
    init = GaussianWave([L,W], ip, 1/(2*kspan), ik)
    
    try:
        if force_evolve:
            raise Exception("force_evolve is set to True.")
        evodat = EvoDat2D.read(FName)
        print("Read data")
    except Exception as e:
        print("Didn't read: {}".format(e))
        evodat = EvoDat2D.from_evolve_m(model2d, L, W, T, dT, init, FName, takedT=tkdT)
        evodat.save()
    
    res = evodat.getRes()
    seldat = np.linalg.norm(res, axis=2)
    if edge.startswith("x"):
        seldat = seldat[:,y0,:]
    else:
        seldat = seldat[x0,:,:]
    seldat = seldat**2
    seldat /= np.sum(seldat, axis=0)

    # Construct the projected wave function in one dimension
    if para_edge == 0:
        ip_perp = y0
        L1d = L
    else:
        ip_perp = x0
        L1d = W
    # For each k, we treat this as a 1D system and get an effective amplitude
    # This gives us the projection of our wave function onto the edge mode.
    xs = np.arange(L1d)
    ks = xs*(2*np.pi)/L1d

    # For each k we want to construct the <x|E><<E|x> Green's function in the perpendicular direction
    xmin = max(0, ip_perp-int(5/kspan))
    xmax = min(L1d-1, ip_perp+int(5/kspan))

    # Sample Ns points of k
    ksamp = np.arange(Ns)*(2*np.pi)/Ns
    ksamp_zls = np.zeros((Ns, xmax-xmin+1), dtype=complex)
    for ki,k in enumerate(ksamp):
        spf = model2d.SPFMMAProj(k, para_edge)
        found_sp = False
        for row in spf:
            if row[3]: # Find the relevant saddle point
                z = row[0]
                E = row[1]
                Hpp = row[4]
                if perp_edge == 0:
                    zl = model2d.GFMMAProj(E, k, para_edge, *[(x+1,ip_perp+1) for x in range(xmin, xmax+1)])
                else:
                    zl = model2d.GFMMAProj(E, k, para_edge, *[(x-L1d,ip_perp-L1d) for x in range(xmin, xmax+1)])
                ksamp_zls[ki,:] = np.array(zl) * np.sqrt(1/(8*np.pi))*Hpp/z
                found_sp = True
                break
        if not found_sp:
            raise Exception(f"No relevant saddle points found for k = {k}!")
    proj_wv = np.apply_along_axis(lambda kszl:np.interp(ks, ksamp, kszl, period=2*np.pi), 0, ksamp_zls)

    # Do FFT on the 2D array in one dimension and inner product with proj_wv
    # First switch the FFT axis to the first axis
    if para_edge == 1:
        res = np.transpose(res, (1,0,2,3))
    init_ft = np.fft.fft(res[:,np.arange(xmin,xmax+1),:,:], axis=0)
    init_psi_k = np.diagonal(np.tensordot(proj_wv, init_ft, axes=[[1],[1]]), axis1=0, axis2=1)
    # init_psi_k has shape (int_dof, len(times), L1d), with the last index being k
    # Multiply it by the exp(i E_s t) factor
    Eks = np.array(model2d.EnergiesMMA(edge=edge[:1], krange=(ks[0], ks[-1]), prec=L1d-1))
    new_psi_k = init_psi_k * np.exp(-1j*Eks[np.newaxis,:]*evodat.getTimes()[:,np.newaxis])
    # Inverse Fourier transform to (int_dof, len(times), L1d) with the last index being x
    new_psi_x = np.fft.ifft(new_psi_k, axis=-1)
    # Take the norm squared
    seldat1d = np.transpose(np.linalg.norm(new_psi_x, axis=0))**2
    seldat1d /= np.sum(seldat1d, axis=0)
    
    evodat.animate_with_curves(np.arange(L1d), [seldat, seldat1d], FName, legends=["Numerical", "Theory"], xlabel="x", ylabel="Amplitude", title=f"Wave Packet Amplitude in {edge[0]} direction")