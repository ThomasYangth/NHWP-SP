from .HamClass import HamModel
from .EvoDat import EvoDat1D, EvoDat2D
from .WaveFuns import GaussianWave
from .Config import *

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps
from math import floor


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


def plot_pointG_1D_exponent (model:HamModel, L, T, comp=1, dT=0.1, force_evolve = False, ip = None, iprad = 1, ik = 0, addn="", plotre=True, ploterr = False, sel_mode="avg", start_t = 5, orig_t = 2, precision = 0, plot_ax = None):
    """
    For a 1D chain, do time evolution to extract G(x,x,t) at a point x.
    Try to compare G(x,x,t) to the theoretical prediction ~ \sum_{E_s} t^{-a}exp(-i E_s t), where a is 1/2 in the bulk and 3/2 on the boundary.
    Specifically, generates a plot of d/dt[log[G(x,x,t)*t^a]] v.s. t, and compares it to the theoretical prediction.

    Parameters:
    model: HamModel
        The Hamiltonian model to be used.
    L: int
        The size of the 1D chain.
    T: float
        The time to evolve to.
    comp: int
        (Maximal) number of relevant saddle points to take. Default 1.
    plotre: bool
        Whether or not to add a panel to plot Im d/dt G(x,x,t). Default False.
    ploterr: bool
        Whether or not to add a panel to plot the different between the numerical and theoretical growth rates.
    sel_mode: string, "avg" or "inst" or "half"
    start_t: float
        The time at which to start the plot.
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
    if ip < L/4:
        which_boundary = -1
    elif ip > 3*L/4:
        which_boundary = 1

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

    dats = evodat.res[ip,0,:]
    if which_boundary == 0:
        amps = evodat.norms + np.log(np.abs(dats)) + np.log(evodat.times + 1e-10)/2
    else:
        amps = evodat.norms + np.log(np.abs(dats)) + np.log(evodat.times + 1e-10)*3/2
    phas = np.unwrap(np.angle(dats))

    init = evodat.res[:,0,0]

    spf = model.SPFMMA()

    count = 0
    sp0 = None
    fitaddnow = 1
    spfits = []
    for row in spf:
        if row[3]: # Look for relevant saddle points
            z = row[0]
            E = row[1]
            Hpp = row[4]

            rad = max(1, int(5*iprad))

            if which_boundary == -1:
                xmin = max(0, ip-rad)
                xmax = min(L-1, ip+rad)
                zl = model.GFMMA(E, *[(x+1,ip+1) for x in range(xmin, xmax+1)])
                tamp = np.sum(zl * (init[xmin:xmax+1]))
            elif which_boundary == 1:
                xmin = max(0, ip-rad)
                xmax = min(L-1, ip+rad)
                zl = model.GFMMA(E, *[(x-L,ip-L) for x in range(xmin, xmax+1)])
                tamp = np.sum(zl * (init[xmin:xmax+1]))
            else:
                zl = z ** (ip - np.arange(L))
                tamp = np.sum(zl*init)

            # thissp is the log of G(x,x,t) contributed by this saddle point, with the t-power-law part deducted
            if which_boundary == 0:
                thissp = np.log(tamp/z*np.sqrt(1/(2*np.pi))*Hpp/1j) - 1j*E*evodat.times
            else:
                thissp = np.log(tamp/z*np.sqrt(1/(8*np.pi))*Hpp) - 1j*E*evodat.times

            if count == 0:
                sp0 = thissp
                spfits.append(sp0)
            else:
                fitaddnow += np.exp(thissp - sp0)
                # We divide the later saddle point contributions by the first saddle point contribution
                # in order to avoid storing too-large numbers.
                spfits.append(sp0 + np.log(fitaddnow))

            # We take into at max `comp` saddle points.
            count += 1
            if count >= comp:
                break

    sti = int((start_t-evodat.times[0]) / (evodat.times[1]-evodat.times[0]))
    oti = int((orig_t-evodat.times[0]) / (evodat.times[1]-evodat.times[0]))

    def proc_data (amps, phas):
        # See explanation in the function documentation above.
        if sel_mode == "avg":
            ampr = (amps[sti:]-amps[oti]) / (evodat.times[sti:]-evodat.times[oti])
            phsr = - (phas[sti:]-phas[oti]) / (evodat.times[sti:]-evodat.times[oti])
            times = evodat.times[sti:]
        elif sel_mode == "half":
            inds = np.arange(int(sti/2), floor(len(amps)/2))
            ampr = (amps[2*inds]-amps[inds]) / (evodat.times[2*inds]-evodat.times[inds])
            phsr = - (phas[2*inds]-phas[inds]) / (evodat.times[2*inds]-evodat.times[inds])
            times = evodat.times[2*inds]
        elif sel_mode == "inst":
            ampr = (amps[sti:]-amps[oti:-(sti-oti)]) / (evodat.times[sti]-evodat.times[oti])
            phsr = - (phas[sti:]-phas[oti:-(sti-oti)]) / (evodat.times[sti]-evodat.times[oti])
            times = evodat.times[sti:]
        else:
            raise Exception(f"Invalid argument sel_mode={sel_mode}")
        
        return ampr, phsr, times
    
    ampr, phsr, times = proc_data(amps, phas)

    if sel_mode == "inst":
        if which_boundary == 0:
            rename = r"$\mathrm{Re}[i \frac{\mathrm{d}}{\mathrm{d}t}[\log(G(0,0,t)\sqrt{t})]]$"
            imname = r"$\mathrm{Im}[i \frac{\mathrm{d}}{\mathrm{d}t}[\log(G(0,0,t)\sqrt{t})]]$"
        else:
            rename = r"$\mathrm{Re}[i \frac{\mathrm{d}}{\mathrm{d}t}[\log(G(0,0,t)t^{3/2})]]$"
            imname = r"$\mathrm{Im}[i \frac{\mathrm{d}}{\mathrm{d}t}[\log(G(0,0,t)t^{3/2})]]$"
    else:
        if which_boundary == 0:
            rename = r"Re[i log(G(0,0,t)$\sqrt{t}$)/t]"
            imname = r"Im[i log(G(0,0,t)$\sqrt{t}$)/t]"
        else:
            rename = r"Re[i log(G(0,0,t)$t^{3/2}$)/t]"
            imname = r"Im[i log(G(0,0,t)$t^{3/2}$)/t]"

    # Generate the plots
    if plot_ax is None:
        plt.rcParams.update(**PLT_PARAMS)
        total_plots = 1 + plotre + ploterr
        if total_plots == 1:
            plt.figure(figsize=SINGLE_FIGSIZE, layout="constrained")
            axim = plt.gca()
        else:
            if total_plots == 2:
                _, axs = plt.subplots(1, 2, figsize=DOUBLE_FIGSIZE, layout="constrained")
            elif total_plots == 3:
                _, axs = plt.subplots(1, 3, figsize=TRIPLE_FIGSIZE, layout="constrained")

            axim = axs[0]
            if plotre:
                axre = axs[1]
            if ploterr:
                axerr = axs[1+plotre]
    else:
        if isinstance(plot_ax, list) or isinstance(plot_ax, tuple):
            axim = plot_ax[0]
        else:
            axim = plot_ax
            if plotre:
                axre = plot_ax[1]
            if ploterr:
                axerr = plot_ax[1+plotre]

    # Plot real part and imag part
    axim.plot(times, ampr, color="C0", label="Numerical")
    if plotre:
        axre.plot(times, phsr, color="C1", label="Numerical")
    
    for j in range(comp):
        # Plot 'first j saddle points contriubtion' and compare to numerics, for j = 1, ..., comp
        this_sp = spfits[j]
        this_amp = np.real(this_sp)
        this_phs = np.unwrap(np.imag(this_sp))
        this_ampr, this_phsr, this_times = proc_data(this_amp, this_phs)

        axim.plot(this_times, this_ampr, color=f"C{2*j+2}", linestyle="--", label=f"{j+1}SP Theo.")
        if plotre:
            axre.plot(this_times, this_phsr, color=f"C{2*j+3}", linestyle="--", label=f"{j+1}SP Theo.")
        if ploterr:
            axerr.plot(times, np.abs(ampr-this_ampr), color=f"C{2*j+2}", label=f"Im, {j+1}SP")
            axerr.plot(times, np.abs(phsr-this_phsr), color=f"C{2*j+3}", label=f"Re, {j+1}SP")
        
    axim.set_xlabel(r"$t$")
    axim.set_ylabel(imname)
    axim.set_title(imname)
    axim.legend()

    if plotre:
        axre.set_xlabel(r"$t$")
        axre.set_ylabel(rename)
        axre.set_title(rename)
        axre.legend()

    if ploterr:
        axerr.set_xscale("log")
        axerr.set_yscale("log")

        axerr.set_xlabel(r"$t$")
        axerr.set_ylabel("Error")
        axerr.set_title("Numerical v.s. SP Error")

    if plot_ax is not None:
        plt.savefig(FName+sel_mode+".pdf")
        plt.close()


def plot_pointG_1D_WF (model:HamModel, L, T, which_edge, depth, takerange, dT=0.1, force_evolve = False, ip = 0, iprad = 1, ik = 0, addn="", plot_ax = None, takets = [], precision = 0, potential = None):
    """
    For a 1D chain, do time evolution to extract G(x,x0,t) for a given point x0 and a range of x.
    Plot G(x,x0,t) as a function of x, and compare to theoretical expectations.
    Two plots will be generated, corresponding to the amplitude and phase respectively.
    In each plot, the actual wave function at different time steps is superimposed with the theoretical expectation.

    Parameters:
    model: HamModel
        The Hamiltonian model to be used.
    L: int
        The size of the 1D chain.
    T: float
        The time to evolve to.
    which_edge: int
        If 0, plot near the left edge; if 1, plot near the right edge.
    depth: int
        The initial position of the wave packet, as measured from the edge. 0 means the site on the boundary, 1 means one site from the boundary, etc.
    takerange: int
        The number of sites, measured from the boundary, to be plotted.
    plot_ax: tuple of two plt.Axes
        If given, the plot will be drawn on these axes. If not given, a new figure will be created.
    takets: list of floats
        Each element corresponds to a time when we take a snapshot of the wave function. Time points should be in the range [0,T].
        Default empty, in which case only one snapshot is taken at T.
    potential: array of length L*model.int_dim
        Corresponding to a diagonal potential that is added to the Hamiltonian by hand. 0 by default.
    ** FOR OTHER PARAMETERS SEE `plot_pointG_1D_tslope` **
    """

    Ham = model([L])
    Hamname = model.name+(("_"+addn) if addn!="" else "")
    FName = "{}_WF_{}{}i{}_L{}T{}".format(Hamname, 'L' if which_edge<=0 else 'R', depth, ip, L, T)
    
    if which_edge > 0:
        ip = L-1-depth
        left = L-takerange
        right = L
    else:
        ip = depth
        left = 0
        right = takerange

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
        evodat = EvoDat1D.from_evolve(Ham, T, dT, init, FName, precision=precision, potential=potential, return_idnorm = True)
        evodat.save()

    if plot_ax is None:
        evodat.plot_profile()
        plt.rcParams.update(**PLT_PARAMS)
        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=DOUBLE_FIGSIZE, layout="constrained", width_ratios=[100,100,1])
        ax3.axis('off')
    else:
        ax1, ax2 = plot_ax

    if len(takets) == 0:
        takets = [T]
    
    # Convert the time points in takets into indices
    itakets = [min(round(((t-evodat.times[0])/(evodat.times[1]-evodat.times[0]))), evodat.T-1) for t in takets]

    spf = model.SPFMMA()
    for row in spf:
        if row[3]:
            E = row[1]
            break
    offset = -L if which_edge>0 else 1

    xs = np.arange(left, right)
    theos = model.GFMMA(E, *[(x+offset,ip+offset) for x in range(left, right)])
    theo_amp = np.abs(theos)
    theo_phs = np.unwrap(np.angle(theos))
    theo_amp /= theo_amp[ip-left]
    theo_phs = theo_phs[1:] - theo_phs[:-1]

    ax1.plot(xs, theo_amp, label = "Theory", color="gray", linestyle="--", zorder=2, linewidth=THICK_LINEWIDTH)
    ax2.plot(xs[1:] if which_edge<=0 else xs[:-1], theo_phs, label = "Theory", color="gray", linestyle="--", zorder=2, linewidth=THICK_LINEWIDTH)

    ax1.set_title("Amplitude")
    ax2.set_title("Phase")
    ax1.set_xlabel("$x$")
    ax2.set_xlabel("$x$")
    ax1.set_ylabel(r"$|\psi(x)|$")
    ax2.set_ylabel(r"$\Delta \mathrm{arg}\psi(x)$")
    ax1.set_yscale("log")

    colors = colormaps["plasma"](np.linspace(0,1,len(itakets)))

    for i, taketi in enumerate(itakets):
        dat = evodat.res[left:right, 0, taketi]
        amp = np.abs(dat) * np.exp(evodat.idnorm[left:right, 0, taketi])
        phs = np.unwrap(np.angle(dat))
        amp /= amp[ip-left]
        phs = phs[1:] - phs[:-1]
        ax1.plot(xs, amp, label = "t = {:.1f}".format(evodat.times[taketi]), color=colors[i], zorder=1)
        ax2.plot(xs[1:] if which_edge<=0 else xs[:-1], phs, label = "t = {:.1f}".format(evodat.times[taketi]), color=colors[i], zorder=1)
    
    if plot_ax is None:
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    else:
        ax1.legend()
        ax2.legend()
    
    if plot_ax is not None:
        plt.savefig(FName+f"_wf[{left}:{right}].pdf")
        plt.close()

def plot_pointG_1D_vec (model:HamModel, L, T, dT=0.1, force_evolve=False, ip = None, iprad = 1, ik = 0, addn="", start_t = 5, error_log = False, precision = 0, plot_ax = None):
    """
    For a 1D chain, do time evolution to extract G_ab(x,x,t) for a given point x and internal degrees of freedom a, b.
    Plot G_ab(x,x,t)/G_11(x,x,t) as a function of x, and compare to theoretical expectations (for now the theory is only available in the bulk / PBC).

    Parameters:
    model: HamModel
        The Hamiltonian model to be used.
    L: int
        The size of the 1D chain.
    T: float
        The time to evolve to.
    plot_ax: plt.Axes or tuple of two plt.Axes
        If given, the plot will be drawn on this axi(e)s. If not given, a new figure will be created.
    error_log: bool, default False
        If True, generates two plots, the first plots G_ab(t)/G_11(t),
        the second plots in log scale the different of this numerical value with the theoretical prediction.
    ** FOR OTHER PARAMETERS SEE `plot_pointG_1D_tslope` **
    """

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
        ip = int(abs(L)/2)
    else:
        FName0 += f"ip{ip}"
        if ip < 0:
            ip = abs(L)+ip

    for i in range(B):
        FName = f"{FName0}_B{i}"
        # Initialize with the spin points in the i-th axis
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
                    print("****WARNING**** CLOSE EIGENVALUES, RESULT MAY BE OFF")
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
    
    if plot_ax is None:
        plt.rcParams.update(**PLT_PARAMS)
        # Generate the plots
        if not error_log:
            plt.figure(figsize=SINGLE_FIGSIZE, layout="constrained")
            ax1 = plt.gca()
        else:
            _, (ax1,ax2) = plt.subplots(1, 2, figsize=DOUBLE_FIGSIZE, layout="constrained")
    else:
        if error_log:
            (ax1, ax2) = plot_ax
        else:
            ax1 = plot_ax

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

    if plot_ax is not None:
        plt.savefig(FName0+"_VEC.pdf")
        plt.close()

def plot_and_compare_2D_Edge (model2d:HamModel, L, W, T, Ns = 0, edge="x-", k = 0, ipdepth = 0, kspan = 0.1, dT=0.2, force_evolve = False, addname = "", snapshots = []):

    tkdT = 0.25

    Lp = abs(L)
    Wp = abs(W)

    FName = f"{model2d.name}_L{L}W{W}_{edge}{ipdepth}_k{k}s{kspan}_T{T}_{addname}"
    if edge == "x-":
        ip = (int(Lp/2), 0+ipdepth)
        ik = (k,0)
        para_edge = 0
        perp_edge = 0
    elif edge == "x+":
        ip = (int(Lp/2), Wp-1-ipdepth)
        ik = (k,0)
        para_edge = 0
        perp_edge = 1
    elif edge == "y-":
        ip = (0+ipdepth, int(Wp/2))
        ik = (0,k)
        para_edge = 1
        perp_edge = 0
    elif edge == "y+":
        ip = (Lp-1-ipdepth, int(Wp/2))
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
    if para_edge == 0:
        seldat = seldat[:,y0,:]
    else:
        seldat = seldat[x0,:,:]
    seldat = seldat**2
    seldat /= np.sum(seldat, axis=0)

    print("Initial WP at t=0: ", seldat[:,0])

    # Construct the projected wave function in one dimension
    if para_edge == 0:
        ip_perp = y0
        L1d = Lp
        Lperp = Wp
    else:
        ip_perp = x0
        L1d = Wp
        Lperp = Lp
    # For each k, we treat this as a 1D system and get an effective amplitude
    # This gives us the projection of our wave function onto the edge mode.
    xs = np.arange(L1d)
    ks = xs*(2*np.pi)/L1d

    # For each k we want to construct the <x|E><<E|x> Green's function in the perpendicular direction
    xmin = max(0, ip_perp-int(2/kspan))
    xmax = min(Lperp-1, ip_perp+int(2/kspan))

    # Sample Ns points of k
    if Ns <= 0:
        Ns = L1d
    ksamp = np.arange(Ns)*(2*np.pi)/Ns
    if perp_edge == 0:
        ksamp_zls_gfmma = model2d.GFMMAProj(Ns, para_edge,  *[(ip_perp+1,x+1) for x in range(xmin, xmax+1)])
    else:
        ksamp_zls_gfmma = model2d.GFMMAProj(Ns, para_edge,  *[(ip_perp-Lperp,x-Lperp) for x in range(xmin, xmax+1)])
    Eks = np.array([row[0] for row in ksamp_zls_gfmma])
    print("List of Eks:")
    print(Eks)
    print("======")
    ksamp_zls = np.array([row[1] for row in ksamp_zls_gfmma])
        
    if Ns != L1d:
        proj_wv = np.apply_along_axis(lambda kszl:np.interp(ks, ksamp, kszl, period=2*np.pi), 0, ksamp_zls)
        Eks = np.interp(ks, ksamp, Eks, period=2*np.pi)
    else:
        proj_wv = ksamp_zls

    # Do FFT on the 2D array in one dimension and inner product with proj_wv
    # First switch the FFT axis to the first axis
    if para_edge == 1:
        res = np.transpose(res, (1,0,2,3))
    init_ft = np.fft.fft(res[:,np.arange(xmin,xmax+1),:,:], axis=0)
    print("FFT at t=0: ", init_ft[:,ip_perp-xmin,0,0])
    print("At k=0, init_ft is:", init_ft[0,:,0,0])
    print("At k=0, proj_wv is:", proj_wv[0,:])
    init_psi_k = np.diagonal(np.tensordot(proj_wv, init_ft, axes=[[1],[1]]), axis1=0, axis2=1)
    print("Initial Psi_k at t=0: ", init_psi_k[0,0,:])
    # init_psi_k has shape (int_dof, len(times), L1d), with the last index being k
    # Multiply it by the exp(i E_s t) factor
    new_psi_k = init_psi_k * np.exp(-1j*Eks[np.newaxis,:]*evodat.getTimes()[:,np.newaxis])
    # Inverse Fourier transform to (int_dof, len(times), L1d) with the last index being x
    new_psi_x = np.fft.ifft(new_psi_k, axis=-1)
    # Take the norm squared
    seldat1d = np.transpose(np.linalg.norm(new_psi_x, axis=0))**2
    seldat1d /= np.sum(seldat1d, axis=0)
    
    evodat.animate_with_curves(np.arange(L1d), [seldat, seldat1d], "EffEdge", legends=["Numerical", "Theory"], xlabel="x", ylabel="Amplitude", title=f"Wave Packet Amplitude in {edge[0]} direction", snapshots=snapshots)