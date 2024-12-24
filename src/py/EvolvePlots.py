# EvolvePlots.py
# Contains several functions that generate sophisticated plots comparing saddle-point results to real-time results.

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

if USE_GPU:
    import cupy as cp


def plot_velocities_1D (model:HamModel, L, T, dT=0.01, tkpts=2, vmax = 0, force_evolve = False, output=print, ip = None, comp=0, addn="", doplots=True):
    """Fits the growth rate on a series of world lines.

    Inputs:
    model - 
    comp - whether or not we compare the velocities to theoretical predictions. comp = 0 indicates no comparion,
        comp = 1 indicates compare to the growth rate of the valid saddle point, comp = 2 means also plotting other
        saddle points.
    """
    Ham = model([L])
    Hamname = model.name+(("_"+addn) if addn!="" else "")
    FName = "{}_Vel_L{}_T{}".format(Hamname, L, T)
    if ip is None:
        ip = int(L/2)
    else:
        FName += "_ip{}".format(ip)
        if ip < 0:
            ip = L-ip

    if model.int_dim == 1:
        intvec = [1]
    else:
        intvec = np.random.randn(model.int_dim) + 1j*np.random.randn(model.int_dim)
        intvec = intvec / np.linalg.norm(intvec)

    init = GaussianWave([L], (ip,), 1, intvec = intvec)
    tkdT = T/tkpts
    try:
        if force_evolve:
            raise Exception()
        evodat = EvoDat1D.read(FName+".txt")
    except Exception:
        evodat = EvoDat1D.from_evolve(Ham.time_evolve, T+0.1*tkdT, dT, init, FName, takedT = tkdT)
        evodat.save()

    if doplots:
        evodat.plot_profile()
    allnorms = np.linalg.norm(evodat.res, axis=1)
    profiles = np.log(allnorms) + evodat.norms[np.newaxis,:] # The profile at each time slice
    leftdist = floor(ip/tkpts)
    rightdist = floor((L-1-ip)/tkpts)
    if vmax > 0:
        leftdist = min(int(vmax*tkdT), leftdist)
        rightdist = min(int(vmax*tkdT), rightdist)
    select_profiles = [profiles[ip-leftdist*i:ip+rightdist*i+1:i,i] for i in range(1,len(evodat.norms))]
    lyaps = (np.diff(select_profiles, axis=0) + np.log([1+1/(i+1) for i in range(tkpts-1)])[:,np.newaxis]/2)/tkdT
    vs = np.arange(-leftdist, rightdist+1)/tkdT
    if comp > 0:

        #prec = rightdist+leftdist
        prec = 20

        nvs = [vs[0]*(1-i/prec) + vs[-1]*i/prec for i in range(prec+1)]
        
        if comp == 1:
            mmalyaps = np.array(Ham.GrowthsMMA(vs[0],vs[-1],prec))

        else:
            mmalyaps = []
            invalidlyaps = []
            for v in nvs:
                mmasp = Ham.SPFMMA(v)
                thisinvalids = []
                foundvalid = False
                for ls in mmasp:
                    if not foundvalid and ls[3]:
                        foundvalid = True
                        mmalyaps.append(ls[2])
                    else:
                        thisinvalids.append(ls[2])
                invalidlyaps.append(thisinvalids)
            mmalyaps = np.array(mmalyaps)
            invalidlyaps = np.array(invalidlyaps).transpose().tolist()

    if doplots:
        if tkpts > 2 and comp:
            fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15,6)) # If there are more than one curves,
                                                # create a subplot that compares the deviation of different time-slices with theory
        else:
            fig, ax1 = plt.subplots(1, 1)
        for i in range(tkpts-1):
            ax1.plot(vs, lyaps[i,:], label = "Num. t = {}".format(evodat.times[i+2]), zorder = 3)

        num_lim = ax1.get_ylim()

        if comp > 0:
            ax1.plot(nvs, mmalyaps, c = "lightgray", linewidth = 7, alpha = 1, label = "Theory", zorder = 2)
        if comp == 2:
            labelled = False
            for lyaps in invalidlyaps:
                if not labelled:
                    ax1.plot(nvs, lyaps, c = "lightgray", linewidth = 1, label = "Invalid SP", zorder = 1)
                    labelled = True
                else:
                    ax1.plot(nvs, lyaps, c = "lightgray", linewidth = 1, zorder = 1)

            ax1.set_ylim([(3*num_lim[0]-num_lim[1])/2, (3*num_lim[1]-num_lim[0])/2])

        if tkpts > 2 and comp > 0:
            devs = np.linalg.norm(lyaps - mmalyaps[np.newaxis,:], axis=1)
            ax2.plot(evodat.times[2:], devs)
            ax2.set_xlabel("t")
            ax2.set_ylabel("Deviation from Theory")
            ax2.set_title("Lyapunov Exponent Curve - Deviation from Theory")
        ax1.set_xlabel("v")
        ax1.set_ylabel("lambda")
        ax1.legend()
        ax1.set_title("Lyapnov exponents")
        plt.savefig(FName+".jpg")
        plt.close()

    return vs, lyaps, evodat

def plot_pointG_1D (model:HamModel, L, T, dT=0.1, force_evolve = False, output=print, ip = None, iprad = 1, ik = 0, comp=True, addn="", doplots=True, plotre=True, alt=None, sel_mode="avg", start_t = 5, orig_t = 2, error_log = False):

    Ham = model([L])
    Hamname = model.name+(("_"+addn) if addn!="" else "")
    FName = "{}_FG_L{}_T{}".format(Hamname, L, T)
    if ip is None:
        ip = int(L/2)
    else:
        FName += "_ip{}".format(ip)
        if ip < 0:
            ip = L+ip

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
        evodat = EvoDat1D.from_evolve(Ham.time_evolve, T, dT, init, FName)
        evodat.save()

    if doplots:
        evodat.plot_profile()

    dats = evodat.res[ip,0,:]
    amps = evodat.norms + np.log(np.abs(dats)) + np.log(evodat.times + 1e-10)/2
    phas = np.unwrap(np.angle(dats))

    sti = int(start_t / (evodat.times[1]-evodat.times[0]))

    if sel_mode == "avg":
        oti = int(orig_t / (evodat.times[1]-evodat.times[0]))
        ampr = (amps[sti:]-amps[oti]) / (evodat.times[sti:]-evodat.times[oti])
        phsr = - (phas[sti:]-phas[oti]) / (evodat.times[sti:]-evodat.times[oti])
        times = evodat.times[sti:]
    elif sel_mode == "half":
        inds = np.arange(int(sti/2), floor(len(amps)/2))
        ampr = (amps[2*inds]-amps[inds]) / (evodat.times[2*inds]-evodat.times[inds])
        phsr = - (phas[2*inds]-phas[inds]) / (evodat.times[2*inds]-evodat.times[inds])
        times = evodat.times[2*inds]
    elif sel_mode == "inst":
        ampr = (amps[sti:]-amps[:-sti]) / (evodat.times[sti]-evodat.times[0])
        phsr = - (phas[sti:]-phas[:-sti]) / (evodat.times[sti]-evodat.times[0])
        times = evodat.times[sti:]
    else:
        raise Exception(f"Invalid argument sel_mode={sel_mode}")

    E = None

    if comp:

        spf = model.SPFMMA()
        for row in spf:
            if row[3]:
                E = row[1]
                break

    if sel_mode == "inst":
        rename = r"Re[i d/dt[log(G(0,0,t)$\sqrt{t}$)]]"
        imname = r"Im[i d/dt[log(G(0,0,t)$\sqrt{t}$)]]"
    else:
        rename = r"Re[i log(G(0,0,t)$\sqrt{t}$)/t]"
        imname = r"Im[i log(G(0,0,t)$\sqrt{t}$)/t]"

    if doplots:
        plt.figure()
        ax1 = plt.gca()

        if comp and error_log:

            ax1.plot(times, np.abs(ampr-np.imag(E)), color="C0", label="Error Im")
            if plotre:
                ax1.plot(times, np.abs(phsr-np.real(E)), color="C1", label="Error Re")
            ax1.set_yscale("log")
            ax1.set_xscale("log")
            ax1.set_ylim([max(1e-5, ax1.get_ylim()[0]), ax1.get_ylim()[1]])
            ax1.set_xlabel("t")
            ax1.set_ylabel(imname[3:-1])

        else:

            ax1.plot(times, ampr, color="C0", label=imname)
            if plotre:
                ax2 = ax1.twinx()
                ax2.plot(times, phsr, color="C1", label=rename)
            
            if comp:
                ax1.plot([times[0], times[-1]], [np.imag(E)]*2, color="C0", linestyle="--", label=r"Im$\epsilon^\ast$")
                if alt is not None:
                    ax1.plot([times[0], times[-1]], [alt]*2, color="C2", linestyle="--", label=r"Im$\epsilon^\ast$ alt.theo.")
                ax1.set_ylim([min(ax1.get_ylim()[0], np.imag(E)-0.7), max(ax1.get_ylim()[1], np.imag(E)+0.3)])
                #ax1.set_ylim([-3,0.5])
                if plotre:
                    ax2.plot([times[0], times[-1]], [np.real(E)]*2, color="C1", linestyle="--", label=r"Re$\epsilon^\ast$")
                    ax2.set_ylim([min(ax2.get_ylim()[0], np.real(E)-0.3), max(ax2.get_ylim()[1], np.real(E)+0.7)])
                    #ax2.set_ylim([-1.5,4.5])
                    #ax2.set_ylim([ax2.get_ylim()[0], np.real(E)+1])

            ax1.set_xlabel("t")
            #ax1.set_ylim([-0.9,0])
            ax1.set_ylabel(imname)

            if plotre:
                ax2.set_ylabel(rename)
                h1,l1 = ax1.get_legend_handles_labels()
                h2,l2 = ax2.get_legend_handles_labels()
                ax1.legend(h1+h2, l1+l2, loc="upper right")
            else:
                plt.legend()

        plt.savefig(FName+sel_mode+".jpg")

    if comp:
        return ampr, phsr, E
    else:
        return ampr, phsr
    
def analyze_pointG_1D (model:HamModel, L, T, dT=0.1, force_evolve = False, output=print, ip = None, iprad = 1, ik = 0, comp=True, addn="", doplots=True, plotre=True, alt=None, sel_mode="avg", start_t = 5, orig_t = 2, error_log = False, deduct_average = False, precision=20):

    Ham = model([L])
    Hamname = model.name+(("_"+addn) if addn!="" else "")
    FName = "{}_FG_L{}_T{}_p{}".format(Hamname, L, T, precision)
    if ip is None:
        ip = int(L/2)
    else:
        FName += "_ip{}".format(ip)
        if ip < 0:
            ip = L+ip

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
        evodat = EvoDat1D.from_evolve(Ham.time_evolve, T, dT, init, FName, precision=precision)
        evodat.save()

    if doplots:
        evodat.plot_profile()

    dats = evodat.res[ip,0,:]

    sti = int(start_t / (evodat.times[1]-evodat.times[0]))

    E = None

    if comp:

        spf = model.SPFMMA()
        for row in spf:
            if row[3]:
                E = row[1]
                break

    amps = evodat.norms + np.log(np.abs(dats)) + np.log(evodat.times + 1e-10)/2
    phas = np.unwrap(np.angle(dats))

    if sel_mode == "avg":
        oti = int(orig_t / (evodat.times[1]-evodat.times[0]))
        ampr = (amps[sti:]-amps[oti]) / (evodat.times[sti:]-evodat.times[oti])
        phsr = - (phas[sti:]-phas[oti]) / (evodat.times[sti:]-evodat.times[oti])
        times = evodat.times[sti:]
    elif sel_mode == "half":
        inds = np.arange(int(sti/2), floor(len(amps)/2))
        ampr = (amps[2*inds]-amps[inds]) / (evodat.times[2*inds]-evodat.times[inds])
        phsr = - (phas[2*inds]-phas[inds]) / (evodat.times[2*inds]-evodat.times[inds])
        times = evodat.times[2*inds]
    elif sel_mode == "inst":
        ampr = (amps[sti:]-amps[:-sti]) / (evodat.times[sti]-evodat.times[0])
        phsr = - (phas[sti:]-phas[:-sti]) / (evodat.times[sti]-evodat.times[0])
        times = evodat.times[sti:]
    elif sel_mode == "logv":
        ampr = amps[sti:]
        phsr = - phas[sti:]
        times = evodat.times[sti:]
    else:
        raise Exception(f"Invalid argument sel_mode={sel_mode}")
    
    if comp and deduct_average:

        avg_amp = np.average(dats[sti:]*np.sqrt(evodat.times[sti:])*np.exp(1j*E*evodat.times[sti:]+evodat.norms[sti:]))
        dats_da = dats - avg_amp*np.exp(-1j*E*evodat.times-evodat.norms)/np.sqrt(evodat.times+1e-10)

        amps_da = evodat.norms + np.log(np.abs(dats_da)) + np.log(evodat.times + 1e-10)/2
        phas_da = np.unwrap(np.angle(dats_da))

        if sel_mode == "avg":
            oti = int(orig_t / (evodat.times[1]-evodat.times[0]))
            ampr_da = (amps_da[sti:]-amps_da[oti]) / (evodat.times[sti:]-evodat.times[oti])
            phsr_da = - (phas_da[sti:]-phas_da[oti]) / (evodat.times[sti:]-evodat.times[oti])
        elif sel_mode == "half":
            inds = np.arange(int(sti/2), floor(len(amps_da)/2))
            ampr_da = (amps_da[2*inds]-amps_da[inds]) / (evodat.times[2*inds]-evodat.times[inds])
            phsr_da = - (phas_da[2*inds]-phas_da[inds]) / (evodat.times[2*inds]-evodat.times[inds])
        elif sel_mode == "inst":
            ampr_da = (amps_da[sti:]-amps_da[:-sti]) / (evodat.times[sti]-evodat.times[0])
            phsr_da = - (phas_da[sti:]-phas_da[:-sti]) / (evodat.times[sti]-evodat.times[0])
        elif sel_mode == "logv":
            ampr_da = amps_da[sti:]
            phsr_da = - phas_da[sti:]
            times_da = evodat.times[sti:]

    if sel_mode == "inst":
        rename = r"Re[i d/dt[log(G(0,0,t)$\sqrt{t}$)]]"
        imname = r"Im[i d/dt[log(G(0,0,t)$\sqrt{t}$)]]"
    elif sel_mode == "logv":
        rename = r"Re[i log(G(0,0,t)$\sqrt{t}$)]"
        imname = r"Im[i log(G(0,0,t)$\sqrt{t}$)]"
    else:
        rename = r"Re[i log(G(0,0,t)$\sqrt{t}$)/t]"
        imname = r"Im[i log(G(0,0,t)$\sqrt{t}$)/t]"

    if doplots:
        plt.figure()
        ax1 = plt.gca()

        if comp and error_log:

            ax1.plot(times, np.abs(ampr-np.imag(E)), color="C0", label="Error Im")
            if plotre:
                ax1.plot(times, np.abs(phsr-np.real(E)), color="C1", label="Error Re")
            ax1.set_yscale("log")
            ax1.set_xscale("log")
            ax1.set_ylim([max(1e-5, ax1.get_ylim()[0]), ax1.get_ylim()[1]])
            ax1.set_xlabel("t")
            ax1.set_ylabel(imname[3:-1])

        elif comp and deduct_average:

            ax1.plot(times, ampr, color="C0", label=imname)
            ax1.plot(times, ampr_da, color="C2", label=imname+"_DA")
            if sel_mode != "logv":
                ax1.plot([times[0], times[-1]], [np.imag(E)]*2, color="C0", linestyle="--", label=r"Im$\epsilon^\ast$")
            if plotre:
                ax2 = ax1.twinx()
                ax2.plot(times, phsr, color="C1", label=rename)
                ax2.plot(times, phsr_da, color="C3", label=rename+"_DA")
                if sel_mode != "logv":
                    ax2.plot([times[0], times[-1]], [np.real(E)]*2, color="C1", linestyle="--", label=r"Re$\epsilon^\ast$")
            ax1.set_xlabel("t")
            ax1.set_ylabel(imname)

            if plotre:
                ax2.set_ylabel(rename)
                h1,l1 = ax1.get_legend_handles_labels()
                h2,l2 = ax2.get_legend_handles_labels()
                ax1.legend(h1+h2, l1+l2, loc="upper right")
            else:
                plt.legend()

        else:

            ax1.plot(times, ampr, color="C0", label=imname)
            if plotre:
                ax2 = ax1.twinx()
                ax2.plot(times, phsr, color="C1", label=rename)
            
            if comp:
                ax1.plot([times[0], times[-1]], [np.imag(E)]*2, color="C0", linestyle="--", label=r"Im$\epsilon^\ast$")
                if alt is not None:
                    ax1.plot([times[0], times[-1]], [alt]*2, color="C2", linestyle="--", label=r"Im$\epsilon^\ast$ alt.theo.")
                ax1.set_ylim([min(ax1.get_ylim()[0], np.imag(E)-0.7), max(ax1.get_ylim()[1], np.imag(E)+0.3)])
                #ax1.set_ylim([-3,0.5])
                if plotre:
                    ax2.plot([times[0], times[-1]], [np.real(E)]*2, color="C1", linestyle="--", label=r"Re$\epsilon^\ast$")
                    ax2.set_ylim([min(ax2.get_ylim()[0], np.real(E)-0.3), max(ax2.get_ylim()[1], np.real(E)+0.7)])
                    #ax2.set_ylim([-1.5,4.5])
                    #ax2.set_ylim([ax2.get_ylim()[0], np.real(E)+1])

            ax1.set_xlabel("t")
            #ax1.set_ylim([-0.9,0])
            ax1.set_ylabel(imname)

            if plotre:
                ax2.set_ylabel(rename)
                h1,l1 = ax1.get_legend_handles_labels()
                h2,l2 = ax2.get_legend_handles_labels()
                ax1.legend(h1+h2, l1+l2, loc="upper right")
            else:
                plt.legend()

        plt.savefig(FName+sel_mode+".jpg")

    if comp:
        return ampr, phsr, E
    else:
        return ampr, phsr    


def plot_pointG_1D_spsimp (model:HamModel, L, T, dT=0.1, force_evolve = False, output=print, ip = None, iprad = 1, ik = 0, comp=1, addn="", doplots=True, plotre=True, ploterr = False, alt=None, sel_mode="avg", start_t = 5, orig_t = 2, error_log = False, precision = 0):

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
        evodat = EvoDat1D.from_evolve(Ham.time_evolve, T, dT, init, FName, precision=precision, return_idnorm=False)
        evodat.save()

    if doplots:
        evodat.plot_profile()

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
        if row[3]:
            z = row[0]
            E = row[1]
            Hpp = row[4]

            rad = max(1, int(5*iprad))

            if which_boundary == -1:
                xmin = max(0, ip-rad)
                xmax = min(L-1, ip+rad)
                zl = model.GFDMMA(E, *[(x+1,ip+1) for x in range(xmin, xmax+1)])
                tamp = np.sum(zl * (init[xmin:xmax+1]))
            elif which_boundary == 1:
                xmin = max(0, ip-rad)
                xmax = min(L-1, ip+rad)
                zl = model.GFDMMA(E, *[(x-L,ip-L) for x in range(xmin, xmax+1)])
                tamp = np.sum(zl * (init[xmin:xmax+1]))
            else:
                zl = z ** (ip - np.arange(L))
                tamp = np.sum(zl*init)

            if which_boundary == 0:
                thissp = np.log(tamp/z*np.sqrt(1/(2*np.pi))*Hpp/1j) - 1j*E*evodat.times
            else:
                thissp = np.log(tamp/z*np.sqrt(1/(4*np.pi))*Hpp) - 1j*E*evodat.times

            if count == 0:
                sp0 = thissp
                spfits.append(sp0)
            else:
                fitaddnow += np.exp(thissp - sp0)
                spfits.append(sp0 + np.log(fitaddnow))

            count += 1
            if count >= comp:
                break

    sti = int((start_t-evodat.times[0]) / (evodat.times[1]-evodat.times[0]))
    oti = int((orig_t-evodat.times[0]) / (evodat.times[1]-evodat.times[0]))

    def proc_data (amps, phas):

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


    lw = 1

    plt.rcParams.update({"font.size":8, "lines.linewidth":lw, "mathtext.fontset":"cm"})

    # Generate the plots
    total_plots = 1 + plotre + ploterr

    if total_plots == 1:
        plt.figure(figsize=(3.375, 2.4), layout="constrained")
        axim = plt.gca()
    else:
        if total_plots == 2:
            _, axs = plt.subplots(1, 2, figsize=(6, 2.4), layout="constrained")
        elif total_plots == 3:
            _, axs = plt.subplots(1, 3, figsize=(7, 2.4), layout="constrained")

        axim = axs[0]
        if plotre:
            axre = axs[1]
        if ploterr:
            axerr = axs[1+plotre]

    # Plot real part and imag part

    axim.plot(times, ampr, color="C0", linewidth=lw, label="Numerical")
    if plotre:
        axre.plot(times, phsr, color="C1", linewidth=lw, label="Numerical")
    
    for j in range(comp):

        this_sp = spfits[j]
        this_amp = np.real(this_sp)
        this_phs = np.unwrap(np.imag(this_sp))
        this_ampr, this_phsr, this_times = proc_data(this_amp, this_phs)

        axim.plot(this_times, this_ampr, color=f"C{2*j+2}", linewidth=lw, linestyle="--", label=f"{j+1}SP Theo.")
        if plotre:
            axre.plot(this_times, this_phsr, color=f"C{2*j+3}", linewidth=lw, linestyle="--", label=f"{j+1}SP Theo.")
        if ploterr:
            axerr.plot(times, np.abs(ampr-this_ampr), color=f"C{2*j+2}", linewidth=lw, label=f"Im, {j+1}SP")
            axerr.plot(times, np.abs(phsr-this_phsr), color=f"C{2*j+3}", linewidth=lw, label=f"Re, {j+1}SP")

        
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

    plt.savefig(FName+sel_mode+".pdf")
    plt.close()

########## UPDATED METHOD ##############
def plot_pointG_1D_tslope (model:HamModel, L, T, dT=0.1, force_evolve = False, ip = None, iprad = 1, ik = 0, start_t=5, addn="", precision = 0, plot_ax = None):

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
def plot_pointG_1D_WF (model:HamModel, L, T, dT=0.1, force_evolve = False, ip = None, iprad = 1, ik = 0, addn="", doplots = True, takets = [], precision = 0, takerange = 20, fit = False, potential = None):

    Ham = model([L])
    Hamname = model.name+(("_"+addn) if addn!="" else "")
    FName = "{}_WF_L{}_T{}".format(Hamname, L, T)
    if ip is None:
        ip = int(L/2)
    else:
        FName += "_ip{}".format(ip)
        if ip < 0:
            ip = L+ip

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

    if doplots:
        evodat.plot_profile()

    if len(takets) == 0:
        takets = [T]
    
    itakets = [ceil(((t-evodat.times[0])/(evodat.times[1]-evodat.times[0])))-1 for t in takets]

    for i, taketi in enumerate(itakets):
        left = max(0,ip-takerange)
        right = min(L,ip+takerange)
        dat = evodat.res[left:right, 0, taketi]
        amp = np.log(np.abs(dat)) + evodat.idnorm[left:right, 0, taketi]
        phs = np.unwrap(np.angle(dat) - np.angle(evodat.res[ip,0,taketi]))
        _, (ax1,ax2) = plt.subplots(1, 2)
        xs = np.arange(left, right)
        ax1.plot(xs, amp, label = "Amplitude")
        ax2.plot(xs, phs, label = "Phase")

        if fit:
            poly1 = np.polyfit(xs, amp, deg=1)
            ampc = poly1[0]
            ax1.plot(xs, np.poly1d(poly1)(xs), color="C1", linestyle="--")
            poly2 = np.polyfit(xs, phs, deg=1)
            phsc = poly2[0]
            ax2.plot(xs, np.poly1d(poly2)(xs), color="C1", linestyle="--")
            beta = np.exp(ampc+1j*phsc)
            ax2.set_title("Fitted beta = {:.5f} + {:.5f}j".format(np.real(beta), np.imag(beta)))

        ax1.set_xlabel("x")
        ax1.set_ylabel("Log |psi|")
        ax2.set_xlabel("x")
        ax2.set_ylabel("Phase")
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

def plot_pointG_1D_spsimp_div (model:HamModel, L, T, dT=0.1, force_evolve = False, output=print, ip = None, iprad = 1, ik = 0, comp=1, addn="", doplots=True, plotre=True, alt=None, sel_mode="avg", start_t = 5, orig_t = 2, error_log = False, precision = 0):

    Ham = model([L])
    Hamname = model.name+(("_"+addn) if addn!="" else "")
    FName = "{}_FG_L{}_T{}".format(Hamname, L, T)
    if ip is None:
        ip = int(L/2)
    else:
        FName += "_ip{}".format(ip)
        if ip < 0:
            ip = L+ip

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
        evodat = EvoDat1D.from_evolve(Ham.time_evolve, T, dT, init, FName, precision=precision, return_idnorm=False)
        evodat.save()

    if doplots:
        evodat.plot_profile()

    dats = evodat.res[ip,0,:]
    amps = evodat.norms + np.log(np.abs(dats)) + np.log(evodat.times + 1e-10)/2
    phas = np.angle(dats)

    #same = np.argwhere(np.abs(np.diff(dats)) < 1e-8)
    #print(evodat.times[same])

    init = evodat.res[:,0,0]

    spf = model.SPFMMA()

    count = 0
    sp0 = None
    fitaddnow = 1
    spfits = []
    for row in spf:
        if row[3]:
            z = row[0]
            E = row[1]
            Hpp = row[4]

            zl = z ** (ip - np.arange(L))
            tamp = np.sum(zl*init)
            thissp = np.log(tamp/z*np.sqrt(1j/(2*np.pi*Hpp))) - 1j*E*evodat.times

            if count == 0:
                sp0 = thissp
                spfits.append(sp0)
            else:
                fitaddnow += np.exp(thissp - sp0)
                spfits.append(sp0 + np.log(fitaddnow))

            count += 1
            if count >= comp:
                break

    sti = int(start_t / (evodat.times[1]-evodat.times[0]))

    if sel_mode == "inst":
        rename = r"$\mathrm{Re}[i \frac{\mathrm{d}}{\mathrm{d}t}[\log(G(0,0,t)\sqrt{t})]]$"
        imname = r"$\mathrm{Im}[i \frac{\mathrm{d}}{\mathrm{d}t}[\log(G(0,0,t)\sqrt{t})]]$"
    else:
        rename = r"Re[i log(G(0,0,t)$\sqrt{t}$)/t]"
        imname = r"Im[i log(G(0,0,t)$\sqrt{t}$)/t]"

    lw = 1

    plt.rcParams.update({"font.size":8, "lines.linewidth":lw, "mathtext.fontset":"cm"})

    if plotre:
        _, (ax1,ax2) = plt.subplots(1, 2, figsize=(5, 2.2), layout="constrained")
    else:
        plt.figure(figsize=(3.375, 2.4))
        ax1 = plt.gca()
    
    for j in range(comp):

        this_sp = spfits[j]
        this_amp = amps - np.real(this_sp)
        this_phs = np.unwrap(phas - np.imag(this_sp))
        this_ampr = this_amp[sti:]
        this_phsr = this_phs[sti:]
        this_times = evodat.times[sti:]

        oscil_points = np.argwhere(np.abs(np.diff(this_amp)) > 0.001)
        for pt in oscil_points:
            try:
                pass
                #print(f"t = {evodat.times[pt-1]}, amp = {this_amp[pt-1]}, phs = {this_phs[pt-1]}, res = {dats[pt-1]}, norm = {evodat.norms[pt-1]}")
                #print(f"t = {evodat.times[pt]}, amp = {this_amp[pt]}, phs = {this_phs[pt]}, res = {dats[pt]}, norm = {evodat.norms[pt]}")
                #print(f"t = {evodat.times[pt+1]}, amp = {this_amp[pt+1]}, phs = {this_phs[pt+1]}, res = {dats[pt+1]}, norm = {evodat.norms[pt+1]}")
            except:
                pass

        ax1.scatter(this_times, this_ampr, s=1, c=f"C{2*j}", label=f"{j+1}SP Theo.")
        if plotre:
            ax2.scatter(this_times, this_phsr, s=1, c=f"C{2*j}", label=f"{j+1}SP Theo.")

    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(imname)
    ax1.set_title(imname)
    ax1.legend()

    if plotre:
        ax2.set_xlabel(r"$t$")
        ax2.set_ylabel(rename)
        ax2.set_title(rename)
        ax2.legend()

    plt.savefig(FName+sel_mode+"_div.pdf")
    plt.close()
    
def plot_pointG_1D_spcmp (model:HamModel, L, T, dT=0.1, force_evolve = False, output=print, ip = None, iprad = 1, ik = 0, addn="", doplots = True, deduct_average = True, nsp = 1, start_t = 1, precision=20, by_fit = False):

    Ham = model([L])
    Hamname = model.name+(("_"+addn) if addn!="" else "")
    FName = "{}_SC_L{}_T{}".format(Hamname, L, T)
    if ip is None:
        ip = int(L/2)
    else:
        FName += "_ip{}".format(ip)
        if ip < 0:
            ip = L+ip

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
        evodat = EvoDat1D.from_evolve(Ham.time_evolve, T, dT, init, FName, precision=precision)
        evodat.save()

    if doplots:
        evodat.plot_profile()

    dats = evodat.res[ip,0,:]
    amps = evodat.norms + np.log(np.abs(dats))
    angl = np.angle(dats)
    phas = np.unwrap(np.angle(dats))

    init = evodat.res[:,0,0]

    spf = model.SPFMMA()

    count = 0
    sp0 = None
    fitaddnow = 1
    fitbasis = []
    spfits = []
    for row in spf:
        if row[3]:
            z = row[0]
            E = row[1]
            Hpp = row[4]

            zl = z ** (ip - np.arange(L))
            tamp = np.sum(zl*init)
            thissp = np.log(tamp/z*np.sqrt(1j/(2*np.pi*Hpp))) - 1j*E*evodat.times - np.log(evodat.times + 1e-10)/2

            if by_fit:

                if count == 0:
                    sp0 = thissp
                fitbasis.append(np.exp(thissp-sp0*0))
                basisnow = np.hstack([vec[:,np.newaxis] for vec in fitbasis])
                sti = int(T/(evodat.times[1]-evodat.times[0]))
                vals = np.linalg.lstsq(basisnow[sti:,:], np.exp(amps+1j*phas-sp0*0)[sti:])[0]
                vals = scipy.optimize.minimize(lambda x:np.linalg.norm(np.log(basisnow[sti:,:]@(x[:count+1]+1j*x[count+1:]))-(amps[sti:]+1j*angl[sti:])), np.concatenate((np.real(vals),np.imag(vals)))).x
                vals = vals[:count+1] + 1j*vals[count+1:]
                spfits.append(sp0*0 + np.log(basisnow@vals))

            else:

                if count == 0:
                    sp0 = thissp
                    spfits.append(sp0)
                else:
                    fitaddnow += np.exp(thissp - sp0)
                    spfits.append(sp0 + np.log(fitaddnow))

            count += 1
            if count >= nsp:
                break

    sti = int((start_t-evodat.times[0])/(evodat.times[1]-evodat.times[0]))

    if doplots:
        _, (ax1r, ax2) = plt.subplots(1,2, figsize=(14, 6))
        ax1i = ax1r.twinx()

        if deduct_average:

            avg_slp_r = np.polyfit(evodat.times[sti:], amps[sti:], 1)
            avg_slp_i = np.polyfit(evodat.times[sti:], phas[sti:], 1)
            dct_r = np.poly1d(avg_slp_r)(evodat.times[sti:])
            dct_i = np.poly1d(avg_slp_i)(evodat.times[sti:])

        else:

            dct_r = 0
            dct_i = 0

        ax1r.plot(evodat.times[sti:], amps[sti:] - dct_r, color="C0", label = "Num - Re")
        ax1i.plot(evodat.times[sti:], phas[sti:] - dct_i, color="C1", label = "Num - Im")

        if nsp == 1:
            ax1r.plot(evodat.times[sti:], fitr[sti:] - dct_r, color=f"C3", label = "SP - Re")
            ax1i.plot(evodat.times[sti:], fiti[sti:] - dct_i, color=f"C4", label = "SP - Im")
            ax2.plot(evodat.times[sti:], np.abs(fitr[sti:]-amps[sti:]), color=f"C3", label = f"SP - Re")
            ax2.plot(evodat.times[sti:], np.abs(fiti[sti:]-phas[sti:]), color=f"C4", label = f"SP - Im")

        else:

            for i in range(nsp):
                fitfun = spfits[i] 
                fitr = np.real(fitfun)
                fiti = np.unwrap(np.imag(fitfun))
                fiti += 2*np.pi*np.round(np.average(phas-fiti)/(2*np.pi))
                ax1r.plot(evodat.times[sti:], fitr[sti:] - dct_r, color=f"C{2*i+2}", label = f"{i+1} SP - Re")
                ax1i.plot(evodat.times[sti:], fiti[sti:] - dct_i, color=f"C{2*i+3}", label = f"{i+1} SP - Im")
                ax2.plot(evodat.times[sti:], np.abs(fitr[sti:]-amps[sti:]), color=f"C{2*i+2}", label = f"{i+1} SP - Re")
                ax2.plot(evodat.times[sti:], np.abs(fiti[sti:]-phas[sti:]), color=f"C{2*i+3}", label = f"{i+1} SP - Im")

        ax2.set_xscale("log")
        ax2.set_yscale("log")

        ax1r.set_xlabel("t")
        ax1r.set_ylabel("Re[log[G(0,0,t)]]")
        ax1i.set_ylabel("Im[log[G(0,0,t)]]")
        ax1r.set_title("log[G], deducted by average slope")
        ax2.set_xlabel("t")
        ax2.set_ylabel("Error")
        ax2.set_title("Numerical v.s. SP Error")

        h1,l1 = ax1r.get_legend_handles_labels()
        h2,l2 = ax1i.get_legend_handles_labels()
        ax1r.legend(h1+h2, l1+l2, loc="lower right")
        ax2.legend()

        plt.savefig(FName+f"_{nsp}sp.jpg")

    return amps, phas, spfits

def plot_pointG_2D (model:HamModel, L, W, T, dT=0.1, force_evolve = False, output=print, ip = None, iprad = 0.1, ik = (0,0), comp=True, addn="", doplots=True, plotre=True, alt=None, sel_mode="avg", start_t = 5, orig_t = 2, error_log = False):

    Ham = model([L,W])
    Hamname = model.name+(("_"+addn) if addn!="" else "")
    FName = "{}_PG_L{}_T{}".format(Hamname, L, T)
    if ip is None:
        ip = (int(L/2), int(W/2))
    else:
        FName += "_ip{}".format(ip)
        ip = (ip[0] if ip[0]>=0 else L+ip[0], ip[1] if ip[1]>=0 else W+ip[1])

    if model.int_dim == 1:
        intvec = [1]
    else:
        intvec = np.random.randn(model.int_dim) + 1j*np.random.randn(model.int_dim)
        intvec = intvec / np.linalg.norm(intvec)

    init = GaussianWave([L,W], ip, iprad, k = ik, intvec = intvec)

    try:
        if force_evolve:
            raise Exception()
        evodat = EvoDat2D.read(FName)
    except Exception:
        evodat = EvoDat2D.from_evolve(model, L, W, T, dT, init, FName)
        evodat.save()
        if doplots:
            evodat.animate()

    dats = evodat.res[*ip,0,:]
    print(dats[-1]*np.exp(evodat.norms[-1]))
    print(evodat.times[-1])
    amps = evodat.norms + np.log(np.abs(dats)) + np.log(evodat.times + 1e-10)
    phas = np.unwrap(np.angle(dats))

    sti = int(start_t / (evodat.times[1]-evodat.times[0]))
    oti = int(orig_t / (evodat.times[1]-evodat.times[0]))

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
        if sti <= oti:
            sti = oti + 1
        ampr = (amps[sti:]-amps[oti:-(sti-oti)]) / (evodat.times[sti]-evodat.times[oti])
        phsr = - (phas[sti:]-phas[oti:-(sti-oti)]) / (evodat.times[sti]-evodat.times[oti])
        times = evodat.times[sti:]
    else:
        raise Exception(f"Invalid argument sel_mode={sel_mode}")

    E = None

    if comp:

        spf = model.SPFMMA()
        for row in spf:
            if int(row[3]) != 0:
                E = row[1]
                break

    if sel_mode == "inst":
        rename = r"Re[i d/dt[log(G(0,0,t)*t)]]"
        imname = r"Im[i d/dt[log(G(0,0,t)*t)]]"
    else:
        rename = r"Re[i log(G(0,0,t)*t)/t]"
        imname = r"Im[i log(G(0,0,t)*t)/t]"

    if doplots:

        lw = 1
        plt.rcParams.update({"font.size":8, "lines.linewidth":lw, "mathtext.fontset":"cm"})

        if plotre:
            _, (ax1,ax2) = plt.subplots(1, 2, figsize=(5, 2.2), layout="constrained")
        else:
            plt.figure(figsize=(3.375, 2.4))
            ax1 = plt.gca()

        if comp and error_log:

            ax1.plot(times, np.abs(ampr-np.imag(E)), color="C0", label="Error Im")
            if plotre:
                ax1.plot(times, np.abs(phsr-np.real(E)), color="C1", label="Error Re")
            ax1.set_yscale("log")
            ax1.set_xscale("log")
            ax1.set_ylim([max(1e-5, ax1.get_ylim()[0]), ax1.get_ylim()[1]])
            ax1.set_xlabel("t")
            ax1.set_ylabel(imname[3:-1])

        else:

            ax1.plot(times, ampr, color="C0", label=imname)
            if plotre:
                ax2.plot(times, phsr, color="C1", label=rename)
            
            if comp:
                ax1.plot([times[0], times[-1]], [np.imag(E)]*2, color="C2", linestyle="--", label=r"Im$\epsilon^\ast$")
                if alt is not None:
                    ax1.plot([times[0], times[-1]], [alt]*2, color="C4", linestyle="--", label=r"Im$\epsilon^\ast$ alt.theo.")
                if plotre:
                    ax2.plot([times[0], times[-1]], [np.real(E)]*2, color="C3", linestyle="--", label=r"Re$\epsilon^\ast$")

            ax1.set_xlabel("t")
            ax1.set_ylabel(imname)
            ax1.legend()

            if plotre:
                ax2.set_xlabel("t")
                ax2.set_ylabel(rename)
                ax2.legend()

        plt.savefig(FName+sel_mode+".jpg")

    if comp:
        return ampr, phsr, E
    else:
        return ampr, phsr
    
def plot_pointG_2D_mod (model:HamModel, L, W, T, dT=0.1, force_evolve = False, ip = None, iprad = 1, ik = (0,0), addn="", doplots=True, plotre=True, ploterr = False, sel_mode="avg", start_t = 5, orig_t = 2, precision = 0):

    Ham = model([L])
    Hamname = model.name+(("_"+addn) if addn!="" else "")
    FName = "{}_FG_L{}_T{}".format(Hamname, L, T)
    if ip is None:
        ip = (int(L/2), int(W/2))
    else:
        FName += "_ip{}".format(ip)
        ip = (ip[0] if ip[0]>=0 else L+ip[0], ip[1] if ip[1]>=0 else W+ip[1])

    if model.int_dim == 1:
        intvec = [1]
    else:
        intvec = np.random.randn(model.int_dim) + 1j*np.random.randn(model.int_dim)
        intvec = intvec / np.linalg.norm(intvec)

    init = GaussianWave([L,W], ip, iprad, k=ik, intvec=intvec)

    try:
        if force_evolve:
            raise Exception()
        evodat = EvoDat2D.read(FName)
    except Exception:
        evodat = EvoDat2D.from_evolve(model, L, W, T, dT, init, FName)
        evodat.save()
        if doplots:
            evodat.animate()

    dats = evodat.res[*ip,0,:]
    amps = evodat.norms + np.log(np.abs(dats)) + np.log(evodat.times + 1e-10)
    phas = np.unwrap(np.angle(dats))

    spf = model.SPFMMA()
    E = None

    for row in spf:
        if int(row[3]) != 0:
            E = row[1]
            break

    sti = int(start_t / (evodat.times[1]-evodat.times[0]))

    def proc_data (amps, phas):

        if sel_mode == "avg":
            oti = int(orig_t / (evodat.times[1]-evodat.times[0]))
            ampr = (amps[sti:]-amps[oti]) / (evodat.times[sti:]-evodat.times[oti])
            phsr = - (phas[sti:]-phas[oti]) / (evodat.times[sti:]-evodat.times[oti])
            times = evodat.times[sti:]
        elif sel_mode == "half":
            inds = np.arange(int(sti/2), floor(len(amps)/2))
            ampr = (amps[2*inds]-amps[inds]) / (evodat.times[2*inds]-evodat.times[inds])
            phsr = - (phas[2*inds]-phas[inds]) / (evodat.times[2*inds]-evodat.times[inds])
            times = evodat.times[2*inds]
        elif sel_mode == "inst":
            ampr = (amps[sti:]-amps[:-sti]) / (evodat.times[sti]-evodat.times[0])
            phsr = - (phas[sti:]-phas[:-sti]) / (evodat.times[sti]-evodat.times[0])
            times = evodat.times[sti:]
        else:
            raise Exception(f"Invalid argument sel_mode={sel_mode}")
        
        return ampr, phsr, times
    
    ampr, phsr, times = proc_data(amps, phas)

    if sel_mode == "inst":
        rename = r"$\mathrm{Re}[i \frac{\mathrm{d}}{\mathrm{d}t}[\log(t\cdot G(0,0,t))]]$"
        imname = r"$\mathrm{Im}[i \frac{\mathrm{d}}{\mathrm{d}t}[\log(t\cdot G(0,0,t))]]$"
    else:
        rename = r"$\mathrm{Re}[i \log(t\cdot G(0,0,t))/t]$"
        imname = r"$\mathrm{Im}[i \log(t\cdot G(0,0,t))/t]$"

    lw = 1

    plt.rcParams.update({"font.size":8, "lines.linewidth":lw, "mathtext.fontset":"cm"})

    # Generate the plots
    total_plots = 1 + plotre + ploterr

    if total_plots == 1:
        plt.figure(figsize=(3.375, 2.4), layout="constrained")
        axim = plt.gca()
    else:
        if total_plots == 2:
            _, axs = plt.subplots(1, 2, figsize=(6, 2.4), layout="constrained")
        elif total_plots == 3:
            _, axs = plt.subplots(1, 3, figsize=(7, 2.4), layout="constrained")

        axim = axs[0]
        if plotre:
            axre = axs[1]
        if ploterr:
            axerr = axs[1+plotre]

    # Plot real part and imag part

    axim.plot(times, ampr, color="C0", linewidth=lw, label="Numerical")
    if plotre:
        axre.plot(times, phsr, color="C1", linewidth=lw, label="Numerical")

    this_amp = np.imag(E)*evodat.times
    this_phs = -np.real(E)*evodat.times
    this_ampr, this_phsr, this_times = proc_data(this_amp, this_phs)

    axim.plot(this_times, this_ampr, color="C2", linewidth=lw, linestyle="--", label="SP Theo.")
    if plotre:
        axre.plot(this_times, this_phsr, color="C3", linewidth=lw, linestyle="--", label="SP Theo.")
    if ploterr:
        axerr.plot(times, np.abs(ampr-this_ampr), color="C2", linewidth=lw, label="Im")
        axerr.plot(times, np.abs(phsr-this_phsr), color="C3", linewidth=lw, label="Re")

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

    plt.savefig(FName+sel_mode+".pdf")
    plt.close()

def plot_pointG_2D_tslope (model:HamModel, L, T, dT=0.1, force_evolve = False, output=print, ip = None, iprad = 1, ik = 0, addn="", doplots=True, alt=None, sel_mode="avg", start_t = 5, orig_t = 2, error_log = False, precision = 0):

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
        evodat = EvoDat1D.from_evolve(Ham.time_evolve, T, dT, init, FName, precision=precision, return_idnorm=False)
        evodat.save()

    if doplots:
        evodat.plot_profile()

    dats = evodat.res[ip,0,:]
    amps = evodat.norms + np.log(np.abs(dats))

    init = evodat.res[:,0,0]

    spf = model.SPFMMA()

    for row in spf:
        if row[3]:
            z = row[0]
            E = row[1]
            Hpp = row[4]

            if which_boundary == -1:
                xmin = max(0, ip-5*iprad)
                xmax = min(L-1, ip+5*iprad)
                zl = model.GFDMMA(E, *[(x+1,ip+1) for x in range(xmin, xmax+1)])
                tamp = np.sum(zl * (init[xmin:xmax+1]))
            elif which_boundary == 1:
                xmin = max(0, ip-5*iprad)
                xmax = min(L-1, ip+5*iprad)
                zl = model.GFDMMA(E, *[(x-L,ip-L) for x in range(xmin, xmax+1)])
                tamp = np.sum(zl * (init[xmin:xmax+1]))
            else:
                zl = z ** (ip - np.arange(L))
                tamp = np.sum(zl*init)

            print(tamp/z*np.sqrt(1/(2*np.pi))*Hpp/1j)
            print(np.log(tamp/z*np.sqrt(1/(2*np.pi))*Hpp/1j))
            print(tamp/z*np.sqrt(1/(4*np.pi))*Hpp)
            print(np.log(tamp/z*np.sqrt(1/(4*np.pi))*Hpp))

            if which_boundary == 0:
                thissp = np.log(tamp/z*np.sqrt(1/(2*np.pi))*Hpp/1j)
            else:
                thissp = np.log(tamp/z*np.sqrt(1/(4*np.pi))*Hpp)

            break

    amps = amps - np.imag(E)*(evodat.times - evodat.times[0])

    sti = int(start_t / (evodat.times[1]-evodat.times[0]))

    times = evodat.times[sti:]
    amps = amps[sti:]

    lw = 1

    plt.rcParams.update({"font.size":8, "lines.linewidth":lw, "mathtext.fontset":"cm"})

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.4), layout="constrained")

    logtimes = np.log(times)
    ax1.plot(logtimes, amps)
    poly = np.polyfit(logtimes, amps, 1)
    ax1.set_title("Slope = {:.3f}, Intersect = {:.3f}".format(poly[0], poly[1]))
    ax1.plot(logtimes, np.poly1d(poly)(logtimes), color="C1", linestyle="--")

    ax2.set_title("Theo. Ints. = {:.3f}".format(np.real(thissp)))
    ax2.plot(logtimes, np.gradient(amps, logtimes))

    plt.savefig(FName+sel_mode+"_texp.pdf")
    plt.close()

def plot_evolution_1D (model:HamModel, L, T, Tplot=None, ip = None, ik = None, intvec = None, kspan = 0.1, dT=0.2, force_evolve = False, return_cp_when_possible = False, fpcut = 1e-10):

    FName = "{}_L{}_T{}".format(model.name, L, T)

    if ip is None:
        ip = int(L/2)
    else:
        FName += "_ip{}".format(ip)

    if ik is None:
        ik = 0
    else:
        FName += "_ik{}".format(ik)

    FName += "_k{}".format(kspan)

    if intvec is not None:
        FName += "_iv{}".format(intvec)
    
    init = GaussianWave([L], (ip,), 1/(2*kspan), (ik,), intvec)

    Ham = model([L])

    try:
        if force_evolve:
            raise Exception()
        evodat = EvoDat1D.read(FName)
    except Exception:
        evodat = EvoDat1D.from_evolve(Ham.time_evolve, T, dT, init, FName, fpcut = fpcut)
        evodat.save()
    evodat.plot_normNgrowth()
    #evodat.plot_profile(logcut=0 if fpcut==0 else -np.log10(fpcut))
    evodat.plot_profile()

    ncp = np #cp if USE_GPU else np

    Xop = ncp.arange(L)
    xs = Xop
    KX = xs * 2*ncp.pi/L - ncp.pi

    # The Fourier transform matrix
    FT = ncp.exp(-1j*(KX[:,ncp.newaxis]*Xop[ncp.newaxis,:])) / ncp.sqrt(L)
    
    if Tplot is None:
        Tplot = dT

    res = evodat.res

    amp = res.conj() * res
    ivs = ncp.sum(amp, axis=0)
    amp_real = ncp.sum(amp, axis=1)
    del amp
    xs = ncp.real(ncp.sum(Xop[:,ncp.newaxis] * amp_real, axis=0))
    del amp_real

    res = ncp.tensordot(FT, res, axes=1)
    amp_k = ncp.sum(res.conj()*res, axis=1)
    del res

    ks = ncp.real(ncp.sum(KX[:,ncp.newaxis] * amp_k, axis=0))
    delk = (ncp.fmod(KX[:,ncp.newaxis]+ncp.pi-ks[ncp.newaxis,:], 2*ncp.pi)-ncp.pi)**2
    vk = ncp.real(ncp.sum(delk * amp_k, axis=0))

    try:
        f = open("WP_"+FName+".txt", 'w+')
        f.write("Hamiltonian\n")
        f.write(model.to_mathematica())
        f.write("\n\nt\n")
        writearray(f.write, evodat.times)
        f.write("x\n")
        writearray(f.write, xs)
        f.write("k\n")
        writearray(f.write, ks)
        f.write("var_k\n")
        writearray(f.write, vk)
        f.write("iv\n")
        writearray(f.write, ivs)
        f.close()
    except:
        if force_evolve:
            print("File f already exists! Cannot write")

    #if USE_GPU and not return_cp_when_possible:
    #    return evodat.times.get(), xs.get(), ks.get(), vk.get(), ivs.get(), FName
    #else:
    return evodat.times, xs, ks, vk, ivs, FName

def fit_accerlation_1D (model:HamModel, L, T, ip = None, ik = None, intvec = None, kspan = 0.1, dT=0.2, Twindow = 0.5, force_evolve = False, compare = False, fpcut = 1e-10):
    
    ts, xs, ks, vk, _, name = plot_evolution_1D(model, L, T, ip=ip, ik=ik, intvec=intvec, kspan=kspan, dT=dT, force_evolve=force_evolve, fpcut=fpcut)

    vs = np.gradient(xs, ts)
    #vs = [(xs[i]-xs[0])/(ts[i]-ts[0]) for i in range(1, len(ts))]
    #vs = [vs[0]]+vs
    kvs = np.gradient(ks, ts) / vk

    vs = gaussfilt(vs, int(Twindow/dT))
    kvs = gaussfilt(kvs, int(Twindow/dT))
    _, (ax1,ax2) = plt.subplots(2,1, figsize=(8,10), gridspec_kw={'height_ratios':[2,1]})
    ax1.scatter(ks, vs, label="v", color="C0")
    ax1.set_xlabel("k")
    ax1.set_ylabel("v")

    ax3 = ax1.twinx()
    ax3.scatter(ks, kvs, label="dk/dt", color="C1")
    ax3.set_ylabel("dk/dt")

    if compare:

        precision = 100

        def unlink_wrap(dat, thresh = 10):
            jump = np.nonzero(np.abs(np.diff(dat)) > thresh*2*np.pi/precision)[0]
            lasti = 0
            for ind in jump:
                yield slice(lasti, ind + 1)
                lasti = ind + 1
            yield slice(lasti, len(dat))

        Hs = model.EnergiesMMA(prec=precision, krange=(np.min(ks), np.max(ks)))
        pks = [(np.max(ks)-np.min(ks))*i/precision+np.min(ks) for i in range(precision+1)]
        #Hcuts = unlink_wrap(Hs)
        #for cut in Hcuts:
            #tHs = Hs[cut]
            #tks = pks[cut]
            #if len(tHs) > 1:

        tHs = Hs
        tks = pks

        grads = np.gradient(tHs, tks)

        v1k = np.interp(ks, tks, np.real(grads))
        v2k = np.interp(ks, tks, np.real(grads)*np.imag(grads)/2)
        ax1.plot(ks, v1k+v2k*vk, label="v theo", color="C0")
        #ax1.plot(tks, np.real(grads), label="v theo", color="C0")
        ax3.plot(tks, 2*np.imag(grads), label="dk/dt theo", color="C1")

    h1,l1 = ax1.get_legend_handles_labels()
    h2,l2 = ax3.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper right")

    plt.title(f"Velocity and Acceleration fit")
    ax2.plot(ks, ts)
    ax2.set_ylabel("t")
    ax2.set_xlabel("k")
    plt.savefig(f"Fit_{name}.jpg")
    plt.close()

"""
def plot_and_fit_2D_Edge (HamF, Hamname, L, W, T, Tplot=None, edge="x-", k = (0,0), dT=0.2, force_evolve = False, doplots = True):

    tkdT = 0.25

    Ham = HamF(L,W)
    if edge == "x-":
        ip = (1, int(W/2))
    elif edge == "x+":
        ip = (L-2, int(W/2))
    elif edge == "y-":
        ip = (int(L/2), 1)
    elif edge == "y+":
        ip = (int(L/2), W-2)
    else:
        raise Exception("ip must be one of: 'x-', 'x+', 'y-', 'y+'.")
    
    init = Ham.GaussianWave(ip, 1, k)
    
    FName = "{}__{}L{}W{}_T{}".format(Hamname, edge, L, W, T)
    if k != (0,0):
        FName += "_k{}".format(k)
    try:
        if force_evolve:
            raise Exception()
        evodat = EvoDat2D.read(FName+".txt")
    except Exception as e:
        evodat = EvoDat2D.from_evolve(Ham, L, W, T, dT, init, FName, takedT=tkdT)
        evodat.save()
    if doplots:
        evodat.animate()
        evodat.plot_normNgrowth()

    if edge == "x-":
        seldat = evodat.res[1,:,:,:]
        vs = (np.arange(W)-int(W/2))/T
    elif edge == "x+":
        seldat = evodat.res[L-2,:,:,:]
        vs = (np.arange(W)-int(W/2))/T
    elif edge == "y-":
        seldat = evodat.res[:,1,:,:]
        vs = (np.arange(L)-int(L/2))/T
    elif edge == "y+":
        seldat = evodat.res[:,W-2,:,:]
        vs = (np.arange(L)-int(L/2))/T
    else:
        raise Exception("ip must be one of: 'x-', 'x+', 'y-', 'y+'.")
    

    seldat = np.linalg.norm(seldat,axis=1)
    Tn = np.shape(seldat)[1]
    
    if Tplot is not None:
        lastind = int(Tplot/tkdT)
        midind = int(Tplot/(2*tkdT))
    else:
        lastind = Tn-1
        midind = int((Tn-1)/2)

    if edge.startswith("x"):
        ip = ip[1]
    else:
        ip = ip[0]

    lastprof = seldat[::2,lastind]
    midprof = seldat[int(ip/2):int(ip/2)+len(lastprof),midind]
    lyaps = (np.log(lastprof)-np.log(midprof)+evodat.norms[lastind]-evodat.norms[midind]+np.log(2))/(T/2)
    vs = (2*np.array([i for i in range(len(lastprof))])-ip)/T

    if doplots:
    
        plt.figure()
        
        plt.plot(vs, lyaps)
        plt.xlabel("v")
        plt.ylabel("lambda")
        plt.title("Lyapunov exponents fitted on edge {} at T={}".format(edge, T))
        plt.savefig(FName+".jpg")
        plt.close()
"""

def plot_velocities_2D (model:HamModel, L, W, T, Tplot=None, ip = None, dT=0.2, force_evolve = False, comp = True, prec = 5):

    tkdT = 0.25

    FName = "{}_L{}W{}_T{}".format(model.name, L, W, T)

    if ip is None:
        ip = (int(L/2), int(W/2))
    else:
        FName += "_{}".format(ip)
    
    init = GaussianWave([L,W], ip, 1)

    try:
        if force_evolve:
            raise Exception()
        evodat = EvoDat2D.read(FName+".txt")
    except Exception as e:
        evodat = EvoDat2D.from_evolve(model, L, W, T, dT, init, FName, takedT=tkdT)
        evodat.save()
    evodat.animate()
    evodat.plot_normNgrowth()

    seldat = np.linalg.norm(evodat.res,axis=2)
    Tn = np.shape(seldat)[-1]
    
    if Tplot is not None:
        lastind = int(Tplot/tkdT)
        midind = int(Tplot/(2*tkdT))
    else:
        lastind = Tn-1
        midind = int((Tn-1)/2)

    lastprof = seldat[::2,::2,lastind]
    datsp = np.shape(lastprof)
    midprof = seldat[int(ip[0]/2):int(ip[0]/2)+datsp[0], int(ip[1]/2):int(ip[1]/2)+datsp[1], midind]
    lyaps = (np.log(lastprof)-np.log(midprof)+evodat.norms[lastind]-evodat.norms[midind]+np.log(2))/(T/2)

    vxs = (2*np.array([i for i in range(datsp[0])])-ip[0])/T
    vys = (2*np.array([i for i in range(datsp[1])])-ip[1])/T

    if comp:
        growths = model.GrowthsMMA(vxs[0], vxs[-1], vys[0], vys[-1], prec=prec)

    try:
        f = open(FName+".txt", 'w+')
        f.write("vxs\n")
        writearray(f.write, vxs)
        f.write("vys\n")
        writearray(f.write, vys)
        f.write("numerical")
        writearray(f.write, lyaps)
        if comp:
            f.write("theoretical")
            writearray(f.write, growths)
        f.close()
    except:
        if force_evolve:
            print("File f already exists! Cannot write")
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    X,Y = np.meshgrid(vxs, vys, indexing="ij")
    ax.plot_surface(X, Y, lyaps, color="C0")

    if comp:
        nvxs = np.array([(vxs[0]*(prec-i)+vxs[-1]*i)/prec for i in range(prec+1) for _ in range(prec+1)])
        nvys = np.array([(vys[0]*(prec-i)+vys[-1]*i)/prec for i in range(prec+1)]*(prec+1))
        ax.scatter(nvxs, nvys, growths, c="C1", s=10)

    ax.set_xlabel("vx")
    ax.set_ylabel("vy")
    ax.set_zlabel("lambda")

    plt.title("Lyapunov exponents fitted at T={}".format(T))
    pickle.dump(fig, open(FName + '.pickle', 'wb'))
    plt.savefig(FName+".jpg", format="jpg")
    plt.savefig(FName+".pdf", format="pdf")
    plt.show()
    plt.close()

def plot_evolution_2D (model:HamModel, L, W, T, Tplot=None, ip = None, cut = None, ik = None, intvec = None, kspan = 0.1, dT=0.2, force_evolve = False, return_cp_when_possible = False):

    tkdT = 0.25

    FName = "{}_L{}W{}_T{}".format(model.name, L, W, T)

    if ip is None:
        ip = (int(L/2), int(W/2))
    else:
        FName += "_ip{}".format(ip)

    if ik is None:
        ik = (0, 0)
    else:
        FName += "_ik{}".format(ik)

    FName += "_ks{}".format(kspan)

    if intvec is not None:
        FName += "_iv{}".format(intvec)
    
    init = GaussianWave([L,W], ip, 1/(2*kspan), ik, intvec)

    try:
        if force_evolve:
            raise Exception()
        evodat = EvoDat2D.read(FName+".txt", return_cp_when_possible = True)
    except Exception:
        evodat = EvoDat2D.from_evolve(model, L, W, T, dT, init, FName, takedT=tkdT, return_cp_when_possible = True)
        evodat.save()
    evodat.animate()
    if cut is not None:
        evodat.animate(cuts=cut)
    evodat.plot_normNgrowth()

    ncp = cp if USE_GPU else np

    if cut is not None:
        cutx, cuty = cut
    else:
        cutx = slice(L)
        cuty = slice(W)

    xs = ncp.arange(L)[cutx]
    ys = ncp.arange(W)[cuty]
    Xop, Yop = ncp.meshgrid(xs, ys, indexing="ij")
    Xop = Xop.flatten()
    Yop = Yop.flatten()

    Lc = len(xs)
    Wc = len(ys)

    kx = (xs-xs[0]) * 2*ncp.pi/Lc - ncp.pi
    ky = (ys-ys[0]) * 2*ncp.pi/Wc - ncp.pi
    KX, KY = ncp.meshgrid(kx, ky, indexing="ij")
    KX = KX.flatten()
    KY = KY.flatten()

    # The Fourier transform matrix
    FT = ncp.exp(-1j*(KX[:,ncp.newaxis]*Xop[ncp.newaxis,:] + KY[:,ncp.newaxis]*Yop[ncp.newaxis,:])) / ncp.sqrt(Lc*Wc)
    #FTd = FT.conj().T

    #Xop = ncp.diag(Xop)
    #Yop = ncp.diag(Yop)

    #KXop = ncp.diag(KX)
    #KYop = ncp.diag(KY)

    #KXop = FTd @ ncp.diag(KX) @ FT
    #KYop = FTd @ ncp.diag(KY) @ FT
    
    if Tplot is None:
        Tplot = dT

    res = evodat.res[cutx, cuty, :, :]
    res = res.reshape((Lc*Wc, ncp.shape(res)[2], -1))

    amp = res.conj() * res
    ivs = ncp.sum(amp, axis=0)
    amp_real = ncp.sum(amp, axis=1)
    norms = ncp.real(ncp.sum(amp_real, axis=0))
    del amp

    xs = ncp.real(ncp.sum(Xop[:,ncp.newaxis] * amp_real, axis=0)) / norms
    ys = ncp.real(ncp.sum(Yop[:,ncp.newaxis] * amp_real, axis=0)) / norms

    del amp_real

    #xs = ncp.real(ncp.diag(ncon([res.conj(), Xop, res], [[1,3,-1],[1,2],[2,3,-2]])))
    #ys = ncp.real(ncp.diag(ncon([res.conj(), Yop, res], [[1,3,-1],[1,2],[2,3,-2]])))

    res = ncp.tensordot(FT, res, axes=1)
    amp_k = ncp.sum(res.conj()*res, axis=1)
    del res

    kxs = ncp.real(ncp.sum(KX[:,ncp.newaxis] * amp_k, axis=0)) / norms
    kys = ncp.real(ncp.sum(KY[:,ncp.newaxis] * amp_k, axis=0)) / norms

    evodat.animate_custom(plotdatas=kxs.get() if USE_GPU and isinstance(kxs, cp.ndarray) else kxs, plotname="kx", reference_levels=[np.arccos(0.5), -np.arccos(0.5)])

    #kxs = ncp.real(ncp.diag(ncon([res.conj(), KXop, res], [[1,3,-1],[1,2],[2,3,-2]])))
    #kys = ncp.real(ncp.diag(ncon([res.conj(), KYop, res], [[1,3,-1],[1,2],[2,3,-2]])))

    # To calculate the standard deviation of k, we construct an operator k-<k> (shifted to be centered at zero)
    delkx = (ncp.fmod(kx[:,ncp.newaxis]+ncp.pi-kxs[ncp.newaxis,:], 2*ncp.pi)-ncp.pi)**2
    delky = (ncp.fmod(ky[:,ncp.newaxis]+ncp.pi-kys[ncp.newaxis,:], 2*ncp.pi)-ncp.pi)**2
    delkx = ncp.repeat(delkx[:,ncp.newaxis,:], Wc, 1).reshape((Lc*Wc, len(kxs))) / norms
    delky = ncp.repeat(delky[ncp.newaxis,:,:], Lc, 0).reshape((Lc*Wc, len(kxs))) / norms

    #inds = ncp.arange(L*W)
    #indt = ncp.arange(len(kxs))
    #DKX = ncp.zeros((len(kxs),L*W,L*W), dtype=complex)
    #DKY = ncp.zeros((len(kxs),L*W,L*W), dtype=complex)
    #DKX[indt, inds[:,ncp.newaxis], inds[:,ncp.newaxis]] = delkx
    #DKY[indt, inds[:,ncp.newaxis], inds[:,ncp.newaxis]] = delky
    #DKX = FTd @ DKX @ FT
    #DKY = FTd @ DKY @ FT

    vkx = ncp.real(ncp.sum(delkx * amp_k, axis=0))
    vky = ncp.real(ncp.sum(delky * amp_k, axis=0))

    #vkx = ncp.real(ncon([res.conj(), DKX, res], [[1,3,-1],[-3,1,2],[2,3,-2]])[indt,indt,indt])
    #vky = ncp.real(ncon([res.conj(), DKY, res], [[1,3,-1],[-3,1,2],[2,3,-2]])[indt,indt,indt])
    #ivs = ncp.sum(amp_k, axis=0)

    try:
        f = open("WP_"+FName+f"{cut if cut is not None else ''}.txt", 'w+')
        f.write("Hamiltonian\n")
        f.write(model.to_mathematica())
        f.write("\n\nt\n")
        writearray(f.write, evodat.times)
        f.write("x\n")
        writearray(f.write, xs)
        f.write("y\n")
        writearray(f.write, ys)
        f.write("kx\n")
        writearray(f.write, kxs)
        f.write("ky\n")
        writearray(f.write, kys)
        f.write("var_kx\n")
        writearray(f.write, vkx)
        f.write("var_ky\n")
        writearray(f.write, vky)
        f.write("iv\n")
        writearray(f.write, ivs)
        f.close()
    except:
        if force_evolve:
            print("File f already exists! Cannot write")

    if USE_GPU and not return_cp_when_possible:
        return evodat.times.get(), xs.get(), ys.get(), kxs.get(), kys.get(), vkx.get(), vky.get(), ivs.get(), FName
    else:
        return evodat.times, xs, ys, kxs, kys, vkx, vky, ivs, FName

def fit_accerlation_2D_edge (model:HamModel, L, W, T, plot_axis = 0, which_edge = 0, ip = None, ik = None, intvec = None, kspan = 0.1, dT=0.2, Twindow = 0.5, force_evolve = False, compare = False, cut_depth = 5):
    
    if plot_axis == 0:
        ik2 = (ik,0)
        if which_edge == 0:
            ip2 = (ip if ip is not None else int(L/2), 0)
            cut = (slice(L), slice(cut_depth))
        else:
            ip2 = (ip if ip is not None else int(L/2), W-1)
            cut = (slice(L), slice(W-cut_depth,W))
        ksp = np.array([kspan, 10])
    else:
        ik2 = (0,ik)
        if which_edge == 0:
            ip2 = (0, ip if ip is not None else int(W/2))
            cut = (slice(cut_depth), slice(W))
        else:
            ip2 = (L-1, ip if ip is not None else int(W/2))
            cut = (slice(L-cut_depth,L), slice(W))
        ksp = np.array([10, kspan])

    ts, xs, ys, kxs, kys, vkx, vky, _, name = plot_evolution_2D(model, L, W, T, ip=ip2, ik=ik2, intvec=intvec, kspan=ksp, dT=dT, force_evolve=force_evolve, cut=cut)

    if plot_axis == 0:
        spaces = xs
        #vs = np.gradient(xs, ts)
        kvs = np.gradient(kxs, ts) / vkx
        ks = kxs
    else:
        spaces = ys
        #vs = np.gradient(ys, ts)
        kvs = np.gradient(kys, ts) / vky
        ks = kys

    vs = [(spaces[i]-spaces[0])/(ts[i]-ts[0]) for i in range(1, len(ts))]
    vs = [vs[0]]+vs

    vs = gaussfilt(vs, int(Twindow/dT))
    kvs = gaussfilt(kvs, int(Twindow/dT))
    _, (ax1,ax2) = plt.subplots(2,1, figsize=(8,10), gridspec_kw={'height_ratios':[2,1]})
    ax1.scatter(ks, vs, label="v", color="C0")
    ax1.set_xlabel("k")
    ax1.set_ylabel("v")

    ax3 = ax1.twinx()
    ax3.scatter(ks, kvs, label="dk/dt", color="C1")
    ax3.set_ylabel("dk/dt")

    if compare:

        precision = 50

        def unlink_wrap(dat, thresh = 10):
            jump = np.nonzero(np.abs(np.diff(dat)) > thresh*2*np.pi/precision)[0]
            lasti = 0
            for ind in jump:
                yield slice(lasti, ind + 1)
                lasti = ind + 1
            yield slice(lasti, len(dat))

        Hs = model.EnergiesMMA(edge=plot_axis+1, prec=precision, krange=(np.min(ks), np.max(ks)))

        pks = [(np.max(ks)-np.min(ks))*i/precision+np.min(ks) for i in range(precision+1)]
        Hcuts = unlink_wrap(Hs)
        haslabels = False
        for cut in Hcuts:
            tHs = Hs[cut]
            tks = pks[cut]
            if len(tHs) > 1:
                grads = np.gradient(tHs, tks)
                if not haslabels:
                    ax1.plot(tks, np.real(grads), label="v theo", color="C0")
                    ax3.plot(tks, 2*np.imag(grads), label="dk/dt theo", color="C1")
                    haslabels = True
                else:
                    ax1.plot(tks, np.real(grads), color="C0")
                    ax3.plot(tks, 2*np.imag(grads), color="C1")

    h1,l1 = ax1.get_legend_handles_labels()
    h2,l2 = ax3.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper right")

    plt.title(f"Velocity and Acceleration fit in {'x' if plot_axis==0 else 'y'} direction")
    ax2.plot(ks, ts)
    ax2.set_ylabel("t")
    ax2.set_xlabel("k")
    plt.savefig(f"Fit{'X' if plot_axis==0 else 'Y'}_{name}.jpg")
    plt.close()


def writearray (write, arr):
    if len(np.shape(arr)) == 1:
        for ele in arr:
            write("{}\t".format(ele))
        write("\n")
    elif len(np.shape(arr)) == 2:
        for row in range(np.shape(arr)[0]):
            writearray(write, arr[row,:])

def plot_and_fit_2D (model:HamModel, L, W, T, init, dT=0.01, force_evolve = False, VKG = None, addname = ""):

    FName = "{}_L{}W{}_T{:.1f}_{}".format(model.name, L, W, T, addname)

    try:
        if force_evolve:
            raise Exception()
        evodat = EvoDat2D.read(FName+".txt")
    except Exception as e:
        evodat = EvoDat2D.from_evolve(model, L, W, T, dT, init, FName, takedT=1)
        evodat.save()
    evodat.animate()
    evodat.plot_profile(0)
    evodat.plot_profile(1)
    evodat.plot_normNgrowth()

def plot_and_fit_2D_Edge (model:HamModel, L, W, T, Tplot=None, edge="x-", k = (0,0), dT=0.2, force_evolve = False, addname = "", doplots = True, comp_prec = 0):

    tkdT = 0.25

    FName = "{}_L{}W{}_{}_T{}_{}".format(model.name, L, W, edge, T, addname)
    if k != (0,0):
        FName += "_k{}".format(k)

    if edge == "x-":
        ip = (1, int(W/2))
    elif edge == "x+":
        ip = (L-2, int(W/2))
    elif edge == "y-":
        ip = (int(L/2), 1)
    elif edge == "y+":
        ip = (int(L/2), W-2)
    else:
        raise Exception("ip must be one of: 'x-', 'x+', 'y-', 'y+'.")
    
    init = GaussianWave([L,W], ip, 1, k)
    
    try:
        if force_evolve:
            raise Exception()
        evodat = EvoDat2D.read(FName+".txt")
        print("Read data")
    except Exception as e:
        print("Didn't read: {}".format(e))
        evodat = EvoDat2D.from_evolve(model, L, W, T, dT, init, FName, takedT=tkdT)
        evodat.save()
    if doplots:
        print("Doing plot")
        evodat.animate()
        evodat.plot_normNgrowth()

    if edge == "x-":
        seldat = evodat.res[1,:,:,:]
        vs = (np.arange(W)-int(W/2))/T
    elif edge == "x+":
        seldat = evodat.res[L-2,:,:,:]
        vs = (np.arange(W)-int(W/2))/T
    elif edge == "y-":
        seldat = evodat.res[:,1,:,:]
        vs = (np.arange(L)-int(L/2))/T
    elif edge == "y+":
        seldat = evodat.res[:,W-2,:,:]
        vs = (np.arange(L)-int(L/2))/T
    else:
        raise Exception("ip must be one of: 'x-', 'x+', 'y-', 'y+'.")
    

    seldat = np.linalg.norm(seldat,axis=1)
    Tn = np.shape(seldat)[1]
    
    if Tplot is not None:
        lastind = int(Tplot/tkdT)
        midind = int(Tplot/(2*tkdT))
    else:
        lastind = Tn-1
        midind = int((Tn-1)/2)

    if edge.startswith("x"):
        ip = ip[1]
    else:
        ip = ip[0]

    lastprof = seldat[::2,lastind]
    midprof = seldat[int(ip/2):int(ip/2)+len(lastprof),midind]
    lyaps = (np.log(lastprof)-np.log(midprof)+evodat.norms[lastind]-evodat.norms[midind]+np.log(2))/(T/2)
    vs = (2*np.array([i for i in range(len(lastprof))])-ip)/T

    if doplots:
        plt.figure()
        plt.plot(vs, lyaps, c="C0")

        if comp_prec > 0:
            complyaps = model.GrowthsMMA(vs[0], vs[-1], edge="y" if edge[0]=="x" else "x", prec=comp_prec)
            nvs = [(vs[0]*(comp_prec-i)+vs[-1]*i)/comp_prec for i in range(comp_prec+1)]
            plt.scatter(nvs, complyaps, c="C1", s=10)
            plt.legend(["Numerical", "Theory"])

        plt.xlabel("v")
        plt.ylabel("lambda")
        plt.title("Lyapunov exponents fitted on edge {} at T={}".format(edge, T))
        plt.savefig(FName+".jpg")
        plt.close()

    return vs, lyaps, evodat

########## UPDATED METHOD ##############
def plot_and_compare_2D_Edge (model2d:HamModel, L, W, T, edge="x-", k = 0, ipdepth = 0, kspan = 0.1, dT=0.2, force_evolve = False, addname = ""):

    tkdT = 0.25

    FName = "{}_L{}W{}_{}{}_T{}_{}".format(model2d.name, L, W, edge, ipdepth, T, addname)
    if k != 0:
        FName += "_k{}".format(k)
    if edge == "x-":
        ip = (int(L/2), 0+ipdepth)
        ik = (k,0)
    elif edge == "x+":
        ip = (int(L/2), W-1-ipdepth)
        ik = (k,0)
    elif edge == "y-":
        ip = (0+ipdepth, int(W/2))
        ik = (0,k)
    elif edge == "y+":
        ip = (L-1-ipdepth, int(W/2))
        ik = (0,k)
    else:
        raise Exception("ip must be one of: 'x-', 'x+', 'y-', 'y+'.")
    
    x0, y0 = ip
    
    init = GaussianWave([L,W], ip, 1/(2*kspan), ik)
    
    try:
        if force_evolve:
            raise Exception()
        evodat = EvoDat2D.read(FName)
        print("Read data")
    except Exception as e:
        print("Didn't read: {}".format(e))
        evodat = EvoDat2D.from_evolve_m(model2d, L, W, T, dT, init, FName, takedT=tkdT)
        evodat.save()
    
    seldat = np.linalg.norm(evodat.getRes(), axis=2)
    if edge.startswith("x"):
        seldat = seldat[:,y0,:]
    else:
        seldat = seldat[x0,:,:]
    seldat = seldat**2
    seldat /= np.sum(seldat, axis=0)

    # Construct the projected wave function in one dimension
    ip_proj = y0 if edge.startswith("x") else x0
    L1d = L if edge.startswith("x") else W
    ind = 0 if edge.startswith("x") else 1
    which_edge = y0<W/2 if edge.startswith("x") else x0<L/2

    # For each k, we treat this as a 1D system and get an effective amplitude
    # This gives us the projection of our wave function onto the edge mode.
    xs = np.arange(L1d)
    ks = xs*(2*np.pi)/L1d - np.pi

    xmin = max(0, ip_proj-int(5/kspan))
    xmax = min(L1d-1, ip_proj+int(5/kspan))
    proj_wv = np.zeros(len(ks), xmax-xmin+1, dtype=complex)

    # Sample Ns points of k
    Ns = 10
    ksamp = np.arange(Ns)*(2*np.pi)/Ns - np.pi
    ksamp_zls = []
    for k in ksamp:
        spf = model2d.SPFMMAProj(k, which_edge)
        for row in spf:
            if row[3]:
                z = row[0]
                E = row[1]
                Hpp = row[4]
                if which_edge:
                    zl = model2d.GFMMA2D(E, k, which_edge, *[(x-L1d,ip-L1d) for x in range(xmin, xmax+1)])
                    tamp = np.sum(zl * (init[xmin:xmax+1]))
                else:
                    zl = model2d.GFMMA2D(E, k, *[(x+1,ip+1) for x in range(xmin, xmax+1)])
                ksamp_zls.append(zl)
                break

    for i in range(Ns):
        if i < Ns:
            inext = i+1
            knext = ksamp[inext]
        else:
            inext = 0
            knext = np.pi
        krng = np.logical_and(ks >= ksamp[i], ks < knext)
        ksel = ks[krng]
        proj_wv[krng, :] = ((ksel-ksamp[i])/(knext-ksamp[i])[:,np.newaxis]*ksamp_zls[inext][np.newaxis,:]
            + (knext-ksel)/(knext-ksamp[i])[:,np.newaxis]*ksamp_zls[i][np.newaxis,:])

    # Do FFT on the 2D array in one dimension and inner product with proj_wv

    Eks = np.array(model2d.EnergiesMMA(edge=edge[:1], krange=(ks[0],ks[-1]), prec=L1d-1))
    Ekt = np.exp(-1j*Eks[np.newaxis,:]*evodat.getTimes()[:,np.newaxis])
    kt = Ekt*k0
    xfin = np.tensordot(kt, FT.conj(), axes=1)
    seldat1d = np.abs(xfin.transpose())**2
    seldat1d /= np.sum(seldat1d, axis=0)
    
    evodat.animate_with_curves(np.arange(L1d), [seldat, seldat1d], legends=["Numerical", "Theory"], xlabel="x", ylabel="Amplitude", title=f"Wave Packet Amplitude in {edge[0]} direction")