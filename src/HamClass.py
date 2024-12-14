import itertools
import re
from sre_compile import isstring
from typing import Any
import numpy as np
from math import floor, ceil
from numpy import logical_not, transpose as Tp
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from matplotlib import cm, animation
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider, Button
from random import random
from time import time
import scipy.integrate as si
import scipy
from scipy.signal import savgol_filter
from scipy.integrate import ode
from os.path import isfile, isdir, expandvars
from os import mkdir, chdir, listdir, popen
import os.path
import pathlib
import sympy as sp
from subprocess import check_output as runcmd
import json
from Config import *
import datetime
from shutil import rmtree

from abc import ABC, abstractmethod

# If USE_GPU is true (as defined in Config.py), we will use cupy
if USE_GPU:
    import cupy as cp

# ncp behaves as numpy if UES_GPU is false, and cupy if USE_GPU is true.
ncp = cp if USE_GPU else np

def callMMA(*args):
    """
    Execute the SPCals.wls script with the given arguments.

    Parameters:
    *args: The arguments to pass to the script.

    Returns:
    The result of the script.
    """

    MMAScript = "./SPCalcs.wls"

    t1 = time()

    args = [str(x) for x in args]
    print(f"callMMA: wolframscript {MMAScript} "+" ".join([f'"{x}"' for x in args]), end=" ")

    output = runcmd(["wolframscript", MMAScript] + args)
    output = output.decode("utf-8")
    if output.startswith("ERR::"):
        raise Exception(output[5:])
    try:
        res = json.loads(output)
    except Exception as e:
        raise Exception(f"Error occured during json loading: {e}\nOutput string: {output}")

    # The Mathematica output has a headed nested list structure, we will parse it into Python list.
    def eliminateList(lst):

        if isinstance(lst,list):

            # Deal with ["Complex",a,b]
            if lst[0] == "Complex":
                return lst[1] + 1j*lst[2]
            
            # Deal with ["DirectedInfinity"]
            if lst[0] == "DirectedInfinity":
                return np.inf * eliminateList(lst[1])

            # Eliminate "List" headings
            if lst[0] == "List":
                lst = lst[1:]
                
            return [eliminateList(x) for x in lst]
        else:
            return lst

    t2 = time()-t1    

    print(f"cost {t2} seconds") # Keep track of the time taken to run the MMA script.
    
    return eliminateList(res)


class HamModel:
    """
    A class representing a Hamiltonian model.
    A HamModel object is a model of a Hamiltonian, which contains information of the hoppings, but not the underlying lattice.
    A LatticeHam object is a realization of a HamModel on a lattice.

    Attributes:
    sp_dim: The spatial dimension of the model.
    int_dim: The internal dimension (i.e. number of 'spin' degree of freedom) of the model.
    terms: The term in the model. See LatticeTerm for more information.
    name: The name of the model. Default is "EmptyName".
    """

    def __init__ (self, sp_dim, int_dim, terms, name="EmptyName"):

        self.sp_dim = sp_dim
        self.int_dim = int_dim
        self.terms = terms
        self.name = name

    
    def getHam (self, bcs):
        """
        Returns a LatticeHam object with the given boundary conditions.
        Equivalent to HamModel(bcs).

        Parameters:
        bcs: The boundary conditions. Should be a list of non-negative numbers.
            Each number corresponds to the number of sites in that direction. "0" would mean periodic boundary condition in a direction.
        """
        if type(bcs) != list:
            if self.sp_dim == 1:
                bcs = [bcs]
            else:
                raise Exception("bcs should be a list!")
        return LatticeHam (self.sp_dim, self.int_dim, self.terms, bcs, self.name+"_bc{}".format(bcs))
    
    def __call__(self, bcs):
        return self.getHam(bcs)
    
    
    def OBCPBCSpec_1D (self, obclen = 50, kdens = 200, drawsp = True):
        """
        Draw a plot on the complex energy plane, containing the OBC and PBC, and optionally the saddle points.

        Parameters:
        obclen: The length of the OBC lattice. Default is 50.
        kdens: The number of k points to sample for hte PBC spectrum. Default is 200.
        drawsp: Whether to draw the saddle points. Default is True.
        """

        plt.rcParams.update(PLT_PARAMS)

        # Size of small and large dots in the plot
        s1 = 4
        s2 = 10

        plt.figure(figsize=SINGLE_FIGSIZE, layout="constrained")
        plt.xlabel("Re(E)")
        plt.ylabel("Im(E)")

        # Get the OBC spectrum: realize the Hamiltonian on an OBC lattice and get the eigenvalues
        H = self.getHam([obclen]).realize()
        w, _ = np.linalg.eig(H)
        plt.scatter(np.real(w), np.imag(w), s=s1, label = "OBC")

        # Get the PBC spectrum: realize the Hamiltonian with PBC, get the eigenvalues corresponding to k's.
        ReE = []
        ImE = []
        H = self.getHam([0]).realize()
        for k in [i*2*np.pi/kdens for i in range(kdens)]:
            w, _ = np.linalg.eig(H([k]))
            for E in w:
                ReE.append(np.real(E))
                ImE.append(np.imag(E))
        plt.scatter(ReE, ImE, s=s1, label = "PBC")

        if drawsp:
            spf = self.SPFMMA()
            valsp = []
            invalsp = []
            # See documentation of SPFMMA for the format of spf
            for row in spf:
                if row[3]:
                    valsp.append(row[1])
                else:
                    invalsp.append(row[1])

            plt.scatter(np.real(valsp), np.imag(valsp), s=s2, label = "Valid SP")
            plt.scatter(np.real(invalsp), np.imag(invalsp), s=s2, label = "Invalid SP")
        
        plt.legend()
        plt.savefig(self.name+"OPSpec.pdf")
        plt.close()
    
    
    def to_mathematica (self, var_name = "z", var_type = "z", number_prec = 5):
        """
        Convert the Hamiltonian to a Mathematica expression.

        Parameters:
        var_name: The name of the variable. Default is "z".
        var_type: The type of the variable. Can be "z" or "k". Default is "z".
            If type is z, a hopping is represented as z, z^(-1), etc.
            If type is k, a hopping is represented as Exp[I*k] or Sin[k] or Cos[k], etc.
        number_prec: The precision of the numbers in the coefficients. Default is 5.

        Returns:
        A string, the Mathematica expression of the Hamiltonian.

        Example:
        >>> H = LatticeHam(1, 2, [LatticeTerm(1, 1, 1, 1), LatticeTerm(1, 1, 1, 1)], [0, 0])
        >>> H.to_mathematica()
        "1 Exp[I z] + 1 Exp[-I z]"
        """

        if self.sp_dim == 1:
            vars = [var_name]
        else:
            vars = [var_name+"{}".format(i+1) for i in range(self.sp_dim)]

        def fl2str (fl):
            st = ""
            if var_type == "k":
                for funi, tlist in fl:
                    if funi == LatticeTerm.FUNCTION_SIN:
                        st += "Sin["
                    elif funi == LatticeTerm.FUNCTION_COS:
                        st += "Cos["
                    elif funi == LatticeTerm.FUNCTION_EXP:
                        st += "Exp[I("
                    for nu, kdir in tlist:
                        kds = vars[kdir]
                        if nu == 1:
                            st += "+"+kds
                        elif nu == -1:
                            st += "-"+kds
                        elif nu > 0:
                            st += "+"+str(nu)+kds
                        else:
                            st += str(nu)+kds
                    if funi == LatticeTerm.FUNCTION_EXP:
                        st += ")]"
                    else:
                        st += "]"
            elif var_type == "z":
                for funi, tlist in fl:
                    kst = ""
                    kstinv = ""
                    for nu, kdir in tlist:
                        kds = vars[kdir]
                        if nu == 1:
                            kst += kds+" "
                            kstinv += kds+"^(-1) "
                        elif nu == -1:
                            kst += kds+"^(-1) "
                            kstinv += kds+" "
                        elif nu > 0:
                            kst += kds+"^"+str(nu)+" "
                            kstinv+= kds+"^(-"+str(nu)+") "
                        else:
                            kst += kds+"^("+str(nu)+") "
                            kstinv += kds+"^"+str(-nu)+" "
                    kst = kst[:-1]
                    kstinv = kstinv[:-1]
                    if funi == LatticeTerm.FUNCTION_SIN:
                        st += "("+kst+"-"+kstinv+")/(2I)"
                    elif funi == LatticeTerm.FUNCTION_COS:
                        st += "("+kst+"+"+kstinv+")/2"
                    elif funi == LatticeTerm.FUNCTION_EXP:
                        st += kst
                    st += " "
            else:
                raise Exception("var_type can only be 'z' or 'k'!")
            return st
        
        def fmtCpx (cmpnum):
            # Format a complex number into a Mathematica expression
            r = np.real(cmpnum)
            i = np.imag(cmpnum)
            precstr = "{:."+str(number_prec)+"f}"
            rst = precstr.format(r)
            ist = precstr.format(i)
            if i == 0:
                if r > 0:
                    return "+"+rst
                elif r < 0:
                    return rst
                else:
                    return "+0"
            elif r == 0:
                if i > 0:
                    return "+"+ist+"I"
                else:
                    return ist+"I"
            else:
                return "+("+("+" if r>0 else "")+rst+("+" if i>0 else "")+ist+"I"+")"

        # If the internal dimension is 1, the Hamiltonian is a (Laurent) polynomial in the z's.
        if self.int_dim == 1:
            ostr = ""
            for term in self.terms:
                tmat = term.mat
                fls = fl2str(term.faclist)
                ostr += fmtCpx(tmat)+" "+fls
            return ostr
        else:
            # If the internal dimension is larger than 1, the Hamiltonian is a matrix of z-polynomials.
            ostr = np.zeros((self.int_dim, self.int_dim), dtype=object)
            ostr.fill("")
            for term in self.terms:
                tmat = term.mat
                fls = fl2str(term.faclist)
                for i in range(self.int_dim):
                    for j in range(self.int_dim):
                        if tmat[i,j] != 0:
                            ostr[i,j] += fmtCpx(tmat[i,j])+" "+fls
            rostr = "{"
            for i in range(self.int_dim):
                rostr += "{"
                for j in range(self.int_dim):
                    if ostr[i,j] == "":
                        rostr += "0 , "
                    else:
                        rostr += ostr[i,j]+", "
                rostr = rostr[:-2]+"} , "
            rostr = rostr[:-2]+"}"
            return rostr
    
    # Returns a list of growth rates
    def GrowthsMMA (self, *vlims, edge = None, prec=20):
        if edge is None:
            return callMMA(self.to_mathematica(), "-dim", self.sp_dim, "-prec", prec, "-lv", *["{:.3f}".format(v) for v in vlims])
        else:
            return callMMA(self.to_mathematica(), "-dim", self.sp_dim, "-prec", prec, "-edge", edge, "-lv", *["{:.3f}".format(v) for v in vlims])
    
    # Returns a list of energies
    def EnergiesMMA (self, edge = None, prec=20, krange = []):
        if edge is None:
            return callMMA(self.to_mathematica(), "-dim", self.sp_dim, "-prec", prec, "-hk", *["{:.5f}".format(k) for k in krange])
        else:
            return callMMA(self.to_mathematica(), "-dim", self.sp_dim, "-prec", prec, "-edge", edge, "-hk", *["{:.5f}".format(k) for k in krange])
    
    def _format_v (self, v):
        if isinstance(v, list):
            s = "{"
            for x in v:
                s += "{:.3f}".format(x)
                s += ", "
            s = s[:-2] + "}"
            return s            
        else:
            return "{:.3f}".format(v)

    # Returns a list of [z,H(z)]
    def SPSMMA (self, v = 0):
        return callMMA(self.to_mathematica(), "-dim", self.sp_dim, "-v", self._format_v(v), "-sps")
    
    # Returns a list of [z,H(z),lambda,validity,H''(z)]
    def SPFMMA (self, v = 0):
        return callMMA(self.to_mathematica(), "-dim", self.sp_dim, "-v", self._format_v(v), "-spflow")
    
    # Get the Green's function amplitude
    def GFMMA (self, E, *xpairs):
        return callMMA(self.to_mathematica(), "-ef", f"Complex[{np.real(E)}, {np.imag(E)}]", "{" + ",".join([f"{{{xp[0]},{xp[1]}}}" for xp in xpairs]) + "}")
    
    # Get the Green's function amplitude in terms of derivative
    def GFDMMA (self, E, *xpairs):
        return callMMA(self.to_mathematica(), "-efd", f"Complex[{np.real(E)}, {np.imag(E)}]", "{" + ",".join([f"{{{xp[0]},{xp[1]}}}" for xp in xpairs]) + "}")

class LatticeHam (HamModel):

    def __init__ (self, sp_dim, int_dim, terms, bcs, name="EmptyName"):
        super().__init__(sp_dim, int_dim, terms)
        self.bcs = bcs
        self.name = name

    def realize (self):
        """Realize this Hamiltonian on a lattice. Boundary conditions in each direction choosable.

        Parameters:
        bcs: a list with length equal to sp_dim. Each element correspodns to the boundary condition in that direction.
            bc = 0 means periodic, =N>0 means open with N sites in that direction.

        Returns: A matrix corresponding to the Hamiltonian, with dimension = int_dim * (product of all OBC site numbers)
            If there are periodic conditions given in certain dimensions, the return would be a function that takes a corresponding
            number of k's and returns the matrix under those k's.
        """
        pbcdims = [i for i, x in enumerate(self.bcs) if x == 0]
        if len(pbcdims) == 0:
            return self.realize_given_k(self.bcs, [])
        else:
            return lambda ks: self.realize_given_k(self.bcs, ks)

    def realize_given_k (self, bcs, ks):
        pbcdims = [i for i, x in enumerate(bcs) if x == 0]
        obcdims = [i for i, x in enumerate(bcs) if x > 0]
        if len(kders) == 0:
            kders = [0]*len(pbcdims)
        if len(xcmts) == 0:
            xcmts = [0]*len(obcdims)
        obcnums = [bcs[i] for i in obcdims]
        Ham_dim = int(np.prod(obcnums)*self.int_dim)
        H = np.zeros([Ham_dim, Ham_dim], dtype=complex)
        if len(obcnums) == 0:
            obcsites = [0]
        else:
            obcsites = itertools.product(*[[i for i in range(tlen)] for tlen in obcnums])
        for obcpos in obcsites:
            if len(obcnums) == 0:
                obcpos = [obcpos]
            else:
                obcpos = list(obcpos)
            for term in self.terms:
                for ht in term.getHopTerms(self.bcs, obcpos):
                    mat, poslist = ht
                    if np.size(mat) == 1 and self.int_dim > 1:
                        mat = mat * np.identity(self.int_dim)
                    newpos = obcpos.copy()
                    out_of_range = False
                    for i, d in enumerate(pbcdims):
                        if poslist[d] != 0:
                            mat = mat * np.exp(1j*poslist[d]*ks[i]) * (1j*poslist[d])**kders[i]
                        elif kders[i] != 0:
                            mat = 0
                            break
                    for i, d in enumerate(obcdims):
                        if poslist[d] != 0:
                            newpos[i] += poslist[d]
                            mat = mat * poslist[d]**xcmts[i]
                            if newpos[i] < 0 or newpos[i] >= obcnums[i]:
                                out_of_range = True
                                break
                        elif xcmts[i] != 0:
                            mat = 0
                            break
                    if not out_of_range:
                        opi = self.pos_to_index(obcpos, obcnums)
                        npi = self.pos_to_index(newpos, obcnums)
                        intd = self.int_dim
                        H[npi*intd:(npi+1)*intd, opi*intd:(opi+1)*intd] += mat
        return H

    def complex_Spectrum (self, kdens = 20):
        pbcdims = [i for i, x in enumerate(self.bcs) if x == 0]
        ReE = []
        ImE = []
        H = self.realize()
        if len(pbcdims) == 0:
            w, _ = np.linalg.eig(H)
            for E in w:
                ReE.append(np.real(E))
                ImE.append(np.imag(E))
        else:
            for ks in itertools.product(*[[i*2*np.pi/kdens for i in range(kdens)]]*len(pbcdims)):
                w, _ = np.linalg.eig(H(ks))
                for E in w:
                    ReE.append(np.real(E))
                    ImE.append(np.imag(E))
        plt.figure()
        plt.scatter(ReE, ImE)
        plt.xlabel("Re(E)")
        plt.ylabel("Im(E)")
        plt.title("Complex Energy Spectrum")
        plt.savefig(self.name+"Spec.pdf")
        plt.savefig(self.name+"Spec.jpg")
        plt.close()

    def complex_Spectrum_k (self, kdens = 20, kind = 0, xlog = False, imrange = None, rerange = None):
        try:
            len(kind)
        except:
            kind = [kind]
        pbcdims = [i for i, x in enumerate(self.bcs) if x == 0]
        ReE = []
        ImE = []
        docks = [[] for k in kind]
        H = self.realize()
        if len(pbcdims) == 0:
            self.complex_Spectrum(kdens)
            return
        else:
            for ks in itertools.product(*[[i*2*np.pi/kdens for i in range(kdens)]]*len(pbcdims)):
                w, _ = np.linalg.eig(H(ks))
                if ks[0] == 0 or abs(ks[0]-np.pi/50)<1e-5:
                    pass
                for E in w:
                    ReE.append(np.real(E))
                    ImE.append(np.imag(E))
                    for i in range(len(kind)):
                        docks[i].append(ks[kind[i]])
                    #if np.abs(E) < 0.75:
        if xlog:
            ReE = np.sign(ReE)*np.log(np.amax(np.concatenate((np.reshape(np.abs(ReE), [1, np.size(ReE)]), np.ones([1, np.size(ReE)])*1e-16), axis=0), axis=0))/np.log(10)
        for i in range(len(kind)):
            ki = kind[i]
            plt.figure()
            cmap = cm.get_cmap('RdYlGn')
            imin = np.min(docks)
            imax = np.max(docks)
            ks = (np.array(docks[i])-imin)/(imax-imin)
            cols = [cmap(k) for k in ks]
            plt.scatter(ReE, ImE, c=cols, s=3)
            if rerange is not None:
                plt.xlim(rerange)
            if imrange is not None:
                plt.ylim(imrange)
            plt.xlabel("Re(E)"+(" (Log)" if xlog else ""))
            plt.ylabel("Im(E)")
            plt.title("Complex Energy Spectrum")
            cbar = plt.colorbar(mappable=cm.ScalarMappable(Normalize(imin, imax, True), cmap))
            cbar.set_label("k{}".format(pbcdims[ki]+1))
            plt.savefig(self.name+"SpecK{}.pdf".format(pbcdims[ki]+1))
            plt.savefig(self.name+"SpecK{}.jpg".format(pbcdims[ki]+1))
            plt.close()

    class TimeEvoData:

        def __init__ (self, times, res, Ham):
            self.norms = res[-1,:]
            self.data = res[:-1,:]
            self.times = times
            self.bc = Ham.bc
            self.int_dim = Ham.int_dim

        def get_norms (self, log_scale = False):
            if log_scale:
                return self.norms
            else:
                return np.exp(self.norms)

        def get_data (self, flattened = False):
            if flattened:
                return self.data.flatten()
            else:
                return self.data

        def get_times (self):
            return self.times()

        def get_size (self):
            return len(self.times), self.bc, self.int_dim
        

    def time_evolve (self, T, dT, init, kdens = 20, return_cp_when_possible = False, precision=0, potential = 0, return_idnorm = False, fpcut = 0): 
        pbcdims = [i for i, x in enumerate(self.bcs) if x == 0]
        obcdims = [i for i, x in enumerate(self.bcs) if x > 0]
        times = [dT*i for i in range(floor(T/dT)+1)]

        ncp = cp if USE_GPU and return_cp_when_possible else np

        if self.sp_dim == len(obcdims):
            H = self.realize() + potential
            t1 = time()
            if return_idnorm:
                res, idnorm = LatticeHam._evolve_Ham(H, init, times, return_cp_when_possible = return_cp_when_possible, use_mathematica_precision=precision, return_idnorm = True, fpcut = fpcut)
                norms = ncp.real(res[-1,:])
                res = ncp.reshape(res[:-1,:], self.bcs + [self.int_dim, len(times)])
                idnorm = ncp.reshape(idnorm, np.shape(res))
                print("Evolution calculation time: {}s".format(time()-t1))
                return norms, res, idnorm
            else:
                res = LatticeHam._evolve_Ham(H, init, times, return_cp_when_possible = return_cp_when_possible, use_mathematica_precision=precision, fpcut = fpcut)
                print("Evolution calculation time: {}s".format(time()-t1))
                norms = ncp.real(res[-1,:])
                res = ncp.reshape(res[:-1,:], self.bcs + [self.int_dim, len(times)])
                if USE_GPU and not return_cp_when_possible:
                    if isinstance(norms, cp.ndarray):
                        norms = norms.get()
                    if isinstance(res, cp.ndarray):
                        res = res.get()

                return norms, res


        
    def unitary_evolve (self, T, init): 
        obcdims = [i for i, x in enumerate(self.bcs) if x > 0]

        N = np.size(init)
        wavef = np.reshape(init, [N, 1])
        tnorm = 0
        norms = np.zeros((T+1,))
        res = np.zeros((N, T+1))

        if self.sp_dim == len(obcdims):
            H = self.realize()
            t1 = time()

            for i in range(T+1):
                norms[i] = tnorm
                res[:,i] = wavef
                wavef = H@wavef
                tabs = np.linalg.norm(wavef)
                wavef = wavef / tabs
                tnorm += np.log(tabs)
            
            print("Evolution calculation time: {}s".format(time()-t1))

            return norms, res
        else:
            raise Exception("Evolve is only possible for OBC Hamiltonians!")

    def plot_time_evolve (self, T, dT, norms, res, kdens = 20, intDOF = -1, plotdims = [0,1], color_as_phase = True, view_range = None, show_index = 0):
        pbcdims = [i for i, x in enumerate(self.bcs) if x == 0]
        obcdims = [i for i, x in enumerate(self.bcs) if x > 0]
        times = [dT*i for i in range(floor(T/dT)+1)]
        if self.sp_dim == len(obcdims):
            H = self.realize()
            t1 = time()
            print("Evolution calculation time: {}s".format(time()-t1))
            if self.sp_dim == 1:
                if self.int_dim > 1:
                    res = np.reshape(res, [self.bcs[0], self.int_dim, len(times)])
                    if intDOF >= 0:
                        res = np.reshape(res[:, :, intDOF, :], [self.bcs[0], len(times)])
                    else:
                        res = np.linalg.norm(res, axis=2)
                thisname = self.name+"_Dims{}".format(plotdims)
                if intDOF >= 0:
                    thisname += "_IntDOF{}".format(intDOF)
                name_index = 1
                if show_index == 0:
                    while isfile(thisname+"TimeEvo{}.pdf".format(name_index)) or isfile(thisname+"TimeEvo{}.jpg".format(name_index)):
                        name_index += 1
                    LatticeHam._plot_evo_1D(res, times, norms, color_as_phase, thisname + "TimeEvo{}".format(name_index), show_range = view_range)
                else:
                    while isfile(thisname+"Anim{}.mp4".format(name_index)):
                        name_index += 1
                    LatticeHam._animate_1D(res, times, norms, True, thisname + "Anim{}.mp4".format(name_index), show_range = view_range)
                #LatticeHam._plot_evo_1D(res, times, norms, not color_as_phase, self.name + "TimeEvo{}".format(name_index+1))
            else:
                if self.sp_dim == 2:
                    res = np.reshape(res, [self.bcs[0], self.bcs[1], self.int_dim, len(times)])
                    if intDOF >= 0:
                        res = np.reshape(res[:, :, intDOF, :], [self.bcs[0], self.bcs[1], len(times)])
                    elif self.int_dim == 1:
                        res = np.reshape(res, [self.bcs[0], self.bcs[1], len(times)])
                    else:
                        res = np.linalg.norm(res, axis=2)
                elif self.sp_dim > 2:
                    res = np.reshape(res, self.bcs + [self.int_dim, len(times)])
                    tracedims = [i for i in range(self.int_dim) if i not in plotdims]
                    if intDOF >= 0:
                        res = np.linalg.norm(np.reshape(res[:, :, intDOF, :], self.bcs+[len(times)]), axis=tuple(tracedims))
                    else:
                        tracedims.append(self.int_dim)
                        res = np.linalg.norm(res, axis=tuple(tracedims))
                thisname = self.name+"_Dims{}".format(plotdims)
                if intDOF >= 0:
                    thisname += "_IntDOF{}".format(intDOF)
                name_index = 1
                while isfile(self.name+"Anim{}.mp4".format(name_index)):
                    name_index += 1
                LatticeHam._animate_2D(res, times, norms, color_as_phase, self.name + "Anim{}.mp4".format(name_index))

    def time_evolve_and_plot (self, T, dT, init, kdens = 20, intDOF = -1, plotdims = [0,1], color_as_phase = True, view_range = None, show_index = 0): 
        pbcdims = [i for i, x in enumerate(self.bcs) if x == 0]
        obcdims = [i for i, x in enumerate(self.bcs) if x > 0]
        times = [dT*i for i in range(floor(T/dT)+1)]
        if self.sp_dim == len(obcdims):
            H = self.realize()
            t1 = time()
            res = LatticeHam._evolve_Ham(H, init, times)
            print("Evolution calculation time: {}s".format(time()-t1))
            norms = res[-1,:]
            if self.sp_dim == 1:
                res = res[:-1,:]
                if self.int_dim > 1:
                    res = np.reshape(res, [self.bcs[0], self.int_dim, len(times)])
                    if intDOF >= 0:
                        res = np.reshape(res[:, :, intDOF, :], [self.bcs[0], len(times)])
                    else:
                        res = np.linalg.norm(res, axis=2)
                thisname = self.name+"_Dims{}".format(plotdims)
                if intDOF >= 0:
                    thisname += "_IntDOF{}".format(intDOF)
                name_index = 1
                if show_index == 0:
                    while isfile(thisname+"TimeEvo{}.pdf".format(name_index)) or isfile(thisname+"TimeEvo{}.jpg".format(name_index)):
                        name_index += 1
                    LatticeHam._plot_evo_1D(res, times, norms, color_as_phase, thisname + "TimeEvo{}".format(name_index), show_range = view_range)
                else:
                    while isfile(thisname+"Anim{}.mp4".format(name_index)):
                        name_index += 1
                    LatticeHam._animate_1D(res, times, norms, True, thisname + "Anim{}.mp4".format(name_index), show_range = view_range)
                #LatticeHam._plot_evo_1D(res, times, norms, not color_as_phase, self.name + "TimeEvo{}".format(name_index+1))
            else:
                if self.sp_dim == 2:
                    res = np.reshape(res[:-1,:], [self.bcs[0], self.bcs[1], self.int_dim, len(times)])
                    if intDOF >= 0:
                        res = np.reshape(res[:, :, intDOF, :], [self.bcs[0], self.bcs[1], len(times)])
                    elif self.int_dim == 1:
                        res = np.reshape(res, [self.bcs[0], self.bcs[1], len(times)])
                    else:
                        res = np.linalg.norm(res, axis=2)
                elif self.sp_dim > 2:
                    res = np.reshape(res[:-1,:], self.bcs + [self.int_dim, len(times)])
                    tracedims = [i for i in range(self.int_dim) if i not in plotdims]
                    if intDOF >= 0:
                        res = np.linalg.norm(np.reshape(res[:, :, intDOF, :], self.bcs+[len(times)]), axis=tuple(tracedims))
                    else:
                        tracedims.append(self.int_dim)
                        res = np.linalg.norm(res, axis=tuple(tracedims))
                thisname = self.name+"_Dims{}".format(plotdims)
                if intDOF >= 0:
                    thisname += "_IntDOF{}".format(intDOF)
                name_index = 1
                while isfile(self.name+"Anim{}.mp4".format(name_index)):
                    name_index += 1
                LatticeHam._animate_2D(res, times, norms, color_as_phase, self.name + "Anim{}.mp4".format(name_index))
                #LatticeHam._animate_2D(res, times, norms, not color_as_phase, self.name + "Anim{}.mp4".format(name_index+1))

    def time_evolve_and_fit (self, T, dT, init, expt_slope = None, addname = ""): 
        #pbcdims = [i for i, x in enumerate(self.bcs) if x == 0]
        obcdims = [i for i, x in enumerate(self.bcs) if x > 0]
        times = [dT*i for i in range(floor(T/dT)+1)]
        if self.sp_dim != len(obcdims):
            raise Exception()

        H = self.realize()
        t1 = time()
        res = LatticeHam._evolve_Ham(H, init, times)
        print("Evolution calculation time: {}s".format(time()-t1))
        norms = res[-1,:]
        res = np.reshape(res[:-1,:], self.bcs + [self.int_dim, len(times)])
        res = np.linalg.norm(res, axis=len(self.bcs))
        return LatticeHam._plot_evo_fit_1D(res, times, norms, self.name+"_EF"+addname, expt_slope)

    def _evolve_Ham (H, init, times, return_cp_when_possible = False, use_mathematica_precision = 20, return_idnorm = False, fpcut = 0):

        use_mathematica = use_mathematica_precision > 0

        if USE_GPU and not use_mathematica:
            H = cp.array(H)
            init = cp.array(init)
            ncp = cp
        else:
            ncp = np

        N = ncp.size(init)
        init = ncp.reshape(init, [N, 1])

        if use_mathematica:

            if fpcut > 0:
                print("WARNING: The parameter fpcut will not work if use_mathematica = True!")

            i = 1
            now = datetime.datetime.now().strftime("%m%d%Y-%H%M%S")
            while True:
                dirn = f"TimeEvoTmp_{now}_{i}"
                if isdir(dirn):
                    i += 1
                else:
                    mkdir(dirn)
                    break
            
            curdir = pathlib.Path().resolve()
            pth = os.path.join(curdir, dirn, "")
            H.astype("complex64").tofile(pth+"H.dat")

            np.seterr(divide = 'ignore') 
            psiA = np.round(np.log10(np.abs(init)))
            psiA[np.isinf(psiA)] = -9223372036854775808
            psiP = init / np.exp(np.log(10)*psiA)
            psiP[np.isnan(psiP)] = 0
            np.seterr(divide = 'warn') 
            psiA.astype("int64").tofile(pth+"A.dat")
            psiP.astype("complex64").tofile(pth+"P.dat")

            commands = ["wolframscript", MMASCRIPT_PATH+"HamEvolve.wls", pth, str(use_mathematica_precision), "{" + ", ".join(map(str, [t-times[0] for t in times[1:]])) + "}"]
            t1 = time()
            print(f"Calling Mathematica for time evolution...", end = " ")
            output = runcmd(commands)
            output = output.decode("utf-8")
            if not output.startswith("Success"):
                raise Exception(f"Error in Mathematica time evolution:\n{output}")
            print(f"cost {time()-t1} seconds")
            
            results = ncp.zeros((N+1, len(times)))
            psiA = np.fromfile(f"{pth}TA.dat", "int64")
            psiP = np.fromfile(f"{pth}TP.dat", "complex64")

            rmtree(pth)

            if not return_idnorm:

                psiA = np.reshape(psiA, (len(times)-1, N))
                psiP = np.reshape(psiP, (len(times)-1, N))

                norms = np.max(psiA, axis=1) * np.log(10)
                psi = np.vstack((np.reshape(init, (1,N)), psiP*np.exp(np.log(10)*psiA - norms[:,np.newaxis]))).transpose()
                norms1 = np.apply_along_axis(np.linalg.norm, 0, psi)
                psi /= norms1

                norms = np.hstack(([0], norms))
                norms += np.log(norms1)
                results = np.vstack((psi, norms.reshape((1,len(times)))))

                if USE_GPU and return_cp_when_possible:
                    results = cp.array(results)

                return results
            
            else:

                psi = np.vstack((np.reshape(init, (1,N)), np.reshape(psiP, (len(times)-1, N)))).transpose()
                idnorm = np.vstack((np.zeros((1,N)), np.log(10)*np.reshape(psiA, (len(times)-1, N)))).transpose()
                norms = np.apply_along_axis(np.linalg.norm, 0, psi)
                psi /= norms
                results = np.vstack((psi, np.log(norms).reshape((1,len(times)))))

                if USE_GPU and return_cp_when_possible:
                    results = cp.array(results)
                    idnorm = cp.array(idnorm)

                return results, idnorm

        else:

            init_norm = ncp.linalg.norm(init)
            init = ncp.concatenate((ncp.real(init)/init_norm, ncp.imag(init)/init_norm, ncp.array([[ncp.log(init_norm)]])))
            HR = ncp.real(H)
            HI = ncp.imag(H)
            KR = (HI+ncp.transpose(HI))/2
            KI = (ncp.transpose(HR)-HR)/2

            def toCol (v):
                v = ncp.reshape(v, [ncp.size(v), 1])
                return v

            def f (y, _):
    #            print("f called")
                dydt = ncp.zeros([2*N+1,1])
                dydt[2*N] = ncp.dot(Tp(y[:N]), ncp.dot(KR, y[:N])) + ncp.dot(Tp(y[N:2*N]), ncp.dot(KR, y[N:2*N])) + 2*ncp.dot(Tp(y[N:2*N]), ncp.dot(KI, y[:N]))
                dydt[:N] = toCol(ncp.dot(HI, y[:N]) + ncp.dot(HR, y[N:2*N]) - dydt[2*N] * y[:N])
                dydt[N:2*N] = toCol(ncp.dot(HI, y[N:2*N]) - ncp.dot(HR, y[:N]) - dydt[2*N] * y[N:2*N])
                if abs(ncp.dot(Tp(y[:2*N]), dydt[:2*N])) > 1e-5:
                    pass
    #              print(ncp.dot(Tp(y[:2*N]), y[:2*N]))
    #               print(ncp.dot(Tp(y[:N]), ncp.dot(KR, y[:N])) - ncp.dot(Tp(y[:N]), ncp.dot(HI, y[:N])))
    #                print(ncp.dot(Tp(y[N:2*N]), ncp.dot(KR, y[N:2*N])) - ncp.dot(Tp(y[N:2*N]), ncp.dot(HI, y[N:2*N])))
    #                print("Change in y^2: {}".format(ncp.dot(Tp(y[:2*N]), dydt[:2*N])))
    #                exit()
                return dydt.flatten()

            def jac (y, _):
                print("jac called")
                dydt = ncp.zeros([2*N+1,2*N+1])
                dydt[2*N, :N] = 2*(ncp.dot(KR, y[:N])-ncp.dot(KI, y[N:2*N])).flatten()
                dydt[2*N, N:2*N] = 2*(ncp.dot(KR, y[N:2*N])+ncp.dot(KI, y[:N])).flatten()
                dydt[:N, :N] = HI
                dydt[:N, N:2*N] = HR
                dydt[N:2*N, :N] = -HR
                dydt[N:2*N, N:2*N] = HI
                dydt[:2*N,:2*N] -= (ncp.dot(Tp(y[:N]), ncp.dot(KR, y[:N])) + ncp.dot(Tp(y[N:2*N]), ncp.dot(KR, y[N:2*N])) + 2*ncp.dot(Tp(y[N:2*N]), ncp.dot(KI, y[:N]))) * ncp.identity(2*N)
                dydt[:2*N,:2*N] -= ncp.reshape(y[:2*N], [2*N, 1]) * dydt[2*N, :2*N]
                return dydt
            
            def odeint_by_part (f, jac, initfun, times, NormT):
                res = ncp.zeros([len(times), len(initfun)])
                res[0,:] = initfun.flatten()
                nti = 0
                while True:
                    if nti == len(times)-1:
                        break
                    pti = max([i for i in range(nti, len(times)) if times[i]<times[nti]+NormT])
                    if pti == nti:
                        pti += 1
                    ttimes = times[nti:pti+1]
                    tresult = si.odeint(f, res[nti,:], ttimes, full_output=0, Dfun=jac)
                    tactnorm = ncp.linalg.norm(tresult[:,:-1], axis=1)
                    tresult[:,-1] += ncp.log(tactnorm)
                    tresult[:,:-1] = ncp.transpose(ncp.transpose(tresult[:,:-1]) / tactnorm)
                    if np.isnan(tresult).any():
                        raise Exception("NaN reached in odeint!")
                    res[nti+1:pti+1,:] = tresult[1:,:]
                    nti = pti
                return res

            """
            r = ode(f, jac).set_integrator('zvode', method='bdf').set_initial_value(init, 0)
            results = ncp.zeros([N+1, ncp.size(times)], dtype=complex)
            for i, t in enumerate(times):
                realresult = r.integrate(t)
                if not r.successful():
                    print("return code: {}".format(r.get_return_code()))
                    raise Exception("Integration failed!!!")
                    break
            """
            UseODEINT = False
            if USE_GPU:
                UseODEINT = False

            if not UseODEINT:
                realresult = ncp.zeros([len(times), 2*N+1])
                realresult[0, :] = init.flatten()
                vnow = init

                def rk45_step(func, t, y, h, tol=1e-6):
                    """
                    Perform a single step of the RK45 method.
                    
                    Parameters:
                        func: callable
                            The ODE system function. It should have the form func(t, y).
                        t: float
                            The current time.
                        y: numpy array
                            The current state.
                        h: float
                            The step size.
                        tol: float, optional
                            The tolerance for error estimation.
                    
                    Returns:
                        t_new: float
                            The new time.
                        y_new: numpy array
                            The new state.
                        h_new: float
                            The new step size.
                    """
                    c = ncp.array([0, 1/4, 3/8, 12/13, 1, 1/2])
                    a = [
                        [],
                        [1/4],
                        [3/32, 9/32],
                        [1932/2197, -7200/2197, 7296/2197],
                        [439/216, -8, 3680/513, -845/4104],
                        [-8/27, 2, -3544/2565, 1859/4104, -11/40]
                    ]
                    b = ncp.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
                    b_star = ncp.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])
                    
                    k = []
                    for i in range(6):
                        y_temp = y + h * ncp.sum(ncp.array([a[i][j] * k[j] for j in range(i)]), axis=0)[:,ncp.newaxis] if i > 0 else y
                        k.append(func(y_temp.flatten(), t + c[i] * h))
                    
                    y_new = y + h * ncp.sum(ncp.array([b[i] * k[i] for i in range(6)]), axis=0)[:,ncp.newaxis]
                    y_new_star = y + h * ncp.sum(ncp.array([b_star[i] * k[i] for i in range(6)]), axis=0)[:,ncp.newaxis]
                    
                    error = ncp.linalg.norm(y_new - y_new_star, ord=ncp.inf)
                    if error > tol:
                        h_new = h * min(0.9 * (tol / error)**0.2, 1.0)
                    else:
                        h_new = h * min(0.9 * (tol / error)**0.25, 1.5)
                    
                    return t + h, y_new, h_new

                def rk45(func, t_span, y0, h0=0.1, tol=1e-6):
                    """
                    Solve an ODE using the RK45 method.
                    
                    Parameters:
                        func: callable
                            The ODE system function. It should have the form func(t, y).
                        t_span: tuple
                            A tuple (t0, tf) where t0 is the initial time and tf is the final time.
                        y0: numpy array
                            The initial state.
                        h0: float, optional
                            The initial step size.
                        tol: float, optional
                            The tolerance for error estimation.
                    
                    Returns:
                        t_vals: list
                            The times at which the solution was evaluated.
                        y_vals: list
                            The values of the solution at the corresponding times.
                    """
                    t0, tf = t_span
                    t = t0
                    y = ncp.array(y0, dtype=float)
                    h = h0
                    
                    t_vals = [t]
                    y_vals = [y]
                    
                    while t < tf:
                        if t + h > tf:
                            h = tf - t
                        t, y, h = rk45_step(func, t, y, h, tol)
                        t_vals.append(t)
                        y_vals.append(y)
                    
                    return y
                
                precision = 10

                for i in range(1, len(times)):

                    """
                    ddT = (times[i]-times[i-1]) / precision

                    for _ in range(precision):
                        vnow = vnow + toCol(f(vnow, 0))*ddT
                    """

                    vnow = rk45(f, (times[i-1],times[i]), vnow)

                    vnorm = ncp.linalg.norm(vnow[:-1])
                    vnow[-1] += ncp.log(vnorm)
                    vnow[:-1] /= vnorm

                    if fpcut > 0:
                        # Chop values with abs less than fpcut
                        vnow[:-1][np.abs(vnow[:-1]) < fpcut] = 0
                        vnorm = ncp.linalg.norm(vnow[:-1])
                        vnow[-1] += ncp.log(vnorm)
                        vnow[:-1] /= vnorm

                    if ncp.isnan(vnow).any():
                        raise Exception("NaN encountered in ODE solving!")

                    realresult[i, :] = vnow.flatten()
            else:
                if fpcut > 0:
                    print("WARNING: The parameter fpcut will not work if odeint is used!")
                realresult = odeint_by_part(f, jac, init, times, 5)
                #realresult = si.odeint(f, init.flatten(), times, full_output=0, Dfun=jac)
                #print(info)

            results = ncp.zeros([N+1, len(times)], dtype=complex)
            for i in range(len(times)):
                results[:N, i] = (realresult[i,:N]+1j*realresult[i,N:2*N]).flatten()
                results[N, i] = realresult[i,2*N]
            if USE_GPU and not return_cp_when_possible:
                results = results.get()

            if return_idnorm:
                return results, np.zeros((N, len(times)))
            else:
                return results

    def _plot_evo_1D (datas, times, norm_datas = None, color_as_phase = True, savename = None, show_range=None, dpoints = 200, logcut = 0):
        if len(np.shape(datas)) == 3:
            datas = np.linalg.norm(datas, axis=1)

        L, Ts = np.shape(datas)
        Lsp = ceil(L/dpoints)
        Tsp = ceil(Ts/dpoints)
        times = times[::Tsp]
        if norm_datas is not None:
            norm_datas = norm_datas[::Tsp]
        datas = datas[::Lsp, ::Tsp]
        L, _ = np.shape(datas)
        if show_range is not None:
            show_range = np.array(show_range)/Lsp

        fig = plt.figure(figsize=(8,8))
        if norm_datas is None:
            map_ax = plt.axes([0.1, 0.15, 0.9, 0.7])
        else:
            map_ax = plt.axes([0.1, 0.3, 0.7, 0.65])
            map2_ax = plt.axes([0.85, 0.3, 0.05, 0.65])
            norm_ax = plt.axes([0.1, 0.04, 0.8, 0.2])
            norm_datas = np.exp(norm_datas)
            norm_ax.plot(times, np.real(norm_datas), color="C0")
            norm_ax.set_xlabel("t")
            norm_ax.set_yscale("log")
            norm_ax.set_ylabel("Amp.")

        
        if show_range is None:
            xs = np.array([i for i in range(L)])
            show_datas = datas
        else:
            xs = np.array([i for i in range(*show_range)])
            show_datas = datas[xs, :]

        show_datas = np.abs(show_datas)
        if logcut > 0:
            show_datas[show_datas < 10**(-logcut)] = 10**(-logcut)
            show_datas = np.log10(show_datas)

        ys = times
        X, Y = np.meshgrid(xs, ys)
        pcm = map_ax.pcolormesh(X, Y, np.transpose(show_datas))
        map_ax.set_xlabel("x")
        map_ax.set_ylabel("t")
        fig.colorbar(pcm, cax=map2_ax, orientation="vertical")
        if savename is not None:
            plt.savefig(savename+".jpg")
            plt.savefig(savename+".pdf")
        plt.close()

        #plt.show()

    def _plot_evo_fit_1D (datas, times, norm_datas = None, savename = None, expt_slope = None):

        xs, ts = np.shape(datas)
        x_cords = np.array([i for i in range(xs)])

        smoother_size = 5

        avg_xs = LatticeHam.average_x(np.shape(datas), 0, 1, datas)
        #avg_devs = LatticeHam.average_x_stdev(np.shape(datas), 0, 1, datas, avg_xs)
        #leg_ts = times[avg_xs>150-2*avg_devs]

        smoother = np.ones(smoother_size) / smoother_size
        data_smooth = np.apply_along_axis(lambda m:np.convolve(m, smoother, mode="valid"), axis=0, arr=datas)

        cutoff_amp = 1e-3
        xcordsred = x_cords[:len(x_cords)-(smoother_size-1)]+(smoother_size-1)/2

        #sensible_ts = []

        discard_len = 15

        fit_coefs = []
        p_fit = np.zeros((len(xcordsred),ts))
        for i in range(ts):
            ds = data_smooth[:,i]
            xsensibles = xcordsred[ds>cutoff_amp]
            if not np.any(xsensibles):
                raise Exception("Max value of ds is now {}, too small!".format(np.max(ds)))
            xm = int(np.max(xsensibles))
            dl = int(min(discard_len, xm/2))
            #if xm+discard_len > len(xcordsred):
            #    raise Exception("Usable length of x not enough!")
            pf = np.polyfit(xcordsred[dl:xm], np.log(ds[dl:xm]), deg=1)
            fit_coefs.append(pf[0])
            p_fit[:,i] = np.exp(np.poly1d(pf)(xcordsred))

        devs = np.linalg.norm(data_smooth-p_fit, axis=0)

        if norm_datas is not None:
            fig, axs = plt.subplots(2, 2)
            ax1, ax2, ax3, ax4  = axs.flatten()
        else:
            fig, axs = plt.subplots(1, 3)
            ax1, ax2, ax3 = axs
        ax1.plot(times, devs)
        ax1.set_xlabel("t")
        ax1.set_ylabel("Delta")
        ax1.set_title("Deviation of Exponential Fit")
        ax2.plot(times, fit_coefs)
        if expt_slope is not None:
            ax2.plot(times, [expt_slope]*len(times))
        ax2.set_xlabel("t")
        ax2.set_ylabel("Exponential Coefficient")
        ax2.set_title("Exponential Fit Slope")
        ax3.plot(times, avg_xs)
        ax3.set_xlabel("t")
        ax3.set_ylabel("x_c")
        ax3.set_title("Average x v.s. Time")
        if norm_datas is not None:
            ax4.plot(times, np.real(np.exp(norm_datas)))
            ax4.set_xlabel("t")
            ax4.set_ylabel("Amp.")
            ax4.set_yscale("log")
            ax4.set_title("Overall Norm v.s. Time")

        if savename is not None:
            fig.suptitle(savename)
            plt.savefig(savename+".jpg")
            plt.savefig(savename+".pdf")

        plt.close()

        if norm_datas is not None:
            tind = int(ts/2)
            return fit_coefs[-1], devs[-1], np.polyfit(times[tind:], norm_datas[tind:]+np.log(times[tind:])/2, 1)[0]
        else:
            return fit_coefs[-1], devs[-1]

    def _animate_2D (datas, times, norm_datas = None, color_as_phase = True, savename = None):
        fig = plt.figure(figsize=(8,8))
        if norm_datas is None:
            map_ax = plt.axes([0.1, 0.2, 0.9, 0.7])
            slider_ax = plt.axes([0.1, 0.02, 0.7, 0.05])
            button_ax = plt.axes([0.85, 0.02, 0.1, 0.05])
        else:
            map_ax = plt.axes([0.1, 0.3, 0.9, 0.65])
            norm_ax = plt.axes([0.1, 0.08, 0.8, 0.17])
            slider_ax = plt.axes([0.1, 0.01, 0.7, 0.04])
            button_ax = plt.axes([0.85, 0.01, 0.1, 0.04])
            norm_datas = np.exp(norm_datas)
            norm_ax.plot(times, np.real(norm_datas), color="C0")
            norm_ax.set_xlabel("t")
            norm_ax.set_yscale("log")
            norm_ax.set_ylabel("Amp.")

        L, W, _ = np.shape(datas)
        xs = np.array([i+1 for i in range(L)]*W)
        ys = [y for yr in [[j+1]*L for j in range(W)] for y in yr]
        if color_as_phase:
            cmap = plt.get_cmap("twilight")
            cbar = fig.colorbar(mappable=cm.ScalarMappable(Normalize(-np.pi, np.pi, True), cmap), ax=map_ax)
            cbar.set_label("Phase")
        else:
            cmap = plt.get_cmap("YlGn")
            cbar = fig.colorbar(mappable=cm.ScalarMappable(Normalize(0, 1, True), cmap), ax=map_ax)
            cbar.set_label("Rel. Amplitude")
        sct = None
        nmp = None
        bar = Slider(slider_ax, 't', times[0], times[-1], valinit=times[0], valstep=times[1]-times[0])
        map_ax.set_title("Relative Amplitude")

        max_sz = round((3*fig.dpi/max(L, W))**2)
        max_datas = np.max(np.abs(datas))

        def getClSz (dat):
            MILDER_EXPONENT = 1
            szs = max_sz/((1-np.log(np.abs(dat.flatten())/max_datas+1e-10))**MILDER_EXPONENT)
            if color_as_phase:
                cls = cmap(np.angle(dat.flatten())/(2*np.pi)+0.5)
            else:
                cls = cmap(szs/max_sz)
            return (cls, szs)
        
        def update (t, animate=False):
            nonlocal sct, nmp, bar, pause
            if animate and pause:
                return sct, nmp#, bar
            tind = min(enumerate(times), key=lambda x:abs(x[1]-t))[0]
            dat = datas[:,:,tind]
            cls, szs = getClSz(dat)
            if sct is not None:
                sct.remove()
            sct = map_ax.scatter(xs, ys, s=szs, c=cls)
            if norm_datas is not None:
                if nmp is not None:
                    nmp.remove()
                nmp = norm_ax.scatter([times[tind]], [np.real(norm_datas[tind])], color="C1")
            if animate:
                bar.eventson = False
                bar.set_val(t)
                bar.eventson = True
                return sct, nmp#, bar
            else:
                fig.canvas.draw_idle()

        pause = False
        def onClick (_):
            nonlocal pause
            pause ^= True

        bar.on_changed(update)

        bnext = Button(button_ax, 'Anim')
        bnext.on_clicked(onClick)

        if savename is not None:
            anim = animation.FuncAnimation(fig, lambda i: update(times[i], animate=True),
                                    frames=len(times), interval=100, blit=True)
            # plt.rcParams['animation.ffmpeg_path'] ='D:\\Downloads\\ffmpeg-5.0-essentials_build\\bin\\ffmpeg.exe'
            plt.rcParams['animation.ffmpeg_path'] = FFMPEG_PATH
            # plt.rcParams['animation.ffmpeg_path'] ='/usr/local/bin/ffmpeg'
            anim.save(savename)
        
        pause = True
        bar.set_val(times[0])

        #plt.show()

    def _animate_1D (datas, times, norm_datas = None, log_scale = False, savename = None, show_range=None):
        fig = plt.figure(figsize=(8,8))
        if norm_datas is None:
            map_ax = plt.axes([0.1, 0.2, 0.9, 0.7])
            slider_ax = plt.axes([0.1, 0.02, 0.7, 0.05])
            button_ax = plt.axes([0.85, 0.02, 0.1, 0.05])
        else:
            map_ax = plt.axes([0.1, 0.3, 0.9, 0.65])
            norm_ax = plt.axes([0.1, 0.08, 0.8, 0.17])
            slider_ax = plt.axes([0.1, 0.01, 0.7, 0.04])
            button_ax = plt.axes([0.85, 0.01, 0.1, 0.04])
            norm_datas = np.exp(norm_datas)
            norm_ax.plot(times, np.real(norm_datas), color="C0")
            norm_ax.set_xlabel("t")
            norm_ax.set_yscale("log")
            norm_ax.set_ylabel("Amp.")

        L, _ = np.shape(datas)
        if show_range is None:
            xs = np.array([i for i in range(L)])
            show_datas = datas
        else:
            xs = np.array([i for i in range(*show_range)])
            show_datas = datas[xs, :]
        show_datas = np.abs(show_datas)
        fit_datas = np.zeros(np.shape(show_datas))
        y_ranges = None
        if log_scale:
            show_datas = np.log(show_datas) / np.log(10)
            y_ranges = np.zeros((2,len(times)))
            for j in range(len(times)):
                p = np.poly1d(np.polyfit(xs[:int(len(xs)/2)], show_datas[:int(len(xs)/2),j], deg=1))
                fit_datas[:,j] = p(xs)
            for j in range(len(times)):
                ymi = np.min(show_datas[:,j])
                yma = np.max(show_datas[:,j])
                m = 8
                y_ranges[:,j] = [((m+1)*ymi-yma)/m, ((m+1)*yma-ymi)/m]
        sct = None
        ft = None
        nmp = None
        bar = Slider(slider_ax, 't', times[0], times[-1], valinit=times[0], valstep=times[1]-times[0])
        map_ax.set_title("Relative Amplitude")
        
        def update (t, animate=False):
            nonlocal sct, nmp, bar, pause, ft
            if animate and pause:
                #print("Animate_and_Puase")
                if ft is not None:
                    return sct, nmp, ft
                else:
                    return sct, nmp#, bar
            tind = min(enumerate(times), key=lambda x:abs(x[1]-t))[0]
            dat = show_datas[:,tind]
            if sct is not None:
                sct.remove()
            if ft is not None:
                ft.remove()
            sct = map_ax.plot(xs, dat, color="C0")[0]
            ft = map_ax.plot(xs, fit_datas[:,tind], color="C1")[0]
            if y_ranges is not None:
                map_ax.set_ylim(y_ranges[:,tind])
            if norm_datas is not None:
                if nmp is not None:
                    nmp.remove()
                nmp = norm_ax.scatter([times[tind]], [np.real(norm_datas[tind])], color="C1")
            if animate:
                bar.eventson = False
                bar.set_val(t)
                bar.eventson = True
            else:
                fig.canvas.draw_idle()
            #print("ANO_Puase")
            if ft is not None:
                return sct, nmp, ft
            else:
                return sct, nmp
            #return sct, nmp#, bar

        pause = False
        def onClick (_):
            nonlocal pause
            pause ^= True

        bar.on_changed(update)

        bnext = Button(button_ax, 'Anim')
        bnext.on_clicked(onClick)

        if savename is not None:
            anim = animation.FuncAnimation(fig, lambda i: update(times[i], animate=True),
                                    frames=len(times), interval=100, blit=True)
            plt.rcParams['animation.ffmpeg_path'] = FFMPEG_PATH
            # plt.rcParams['animation.ffmpeg_path'] ='/usr/local/bin/ffmpeg'
            anim.save(savename)
        
        pause = True
        update(times[0])

        #plt.show()
        plt.close()

    def broadcast (arr, pdim, dims):
        return np.transpose(np.transpose(np.zeros(dims[:pdim+1])+np.reshape(arr, [1]*pdim+[dims[pdim]]))+np.zeros(dims[::-1]))

    def average_x (dims, pdim, int_dim, vec):
        xvec = np.transpose(np.transpose(np.zeros(dims[:pdim+1])+np.reshape([i for i in range(dims[pdim])], [1]*pdim+[dims[pdim]]))+np.zeros([int_dim]+dims[::-1]))
        vec2 = np.abs(vec)**2
        vec2 = np.reshape(vec2, dims+[int_dim])
        return np.sum(xvec*vec2)/np.sum(vec2)

    def average_x_stdev (dims, pdim, int_dim, vec, xmean):
        dims = np.array(dims)
        xvec = (xvec-xmean)**2
        vec2 = np.abs(vec)**2
        vec2 = np.reshape(vec2, dims.tolist()+[int_dim])
        return np.sqrt(np.sum(xvec*vec2, axis=pdim)/np.sum(vec2, axis=pdim))

class HamTerm(ABC):

    @abstractmethod
    def getHopTerms(self, bcs, pos):
        pass

class LatticeTerm(HamTerm):

    FUNCTION_SIN = 1
    FUNCTION_COS = 2
    FUNCTION_EXP = 3

    def __init__ (self, mat, arg, ranmat = 0):
        self.mat = mat
        self.ranmat = ranmat
        if type(arg) is list:
            self.faclist = arg
        elif type(arg) is str:
            self.parseString(arg)
        else:
            raise Exception("Invalid initialization argument!")

    def parseList (lst):
        l = []
        for tp in lst:
            if len(tp)==2:
                m, a = tp
                l.append(LatticeTerm(m, a))
            elif len(tp)==3:
                m, a, rm = tp
                #print("Get 3-term: {}".format(tp))
                l.append(LatticeTerm(m, a, rm))
            else:
                raise Exception("Wrong lattice term:\"{}\"".format(tp))
        return l

    def parseString (self, st):
        st = st.strip("* \n\t")
        st = st.replace("[","(").replace("{","(").replace("]",")").replace("}",")").lower()
        stl = st.split(")")
        terms = []
        for term in stl:
            if len(term) == 0:
                continue
            termlist = term.split("(")
            if len(termlist) != 2:
                raise Exception("Invalid input string!")
            fun, arg = termlist
            if fun == "sin":
                funi = self.FUNCTION_SIN
            elif fun == "cos":
                funi = self.FUNCTION_COS
            elif fun == "exp" or fun == "e^":
                funi = self.FUNCTION_EXP
            else:
                raise Exception("Invalid function name: {}".format(fun))
            if arg[0] not in ["+","-"]:
                arg = "+"+arg
            spl = re.findall(r'[+-]\d+|[^+-]+', arg) # Split the string into integer coefficients and k-names
            i = 0
            tlist = []
            while i < len(spl):
                try:
                    nu = int(spl[i])
                    kname = spl[i+1]
                    i = i + 2
                except ValueError:
                    nu = 1
                    kname = spl[i]
                    i = i + 1
                kname = kname.strip("k").replace("x","1").replace("y","2").replace("z","3")
                if kname=="":
                    kdir = 0
                else:
                    kdir = int(kname) - 1
                if nu != 0:
                    tlist.append((nu,kdir))
            terms.append((funi, tlist))
        self.faclist = terms
        #print("String {} parses into {}".format(st, terms))

    def getHopTerms (self, bcs, _):
        spdim = len(bcs)
        terms = self.faclist
        SCinds = [i for i, term in enumerate(terms) if term[0] != self.FUNCTION_EXP]
        Einds = [i for i, term in enumerate(terms) if term[0] == self.FUNCTION_EXP]
        IsSin = np.array([1 if terms[i][0] == self.FUNCTION_SIN else 0 for i in SCinds])
        coef = 2**(-len(SCinds))*(-1j)**(sum(IsSin))
        ht = []
        if len(SCinds) > 0:
            for sc_choice in itertools.product(*[[1, -1] for _ in SCinds]):
                tcoef = coef*(-1)**((IsSin*np.array(sc_choice)).tolist().count(-1))
                poslist = [0]*spdim
                for d in Einds:
                    _, tlist = terms[d]
                    for nu, kdir in tlist:
                        poslist[kdir] -= nu
                for i,d in enumerate(SCinds):
                    _, tlist = terms[d]
                    for nu, kdir in tlist:
                        poslist[kdir] -= sc_choice[i]*nu
                ht.append((tcoef, poslist))
        else:
            poslist = [0]*spdim
            for d in Einds:
                _, tlist = terms[d]
                for nu, kdir in tlist:
                    poslist[kdir] -= nu
            ht.append((1, poslist))
        mat = self.mat + np.random.randn()*self.ranmat
        return [(mat*hop[0],hop[1]) for hop in ht]
    
class EdgePotential(HamTerm):

    def __init__ (self, mat, ranmat, edge, dir, depth):
        self.mat = mat
        self.ranmat = ranmat
        self.edge = edge
        self.dir = dir
        self.depth = depth

    def getHopTerms (self, bcs, site):

        spdim = len(bcs)

        poslist = [0]*spdim

        mat = (self.mat+np.random.randn()*self.ranmat)
        if self.dir == 0:
            mat *= np.exp(-site[self.edge]/self.depth)
        else:
            mat *= np.exp(-(bcs[self.edge]-1-site[self.edge])/self.depth)

        return [(mat, poslist)]
    
class EvoDat1D:

    def from_evolve (time_evolve, T, dT, init, name, takedT = None, precision = 0, potential = 0, return_idnorm = True, fpcut = 0):
        if takedT is None:
            takedT = dT
        ed = EvoDat1D()
        ed.times = np.array([takedT*i for i in range(floor(T/takedT)+1)])
        if return_idnorm:
            ed.norms, ed.res, ed.idnorm = time_evolve(T, dT, init, precision=precision, potential = potential, return_idnorm = return_idnorm, fpcut = fpcut)
        else:
            ed.norms, ed.res = time_evolve(T, dT, init, precision=precision, potential = potential, return_idnorm = False, fpcut = fpcut)
            ed.idnorm = np.ones(np.shape(ed.res))
        #tindices = np.array([floor(t/dT) for t in ed.times])

        #ed.norms = ed.norms[tindices]
        #ed.res = ed.res[:,:,tindices]
        #ed.idnorm = ed.idnorm[:,:,tindices]
        ed.L, ed.idof, ed.T = np.shape(ed.res)
        ed.name = name
        return ed

    def from_unitary_evolve (unitary_evolve, T, init, name):
        ed = EvoDat1D()
        ed.times = [i for i in range(T+1)]
        ed.norms, ed.res = unitary_evolve(T, init)
        ed.L, ed.idof, ed.T = np.shape(ed.res)
        ed.name = name
        return ed

    def read (filename):

        if not filename.endswith(".npz"):
            filename = filename + ".npz"
        data = np.load(filename)
        ed = EvoDat1D()

        ed.name = data["name"].item()
        ed.L = data["L"].item()
        ed.idof = data["idof"].item()
        ed.T = data["T"].item()
        ed.times = data["times"]
        ed.norms = data["norms"]
        ed.res = data["res"]
        try:
            ed.idnorm = data["idnorm"]
        except:
            ed.idnorm = np.zeros(np.shape(ed.res))

        return ed

    def save (self, filename = None):

        if filename is None:
            filename = self.name + ".npz"
        
        if hasattr(self, "idnorm"):
            np.savez(filename, name = self.name, L = self.L, idof = self.idof, T = self.T, times = self.times, norms = self.norms, res = self.res, idnorm = self.idnorm)
        else:
            np.savez(filename, name = self.name, L = self.L, idof = self.idof, T = self.T, times = self.times, norms = self.norms, res = self.res)

    def plot_normNgrowth (self, pt=None, ref = []):

        ptstr = ""

        if pt is None:
            ttimes = self.times
            tnorms = self.norms
        elif -self.L <= pt < self.L:
            ptnorm = np.linalg.norm(self.res[pt,:,:], axis=0)
            choose = ptnorm > 1e-5
            ttimes = np.array(self.times)[choose]
            tnorms = self.norms[choose] + np.log(ptnorm[choose])
            ptstr = "_pt{}".format(pt)
        else:
            raise Exception("pt {} out of range for L = {}".format(pt, self.L))

        plt.figure()
        plt.plot(ttimes, tnorms)
        plt.xlabel("t")
        plt.ylabel("Norm")
        plt.title("L = {}".format(self.L))
        plt.savefig("Evolve"+ptstr+"_"+self.name+".jpg")
        plt.close()

        if len(ttimes) == 0:
            print("No valid output for pt {}".format(pt))
            return

        plt.figure()
        plt.plot(ttimes, sgradient(tnorms, ttimes), label="Growth")
        lm = 0.5
        lt = len(ttimes)
        col = 1
        for ls, nm in ref:
            if len(ls) > 0:
                addlabel = True
                for l in ls:
                    if addlabel:
                        plt.plot(ttimes, [l]*lt, linestyle = "--", color="C{}".format(col), label=nm)
                    else:
                        plt.plot(ttimes, [l]*lt, linestyle = "--", color="C{}".format(col))
                    addlabel = False
                col += 1
            """
            if len(refg) == 2:
                spg, mxg = refg
                lm = max([lm, abs(spg)*1.5, abs(mxg)*1.5])
                plt.plot(ttimes, [spg]*lt, linestyle="--")
                plt.plot(ttimes, [mxg]*lt, linestyle="--")
                plt.legend(["Growth", "SP", "MaxIm"])
            else:
                lm = max([lm, abs(refg[0])*1.5])
                plt.plot(ttimes, [refg[0]]*lt, linestyle="--")
                plt.legend(["Growth", "SP", "MaxIm"])
            """

        #plt.ylim([-lm,lm])
        plt.xlabel("t")
        plt.ylabel("Growth Rate")
        plt.legend()
        plt.title("L = {}".format(self.L))
        plt.savefig("Growth"+ptstr+"_"+self.name+".jpg")
        plt.close()

    def plot_profile (self, logcut = 0):
        if hasattr(self, "idnorm"):
            res = self.res * np.exp(self.idnorm)
            newnorm = np.linalg.norm(res, axis=(0,1))
            norm = self.norms + np.log(newnorm)
            res /= newnorm
        else:
            res = self.res
            norm = self.norms
        LatticeHam._plot_evo_1D(res, self.times, norm, savename="Profile_"+self.name+".jpg", logcut = logcut)

    def plot_xNv (self, refv = None):
        # Calculate average x by time
        xs = np.array([i for i in range(self.L)])
        xt = xs @ (np.linalg.norm(self.res, axis=1)**2)

        # Plot x vs t
        plt.figure()
        plt.plot(self.times, xt)
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title("L = {}".format(self.L))
        plt.savefig("Pos_"+self.name+".jpg")
        plt.close()

        # Plot v vs t
        plt.figure()
        plt.plot(self.times, sgradient(xt, self.times))
        if refv is not None:
            plt.plot(self.times, [refv]*len(self.times), linestyle="--")
            plt.legend(["v", "v_ref"])
        plt.xlabel("t")
        plt.ylabel("v")
        plt.title("L = {}".format(self.L))
        plt.savefig("Vel_"+self.name+".jpg")
        plt.close()

class EvoDat2D:

    def from_evolve (model:HamModel, L, W, T, dT, init, name, takedT = None, return_cp_when_possible = False):
        if takedT is None:
            takedT = dT

        if USE_GPU and return_cp_when_possible:
            ncp = cp
        else:
            ncp = np

        ed = EvoDat2D()
        ed.times = ncp.array([takedT*i for i in range(floor(T/takedT)+1)])
        Ham = model([L,W])
        ed.norms, ed.res = Ham.time_evolve(T, dT, init, return_cp_when_possible = return_cp_when_possible)
        ed.L = L
        ed.W = W
        _,_, ed.idof, ed.T = ncp.shape(ed.res)
        ed.res = ncp.reshape(ed.res, [L, W, ed.idof, ed.T])
        ed.name = name

        """
        if USE_GPU and not return_cp_when_possible:
            ed.norms = ed.norms.get()
            ed.res = ed.res.get()
            ed.times = ed.times.get()
        """

        return ed

    def read (filename, return_cp_when_possible = False):

        if not filename.endswith(".npz"):
            filename = filename + ".npz"
        data = np.load(filename)
        ed = EvoDat2D()

        ed.name = data["name"].item()
        ed.L = data["L"].item()
        ed.W = data["W"].item()
        ed.idof = data["idof"].item()
        ed.T = data["T"].item()
        ed.times = data["times"]
        ed.norms = data["norms"]
        ed.res = data["res"]

        return ed

        """
        def line_to_datas (line, dtype="float"):
            if dtype == "float":
                return ncp.array([ncp.real(complex(x)) for x in line.split()])
            elif dtype == "complex":
                return ncp.array([complex(x.strip("[]")) for x in line.split()])
            else:
                raise Exception("Invalid data type: "+dtype)

        f = open(filename, 'r')
        ed = EvoDat2D()
        ed.name = f.readline()[:-1]
        ed.L, ed.W, ed.idof, ed.T = [int(i) for i in line_to_datas(f.readline())]
        ed.times = line_to_datas(f.readline())
        ed.norms = line_to_datas(f.readline())
        ed.res = ncp.zeros([ed.L*ed.W, ed.idof, ed.T], dtype="complex")
        for t in range(ed.T):
            for id in range(ed.idof):
                ed.res[:,id,t] = line_to_datas(f.readline(), dtype="complex")
        ed.res = ncp.reshape(ed.res, [ed.L, ed.W, ed.idof, ed.T])

        if USE_GPU and not return_cp_when_possible:
            ed.norms = ed.norms.get()
            ed.res = ed.res.get()
            ed.times = ed.times.get()

        return ed
        """

    def save (self, filename = None):

        
        if filename is None:
            filename = self.name + ".npz"
        
        np.savez(filename, name = self.name, L = self.L, W = self.W, idof = self.idof, T = self.T, times = self.times, norms = self.norms, res = self.res)

        """
        if filename is None:
            filename = self.name + ".txt"
        def datas_to_line (arr):
            li = ""
            for e in arr:
                li += "{}\t".format(e)
            li = li[:-1]+'\n'
            return li
        f = open(filename, 'w+')
        f.write(self.name+'\n')
        f.write("{}\t{}\t{}\t{}\n".format(self.L, self.W, self.idof, self.T))
        f.write(datas_to_line(self.times))
        f.write(datas_to_line(self.norms))
        for t in range(self.T):
            for id in range(self.idof):
                f.write(datas_to_line(np.reshape(self.res[:,:,id,t], [self.W*self.L, 1])))
        f.close()
        """

    def getNorms (self, force_numpy = True):
        if USE_GPU and isinstance(self.norms, cp.ndarray) and force_numpy:
            return self.norms.get()
        else:
            return self.norms

    def getRes (self, force_numpy = True):
        if USE_GPU and isinstance(self.norms, cp.ndarray) and force_numpy:
            return self.res.get()
        else:
            return self.res

    def getTimes (self, force_numpy = True):
        if USE_GPU and isinstance(self.norms, cp.ndarray) and force_numpy:
            return self.times.get()
        else:
            return self.times

    def animate (self, color_as_phase = False, cuts = None):
        fig = plt.figure(figsize=(8,8))
        map_ax = plt.axes([0.1, 0.3, 0.9, 0.65])
        norm_ax = plt.axes([0.1, 0.05, 0.8, 0.22])
        norm_datas = np.real(np.exp(self.getNorms()))
        norm_ax.plot(self.getTimes(), norm_datas, color="C0")
        norm_ax.set_xlabel("t")
        norm_ax.set_yscale("log")
        norm_ax.set_ylabel("Amp.")

        xs = np.arange(self.L)+1
        ys = np.arange(self.W)+1
        if cuts is not None:
            xs = xs[cuts[0]]
            ys = ys[cuts[1]]
        Lc = len(xs)
        Wc = len(ys)
        xs, ys = np.meshgrid(xs, ys, indexing="ij")
        xs = xs.flatten()
        ys = ys.flatten()
        datas = np.linalg.norm(self.getRes(), axis=2)
        if cuts is not None:
            datas = datas[cuts[0], cuts[1], :]
        datas = datas / np.max(datas, axis=(0,1))

        #xs = np.array([i+1 for i in range(self.L)]*self.W)
        #ys = [y for yr in [[j+1]*self.L for j in range(self.W)] for y in yr]
        if color_as_phase:
            cmap = plt.get_cmap("twilight")
            cbar = fig.colorbar(mappable=cm.ScalarMappable(Normalize(-np.pi, np.pi, True), cmap), ax=map_ax)
            cbar.set_label("Phase")
        else:
            cmap = plt.get_cmap("YlGn")
            cbar = fig.colorbar(mappable=cm.ScalarMappable(Normalize(0, 1, True), cmap), ax=map_ax)
            cbar.set_label("Rel. Amplitude")
        sct = None
        nmp = None
        map_ax.set_title("Relative Amplitude")

        max_sz = round((3*fig.dpi/max(Lc, Wc))**2)
        #max_datas = np.max(datas)

        def getClSz (dat):
            MILDER_EXPONENT = 1
            flatdat = dat.flatten()
            szs = max_sz * (np.abs(flatdat)**MILDER_EXPONENT) # No log scale
            #szs = max_sz/((1-np.log(np.abs(flatdat)+1e-10))**MILDER_EXPONENT) # Log scale
            if color_as_phase:
                cls = cmap(np.angle(flatdat)/(2*np.pi)+0.5)
            else:
                cls = cmap(szs/max_sz)
            return (cls, szs)
        
        def update (t):
            nonlocal sct, nmp
            tind = min(enumerate(self.times), key=lambda x:abs(x[1]-t))[0]
            dat = datas[:,:,tind]
            cls, szs = getClSz(dat)
            if sct is not None:
                sct.remove()
            sct = map_ax.scatter(xs, ys, s=szs, c=cls)
            if nmp is not None:
                nmp.remove()
            nmp = norm_ax.scatter([self.getTimes()[tind]], [norm_datas[tind]], color="C1")
            return sct, nmp

        anim = animation.FuncAnimation(fig, lambda i: update(self.times[i]),
                                frames=len(self.times), interval=300*(self.times[1]-self.times[0]), blit=True)
        # plt.rcParams['animation.ffmpeg_path'] ='D:\\Downloads\\ffmpeg-5.0-essentials_build\\bin\\ffmpeg.exe'
        plt.rcParams['animation.ffmpeg_path'] = FFMPEG_PATH
        # plt.rcParams['animation.ffmpeg_path'] ='/usr/local/bin/ffmpeg'
        anim.save(self.name + f"_Anim{'_'+str(cuts) if cuts is not None else ''}.mp4")
        plt.close(fig)

    def animate_custom (self, plotdatas, plotname, reference_levels = [], cuts = None):
        fig = plt.figure(figsize=(8,8))
        map_ax = plt.axes([0.1, 0.5, 0.9, 0.45])
        norm_ax = plt.axes([0.1, 0.05, 0.8, 0.42])
        norm_ax.plot(self.getTimes(), plotdatas, color="C0")
        for (i,level) in enumerate(reference_levels):
            norm_ax.plot(self.getTimes(), [level]*len(self.getTimes()), color="C2", linestyle="--")
        norm_ax.set_xlabel("t")
        norm_ax.set_ylabel(plotname)

        xs = np.arange(self.L)+1
        ys = np.arange(self.W)+1
        if cuts is not None:
            xs = xs[cuts[0]]
            ys = ys[cuts[1]]
        Lc = len(xs)
        Wc = len(ys)
        xs, ys = np.meshgrid(xs, ys, indexing="ij")
        xs = xs.flatten()
        ys = ys.flatten()
        datas = np.linalg.norm(self.getRes(), axis=2)
        if cuts is not None:
            datas = datas[cuts[0], cuts[1], :]
        datas = datas / np.max(datas, axis=(0,1))

        cmap = plt.get_cmap("YlGn")
        cbar = fig.colorbar(mappable=cm.ScalarMappable(Normalize(0, 1, True), cmap), ax=map_ax)
        cbar.set_label("Rel. Amplitude")
        sct = None
        nmp = None
        map_ax.set_title("Relative Amplitude")

        max_sz = round((3*fig.dpi/max(Lc, Wc))**2)
        #max_datas = np.max(datas)

        def getClSz (dat):
            MILDER_EXPONENT = 1
            flatdat = dat.flatten()
            szs = max_sz/((1-np.log(np.abs(flatdat)+1e-10))**MILDER_EXPONENT)
            cls = cmap(szs/max_sz)
            return (cls, szs)
        
        def update (t):
            nonlocal sct, nmp
            tind = min(enumerate(self.times), key=lambda x:abs(x[1]-t))[0]
            dat = datas[:,:,tind]
            cls, szs = getClSz(dat)
            if sct is not None:
                sct.remove()
            sct = map_ax.scatter(xs, ys, s=szs, c=cls)
            if nmp is not None:
                nmp.remove()
            nmp = norm_ax.scatter([self.getTimes()[tind]], [plotdatas[tind]], color="C1")
            return sct, nmp

        anim = animation.FuncAnimation(fig, lambda i: update(self.times[i]),
                                frames=len(self.times), interval=300*(self.times[1]-self.times[0]), blit=True)
        # plt.rcParams['animation.ffmpeg_path'] ='D:\\Downloads\\ffmpeg-5.0-essentials_build\\bin\\ffmpeg.exe'
        plt.rcParams['animation.ffmpeg_path'] = FFMPEG_PATH
        # plt.rcParams['animation.ffmpeg_path'] ='/usr/local/bin/ffmpeg'
        anim.save(self.name + f"_Anim_{plotname}{'_'+str(cuts) if cuts is not None else ''}.mp4")
        plt.close(fig)

    def animate_with_curves (self, plotx, plotdatas, legends=None, xlabel="", ylabel="", title="", cuts = None):

        fig = plt.figure(figsize=(8,8))
        map_ax = plt.axes([0.1, 0.5, 0.9, 0.45])
        norm_ax = plt.axes([0.1, 0.05, 0.8, 0.42])
        nmp = []
        for i,data in enumerate(plotdatas):
            nmp.append(norm_ax.plot(plotx, data[:,0], color=f"C{i}", label=legends[i] if legends is not None else f"Line {i+1}")[0])
        if legends is not None:
            norm_ax.legend()
        norm_ax.set_xlabel(xlabel)
        norm_ax.set_ylabel(ylabel)
        norm_ax.set_title(title)

        xs = np.arange(self.L)+1
        ys = np.arange(self.W)+1
        if cuts is not None:
            xs = xs[cuts[0]]
            ys = ys[cuts[1]]
        Lc = len(xs)
        Wc = len(ys)
        xs, ys = np.meshgrid(xs, ys, indexing="ij")
        xs = xs.flatten()
        ys = ys.flatten()
        datas = np.linalg.norm(self.getRes(), axis=2)
        if cuts is not None:
            datas = datas[cuts[0], cuts[1], :]
        datas = datas / np.max(datas, axis=(0,1))

        cmap = plt.get_cmap("YlGn")
        cbar = fig.colorbar(mappable=cm.ScalarMappable(Normalize(0, 1, True), cmap), ax=map_ax)
        cbar.set_label("Rel. Amplitude")
        sct = None
        map_ax.set_title("Relative Amplitude")

        max_sz = round((3*fig.dpi/max(Lc, Wc))**2)
        #max_datas = np.max(datas)

        def getClSz (dat):
            MILDER_EXPONENT = 1
            flatdat = dat.flatten()
            szs = max_sz/((1-np.log(np.abs(flatdat)+1e-10))**MILDER_EXPONENT)
            cls = cmap(szs/max_sz)
            return (cls, szs)
        
        def update (t):
            nonlocal sct, nmp
            tind = min(enumerate(self.times), key=lambda x:abs(x[1]-t))[0]
            dat = datas[:,:,tind]
            cls, szs = getClSz(dat)
            if sct is not None:
                sct.remove()
            sct = map_ax.scatter(xs, ys, s=szs, c=cls)
            if nmp is not None:
                for plot in nmp:
                    plot.remove()
            nmp = []
            for i,data in enumerate(plotdatas):
                nmp.append(norm_ax.plot(plotx, data[:,tind], color=f"C{i}", label=legends[i] if legends is not None else f"Line {i+1}")[0])
            return sct, *nmp

        anim = animation.FuncAnimation(fig, lambda i: update(self.times[i]),
                                frames=len(self.times), interval=300*(self.times[1]-self.times[0]), blit=True)
        # plt.rcParams['animation.ffmpeg_path'] ='D:\\Downloads\\ffmpeg-5.0-essentials_build\\bin\\ffmpeg.exe'
        plt.rcParams['animation.ffmpeg_path'] = FFMPEG_PATH
        # plt.rcParams['animation.ffmpeg_path'] ='/usr/local/bin/ffmpeg'
        anim.save(self.name + f"_Anim_{title}{'_'+str(cuts) if cuts is not None else ''}.mp4")
        plt.close(fig)

    def plot_profile (self, proj_axis = 0):
        if proj_axis == 0:
            datas = np.linalg.norm(self.res, axis=1)
        else:
            datas = np.linalg.norm(self.res, axis=0)
        LatticeHam._plot_evo_1D(datas, self.times, self.norms, savename=self.name+"Profile_" + ("x" if proj_axis==0 else "y") + ".jpg")

    def plot_normNgrowth (self, pt=None, refg = None):

        ptstr = ""

        if pt is None:
            ttimes = self.getTimes()
            tnorms = self.getNorms()
        elif -self.L <= pt < self.L:
            ptnorm = np.linalg.norm(self.getRes()[pt,:,:], axis=0)
            choose = ptnorm > 1e-5
            ttimes = np.array(self.getTimes())[choose]
            tnorms = self.getNorms()[choose] + 2*np.log(ptnorm[choose])
            ptstr = "_pt{}".format(pt)
        else:
            raise Exception("pt {} out of range for L = {}".format(pt, self.L))

        plt.figure()
        plt.plot(ttimes, tnorms)
        plt.xlabel("t")
        plt.ylabel("Norm")
        plt.title("L = {}".format(self.L))
        plt.savefig(self.name+"Evolve"+"_"+ptstr+".jpg")
        plt.close()

        if len(ttimes) == 0:
            print("No valid output for pt {}".format(pt))
            return

        plt.figure()
        plt.plot(ttimes, sgradient(tnorms, ttimes))
        if refg is not None:
            spg, mxg = refg
            lt = len(ttimes)
            plt.plot(ttimes, [spg]*lt, linestyle="--")
            plt.plot(ttimes, [mxg]*lt, linestyle="--")
            plt.legend(["Growth", "SP", "MaxIm"])
        plt.ylim([-1,1])
        plt.xlabel("t")
        plt.ylabel("Growth Rate")
        plt.title("L = {}".format(self.L))
        plt.savefig(self.name+"Growth"+"_"+ptstr+".jpg")
        plt.close()


def positionOP (obcdims, obcdimno, interDOF = 1):
    obcsites = itertools.product(*[[i for i in range(tlen)] for tlen in obcdims])
    op = np.zeros([np.product(obcdims), np.product(obcdims)])
    for i, obcpos in enumerate(obcsites):
        op[i,i] = obcpos[obcdimno]
    if interDOF > 1:
        op = np.kron(op, np.identity(interDOF))
    return op

def recDot (matlist):
    m = 1
    for mat in matlist:
        m = np.dot(m, mat)
    return m

def getIns (lst, pos, val):
    l1 = lst[:]
    l1.insert(pos, val)
    return l1

def MCIntegrate (fun, dim, lbd, hbd, Npts = 1000):
    t1 = time()
    val = 0
    stde = 0
    for i in range(Npts):
        ks = [lbd + (hbd-lbd)*random() for i in range(dim)]
        v = fun(*ks)
        val += v
        stde += np.abs(v)**2
    val = val/Npts
    stde = np.sqrt((stde/Npts-np.abs(val)**2)/(Npts-1))*(hbd-lbd)**dim
    val = val*(hbd-lbd)**dim
    print("Evalulation time: {}, standard error: {}".format(time()-t1, stde))
    return val

def sgradient (xs, ys, windowl=10):
    g = np.gradient(xs, ys)
    gswin = np.exp(-(np.array([i for i in range(6*windowl+1)])/windowl-3)**2)
    gswin = gswin / np.sum(gswin)
    if len(g) < len(gswin):
        return g
    else:
        return np.convolve(g, gswin, mode="same")

def sgradient_sg (xs, ys):
    g = np.gradient(xs, ys)
    if len(g) <= 10:
        return g
    else:
        return savgol_filter(g, 9, 3)
    
def fmpr (str):
    _, columns = popen('stty size', 'r').read().split()
    columns = int(columns)
    return str + " "*(columns*ceil(len(str)/columns)-len(str))

def GaussianWave (bc, center, radius, k =  None, intvec = None, int_dim = 1):
        if k is None:
            k = [0]*len(bc)
        if intvec is None:
            intvec = [1]*int_dim
        coords = np.meshgrid(*[[i for i in range(tlen)] for tlen in bc], indexing="ij")
        distmat = 0
        for i in range(len(bc)):
            distmat += -(coords[i].flatten()-center[i])**2/radius**2 + 1j*coords[i].flatten()*k[i]
        exponvec = np.exp(distmat)
        totvec = np.kron(np.reshape(exponvec, [np.size(exponvec), 1]), np.reshape(intvec, [np.size(intvec), 1]))
        return totvec / np.linalg.norm(totvec)