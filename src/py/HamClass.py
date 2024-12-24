import itertools
import re
import numpy as np
from math import floor
from numpy import transpose as Tp
from matplotlib import pyplot as plt
from matplotlib import cm, animation
from matplotlib.colors import Normalize
from time import time
import scipy.integrate as si
from os.path import isfile, isdir
from os import mkdir
import os.path
import pathlib
from subprocess import check_output as runcmd
import json
from .Config import *
import datetime
from shutil import rmtree

from abc import ABC, abstractmethod

# If USE_GPU is true (as defined in Config.py), we will use cupy
if USE_GPU:
    import cupy as cp
    print("cupy loaded.")

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

    spscript = MMASCRIPT_PATH + "SPCalcs.wls"
    if not isfile(spscript):
        raise Exception(f"Mathematica script path '{MMASCRIPT_PATH}' is invalid! Script {spscript} not found.")

    t1 = time()

    args = [str(x) for x in args]
    print(f"callMMA: {' '.join(MMA_CALLER)} {spscript} "+" ".join([f'"{x}"' for x in args]), end=" ")

    output = runcmd(MMA_CALLER + [spscript] + args)
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
        """
        Uses the MMA script 
        """
        if edge is None:
            return callMMA(self.to_mathematica(), "-dim", self.sp_dim, "-prec", prec, "-lv", *["{:.3f}".format(v) for v in vlims])
        else:
            return callMMA(self.to_mathematica(), "-dim", self.sp_dim, "-prec", prec, "-edge", edge, "-lv", *["{:.3f}".format(v) for v in vlims])
    
    # Returns a list of energies
    def EnergiesMMA (self, edge = None, prec=20, krange = []):
        """
        Uses the MMA scripts to calculate H(k).
        """
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

    def SPSMMA (self, v = 0):
        """
        Uses the MMA script to calculate the saddle points.

        Returns:
        A list of tuples (z, H(z)) of the saddle points.
        """
        return callMMA(self.to_mathematica(), "-dim", self.sp_dim, "-v", self._format_v(v), "-sps")
    
    def SPFMMA (self, v = 0):
        """
        Uses the MMA script to calculate the saddle points and their properties.

        Returns:
        A nested list, each element is a list which contains:
            z: complex
            H(z): complex
            lambda: float, growth rate
            validity: bool, whether the saddle point is relevant
            prefactor: complex, (i*H''(z))^(-1/2)*winding
            vectors: list of complex, the eigenvectors (of internal degree of freedoms) corresponding to the saddle point
        The list is sorted in descending order of lambda.
        """
        if self.sp_dim > 2:
            raise Exception("SPFMMA is only available for 1D and 2D models.")
        return callMMA(self.to_mathematica(), "-dim", self.sp_dim, "-v", self._format_v(v), "-spflow")
    
    def SPFMMAProj (self, k, which_edge):
        """
        Similar to SPFMMA, but now we do it to the projected Hamiltonian, which, for a 2D model, is the reduced 1D model
            where we resolve one direction using the k quantum number.
        
        Parameters:
        k: The k quantum number. Between 0 and 2pi. Replaces the corresponding z to Exp[I*k]
        which_edge: The edge to project. 0 for x, 1 for y. "Project x" means replacing z_x into Exp[I*k], and vice versa.
        """
        if self.sp_dim != 2:
            raise Exception("SPFMMAProj is only available for 2D models.")
        return callMMA(self.to_mathematica(), "-dim", self.sp_dim, "-k", k, "-edge", "x" if which_edge<=0 else "y", "-spflow")

    # Get the Green's function amplitude
    def GFMMA (self, E, *xpairs):
        """
        Get the Green's function amplitude d/dE(<x1|E><<E|x2>) for a given energy eigenstate E and x1, x2.

        Parameters:
        E should be a saddle point energy, and each of xpairs should be a pair of (x1,x2).
        Positive x corresponds to left edge, and negative x correspond to right edge. Indexing of x starts at 1.

        Returns:
        A list of complex numbers, corresponding to d/dE(<x1|E><<E|x2>) for each (x1,x2) given.
        """
        if self.sp_dim != 1:
            raise Exception("GFMMA works only for 1D models right now!")
        if self.int_dim != 1:
            raise Exception("GFMMA works only for single-band models right now!")
        return callMMA(self.to_mathematica(), "-ef", f"Complex[{np.real(E)}, {np.imag(E)}]", "{" + ",".join([f"{{{xp[0]},{xp[1]}}}" for xp in xpairs]) + "}")
    
    # Get the Green's function amplitude
    def GFMMAProj (self, E, k, which_edge, *xpairs):
        """
        Similar to GFMMA, but for 2D models and projected to one edge. See GFMMA and SPFMMAProj.
        """
        if self.sp_dim != 2:
            raise Exception("GFMMAProj works only for 1D models right now!")
        if self.int_dim != 1:
            raise Exception("GFMMA works only for single-band models right now!")
        return callMMA(self.to_mathematica(), "-dim", self.sp_dim, "-k", k, "-edge", "x" if which_edge<=0 else "y", "-ef", f"Complex[{np.real(E)}, {np.imag(E)}]", "{" + ",".join([f"{{{xp[0]},{xp[1]}}}" for xp in xpairs]) + "}")

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
        """Realizes the Hamiltonian on a lattice."""

        pbcdims = [i for i, x in enumerate(bcs) if x == 0]
        obcdims = [i for i, x in enumerate(bcs) if x > 0]
        obcnums = [bcs[i] for i in obcdims]

        Ham_dim = int(np.prod(obcnums)*self.int_dim)
        H = np.zeros([Ham_dim, Ham_dim], dtype=complex)

        if len(obcnums) == 0:
            obcsites = [0]
        else:
            obcsites = itertools.product(*[[i for i in range(tlen)] for tlen in obcnums])
        
        # Add the hopping terms
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

                    # For each hopping term, find where it brings the current obcpos
                    newpos = obcpos.copy()
                    out_of_range = False
                    for i, d in enumerate(pbcdims):
                        if poslist[d] != 0:
                             # If the hopping is along a PBC direction, it gives a phase
                            mat *= np.exp(1j*poslist[d]*ks[i]) 
                    for i, d in enumerate(obcdims):
                        # If the hopping is along an OBC direction, update newpos
                        if poslist[d] != 0:
                            newpos[i] += poslist[d]
                            if newpos[i] < 0 or newpos[i] >= obcnums[i]:
                                out_of_range = True
                                break
                    # Insert the matrix element
                    if not out_of_range:
                        opi = self.pos_to_index(obcpos, obcnums)
                        npi = self.pos_to_index(newpos, obcnums)
                        intd = self.int_dim
                        H[npi*intd:(npi+1)*intd, opi*intd:(opi+1)*intd] += mat
        return H
    
    def pos_to_index (self, pos, ran):
        if len(pos) == 1:
            return pos[0]
        else:
            return pos[0]*np.prod(ran[1:])+self.pos_to_index(pos[1:],ran[1:])

    def complex_Spectrum (self, kdens = 20, kind = [], imrange = None, rerange = None):
        """
        Plots the spectrum of the Hamiltonian on a complex plane.

        Parameters:
        kdens: the number of k-points to sample in each PBC dimension (if any). Default 20
        kind: an integer index or a list of them, default []. 
            If provided, each index would create a plot where the energy spectrum will be colored by the wave vector of that index.
        imrange: a tuple of numbers, default None. Restricting the imaginary axis range.
        rerange: similar to imrange.
            
        Generates plots with the names {self.name}Spec.pdf/jpg, or {self.name}SpecK{kind}.pdf/jpg.
        """

        # If kind is not iterable, make it a list.
        try:
            len(kind)
        except:
            kind = [kind]
        pbcdims = [i for i, x in enumerate(self.bcs) if x == 0]
        ReE = []
        ImE = []
        docks = [[] for _ in kind]
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
        

        if len(kind) == 0:
            plt.figure()
            plt.scatter(ReE, ImE)
            plt.xlabel("Re(E)")
            plt.ylabel("Im(E)")
            plt.title("Complex Energy Spectrum")
            plt.savefig(self.name+"Spec.pdf")
            plt.savefig(self.name+"Spec.jpg")
            plt.close()
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
            plt.xlabel("Re(E)")
            plt.ylabel("Im(E)")
            plt.title("Complex Energy Spectrum")
            cbar = plt.colorbar(mappable=cm.ScalarMappable(Normalize(imin, imax, True), cmap))
            cbar.set_label("k{}".format(pbcdims[ki]+1))
            plt.savefig(self.name+"SpecK{}.pdf".format(pbcdims[ki]+1))
            plt.savefig(self.name+"SpecK{}.jpg".format(pbcdims[ki]+1))
            plt.close()


    def time_evolve (self, T, dT, init, ks = [], return_cp_when_possible = False, method = 0, precision=0, potential = 0, return_idnorm = False, fpcut = 0): 
        """
        Do time evolution on this Hamiltonian.

        Parameters:
        T: The total time for evolution.
        dT: The time step.
        init: Initial wave funciton.
        ks: list of good quantum numbers k, required only if the Hamiltonian has PBC dimensions.
        return_cp_when_possible: Relevant only when GPU is used. Default False.
            If True, the returned arrays will be cupy arrays. If False, they will be cast into numpy arrays.
        precision: floating point precision for Mathematica evaluation. Default 0.
            If non-zero, time evolution will be done with Mathematica; otherwise it will be done natively in Python.
        method: method used for ODE solving. Default 0. If 0, use our own realizaton of RK45 algorithm, which can take advantage of the GPU.
            If 1, use odeint from scipy.
        potential: A matrix corresponding to added potentials. Default 0.
        return_idnorm: Default False. If True, the resulting wave function amplitude will be split into two parts,
            one "res" documents the phase angle only, while one "idnorm" documents the log of the norm.
            This could prevent floating point overflow/underflow when the evolution time is long.
        fpcut: The floating point round off error, elements in psi whose relative magnitude is smaller than this will be set to zero.
            Default 0. Relevant only when time evolution is done with RK45.

        Returns:
        norms: 1D real array, contains log(|psi(t)|).
        res: Array of shape (self.bcs, self.int_dim, #timesteps), contains the normalized wave function psi(t).
            If return_idnorm is True, psi(t) should be res * np.exp(log(10)*idnorm).
        idnorm: Returned only if return_idnorm is True. Integer array of the same shape as res,
            documents the approximate amplitude of res in powers of 10.
        """

        obcdims = [i for i, x in enumerate(self.bcs) if x > 0]
        H0 = self.realize()

        if self.sp_dim != len(obcdims):
            try:
                H0 = H0(ks)
            except Exception as e:
                raise Exception(f"PBC Hamiltonian cannot be used in time_evolve, try providign the correct ks.\n{e}")
        
        times = [dT*i for i in range(floor(T/dT)+1)]
        ncp = cp if (USE_GPU and return_cp_when_possible) else np

        H = self.realize() + potential
        t1 = time()
        if return_idnorm:
            res, idnorm = LatticeHam._evolve_Ham(H, init, times, return_cp_when_possible = return_cp_when_possible, use_mathematica_precision=precision, method=method, return_idnorm = True, fpcut = fpcut)
            norms = ncp.real(res[-1,:])
            res = ncp.reshape(res[:-1,:], self.bcs + [self.int_dim, len(times)])
            idnorm = ncp.reshape(idnorm, np.shape(res))
            print("Evolution calculation time: {}s".format(time()-t1))
            return norms, res, idnorm
        else:
            res = LatticeHam._evolve_Ham(H, init, times, return_cp_when_possible = return_cp_when_possible, use_mathematica_precision=precision, method=method, fpcut = fpcut)
            print("Evolution calculation time: {}s".format(time()-t1))
            norms = ncp.real(res[-1,:])
            res = ncp.reshape(res[:-1,:], self.bcs + [self.int_dim, len(times)])
            if USE_GPU and not return_cp_when_possible:
                if isinstance(norms, cp.ndarray):
                    norms = norms.get()
                if isinstance(res, cp.ndarray):
                    res = res.get()

            return norms, res
        

    def _evolve_Ham (H, init, times, return_cp_when_possible = False, method = 0, use_mathematica_precision = 20, return_idnorm = False, fpcut = 0):
        """
        Function that realizes time evolution.
        See time_evolve for parameter specifications.
        """

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
            # Create temporary files to pass the Hamiltonian and initial wave function to Mathematica
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

            # Split psi into two arrays psiA and psiP, where psiA is round(log_10(|psi|))
            # Special care has to be taken since |psi| could be zero.
            np.seterr(divide = 'ignore') 
            psiA = np.round(np.log10(np.abs(init)))
            psiA[np.isinf(psiA)] = -9223372036854775808
            psiP = init / np.exp(np.log(10)*psiA)
            psiP[np.isnan(psiP)] = 0
            np.seterr(divide = 'warn') 
            psiA.astype("int64").tofile(pth+"A.dat")
            psiP.astype("complex64").tofile(pth+"P.dat")

            evolve_script = MMASCRIPT_PATH+"HamEvolve.wls"
            if not isfile(evolve_script):
                raise Exception(f"Evolution script {evolve_script} not found!")

            commands = ["wolframscript", evolve_script, pth, str(use_mathematica_precision), "{" + ", ".join(map(str, [t-times[0] for t in times[1:]])) + "}"]
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
        # Use other methods.

            # First initialize the wave function and the Hamiltonian as real arrays.
            # If our wave function is N-dimensional, we will work with a 2N+1-dimensional real array,
            # with the first dimension being the log of the norm, and the rest 2N dimensions being the real and 
            # imaginary parts of the normalized wave function.
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
                dydt = ncp.zeros([2*N+1,1])
                dydt[2*N] = ncp.dot(Tp(y[:N]), ncp.dot(KR, y[:N])) + ncp.dot(Tp(y[N:2*N]), ncp.dot(KR, y[N:2*N])) + 2*ncp.dot(Tp(y[N:2*N]), ncp.dot(KI, y[:N]))
                dydt[:N] = toCol(ncp.dot(HI, y[:N]) + ncp.dot(HR, y[N:2*N]) - dydt[2*N] * y[:N])
                dydt[N:2*N] = toCol(ncp.dot(HI, y[N:2*N]) - ncp.dot(HR, y[:N]) - dydt[2*N] * y[N:2*N])
                if abs(ncp.dot(Tp(y[:2*N]), dydt[:2*N])) > 1e-5:
                    pass
                return dydt.flatten()

            if method == 0:
                # Use our own RK45 method. This could take advantage of GPU.
                
                from .RK45 import rk45

                realresult = ncp.zeros([len(times), 2*N+1])
                realresult[0, :] = init.flatten()
                vnow = init

                for i in range(1, len(times)):

                    vnow = rk45(f, (times[i-1],times[i]), vnow, ncp=ncp)

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
                # Use ODEINT of scipy. We ODEINT by several time slices, and normalize the wave function after each slice.

                if fpcut > 0:
                    print("WARNING: The parameter fpcut will not work if odeint is used!")
                
                # Define the 
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
                
                NormT = 5 # NormT means the number of dT's after which we will be one round of normalization.
                res = ncp.zeros([len(times), len(init)])
                res[0,:] = init.flatten()
                nti = 0
                while True:
                    if nti == len(times)-1:
                        break
                    pti = max([i for i in range(nti, len(times)) if times[i]<times[nti]+NormT])
                    if pti == nti:
                        pti += 1
                    ttimes = times[nti:pti+1]
                    tresult = si.odeint(f, res[nti,:], ttimes, full_output=0, Dfun=jac)
                    # Take the norm of the psi part and put that into the norm part.
                    tactnorm = ncp.linalg.norm(tresult[:,:-1], axis=1)
                    tresult[:,-1] += ncp.log(tactnorm)
                    tresult[:,:-1] = ncp.transpose(ncp.transpose(tresult[:,:-1]) / tactnorm)
                    if np.isnan(tresult).any():
                        raise Exception("NaN reached in odeint!")
                    res[nti+1:pti+1,:] = tresult[1:,:]
                    nti = pti
                realresult = res   

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
