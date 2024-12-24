import numpy as np
from math import floor, ceil
from matplotlib import pyplot as plt
from matplotlib import cm, animation
from matplotlib.colors import Normalize
from os.path import isfile
from .Config import *
from .HamClass import LatticeHam, HamModel

# If USE_GPU is true (as defined in Config.py), we will use cupy
if USE_GPU:
    import cupy as cp

def cast_numpy (arr):
    if USE_GPU and isinstance(arr, cp.ndarray):
        return arr.get()
    else:
        return arr
    
def sgradient (xs, ys, windowl=10):
    g = np.gradient(xs, ys)
    gswin = np.exp(-(np.array([i for i in range(6*windowl+1)])/windowl-3)**2)
    gswin = gswin / np.sum(gswin)
    if len(g) < len(gswin):
        return g
    else:
        return np.convolve(g, gswin, mode="same")
    
class EvoDat1D:
    """
    A class that compiles the result of a time evolution and provides certain visualization functions.
    1D version.

    Attributes:
    L: Size of the system
    idof: Internal degree of freedom
    T: Number of time steps
    name: Name of the Hamiltonian
    norm: Array of length T, containing log|psi(t)|
    res, idnorm: Both array of shape (L, idof, T), together containing normlaized psi(t),
        with psi(t) = res * exp(log(10)*idnorm)

    Notice that since 1D evolutions typically don't require GPU, everything here are stored as numpy arrays.
    """

    def from_evolve (Ham:LatticeHam, T, dT, init, name, takedT = None, precision = 0, potential = 0, return_idnorm = True, fpcut = 0):
        """
        Genereate an EvoDat1D object from time evolution of a LatticeHam

        Parameters:
        Ham: The Hamiltonian from evlution
        T: Evolution time span
        dT: Time step for evolution
        init: Initial wave function
        name: Name of the Hamiltonian
        takedT: Default None. If given, the stored data will have time step takedT.
            I.e., when given, only store part of the evolved data to save space.
        precision, potential, return_idnorm, fpcut: See LatticeHam.time_evolve
        """
        if takedT is None:
            takedT = dT
        ed = EvoDat1D()
        ed.times = np.array([takedT*i for i in range(floor(T/takedT)+1)])
        if return_idnorm:
            ed.norms, ed.res, ed.idnorm = Ham.time_evolve(T, dT, init, return_cp_when_possible=False, precision=precision, potential = potential, return_idnorm = return_idnorm, fpcut = fpcut)
        else:
            ed.norms, ed.res = Ham.time_evolve(T, dT, init, return_cp_when_possible=False, precision=precision, potential = potential, return_idnorm = False, fpcut = fpcut)
            ed.idnorm = np.zeros(np.shape(ed.res))
            
        ed.L, ed.idof, ed.T = np.shape(ed.res)
        ed.name = name
        return ed

    def from_evolve_m(model:HamModel, L, *args, **kwargs):
        return EvoDat1D.from_evolve(model([L]), *args, **kwargs)

    def read (filename):
        # Read an EvoDat1D object from a npz file

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
        # Save the current EvoDat1D object to a npz file
        if filename is None:
            filename = self.name + ".npz"
        
        if hasattr(self, "idnorm"):
            np.savez(filename, name = self.name, L = self.L, idof = self.idof, T = self.T, times = self.times, norms = self.norms, res = self.res, idnorm = self.idnorm)
        else:
            np.savez(filename, name = self.name, L = self.L, idof = self.idof, T = self.T, times = self.times, norms = self.norms, res = self.res)

    def plot_growth (self, pt=None, ref = []):
        """
        Generate a plot of the growth rate of the magnitude of the wave function.
        If pt is not given, will plot the total magnitude, i.e., |psi(t)|
        If pt is given, will plot the magnitude of the wave function at site pt.

        refs is a list of tuples (values, name), where values is a list of real numbers, and name is a string.
        If given, will plot the values as horizontal reference lines.
        """

        ptstr = ""

        if pt is None:
            ttimes = self.times
            tnorms = self.norms
        elif -self.L <= pt < self.L:
            ptnorm = np.linalg.norm(self.res[pt,:,:], axis=0)
            choose = ptnorm > 1e-5
            ttimes = np.array(self.times)[choose]
            tnorms = self.norms[choose] + np.log(ptnorm[choose]) + self.idnorm[pt,:,:]
            ptstr = "_pt{}".format(pt)
        else:
            raise Exception("pt {} out of range for L = {}".format(pt, self.L))

        if len(ttimes) == 0:
            print("No valid output for pt {}".format(pt))
            return
        
        EvoDat1D._plot_growth(ttimes, tnorms, "Growth"+ptstr+"_"+self.name, "L = {}".format(self.L), ref)

    def _plot_growth(times, norms, name, title, ref):

        plt.figure()
        plt.plot(times, sgradient(norms, times), label="Growth")
        lt = len(times)
        col = 1
        for ls, nm in ref:
            if len(ls) > 0:
                addlabel = True
                for l in ls:
                    if addlabel:
                        plt.plot(times, [l]*lt, linestyle = "--", color="C{}".format(col), label=nm)
                    else:
                        plt.plot(times, [l]*lt, linestyle = "--", color="C{}".format(col))
                    addlabel = False
                col += 1

        plt.xlabel("t")
        plt.ylabel("Growth Rate")
        plt.legend()
        plt.title(title)
        plt.savefig(name+".jpg")
        plt.close()

    def plot_profile (self, dpoints = 200, idof = None, show_range = None, plot_norms = True):
        """
        Plot the evolution of the wave packet in a x-t color plot.

        Parameters:
        dpoints: Default 200. If the points in the x or t dimension exceed this number,
            sample it coarse-grain so as not to blow up the plotting time.
        idof: Default None: If None, trace out the internal degree of freedom with 2-norm.
            If an integer, pick the internal degree of freedom to be that particular number.
            If a vector, take the internal degree of freedom to be dotted with the vector.
        show_range: Default None. If give, should be a tuple of two ints that indicate the x-range for the plot.
        plot_norms: Default True. If True, there will be a lower panel that shows the growth of the profile.
        """

        if hasattr(self, "idnorm"):
            res = self.res * np.exp(self.idnorm)
            newnorm = np.linalg.norm(res, axis=(0,1))
            norm = self.norms + np.log(newnorm)
            res /= newnorm
        else:
            res = self.res
            norm = self.norms

        Lsp = ceil(self.L/dpoints)
        Tsp = ceil(self.T/dpoints)
        times = self.times[::Tsp]
        norm_datas = self.norms[::Tsp]
        xs = np.arange(self.L)[::Lsp]
        if show_range is not None:
            xs = xs[np.logical_and(xs >= show_range[0], xs < show_range[1])]

        datas = res[xs, :, ::Tsp]

        if idof is None:
            datas = np.linalg.norm(datas, axis=1)
        elif isinstance(idof, int):
            datas = np.abs(datas[:,idof,:], axis=1)
        else:
            datas = np.abs(np.sum(datas * np.conj(idof)[np.newaxis,:,np.newaxis]))

        fig = plt.figure(figsize=SQUARE_FIGSIZE)

        if plot_norms:
            map_ax = plt.axes([0.1, 0.3, 0.7, 0.65])
            map2_ax = plt.axes([0.85, 0.3, 0.05, 0.65])
            norm_ax = plt.axes([0.1, 0.04, 0.8, 0.2])
            norm_datas = np.exp(norm_datas)
            norm_ax.plot(times, np.real(norm_datas), color="C0")
            norm_ax.set_xlabel("t")
            norm_ax.set_yscale("log")
            norm_ax.set_ylabel("Amp.")
        else:
            map_ax = plt.axes([0.1, 0.15, 0.9, 0.7])

        ys = times
        X, Y = np.meshgrid(xs, ys)
        pcm = map_ax.pcolormesh(X, Y, np.transpose(datas))
        map_ax.set_xlabel("x")
        map_ax.set_ylabel("t")
        fig.colorbar(pcm, cax=map2_ax, orientation="vertical")

        savename = "Profile_"+self.name
        plt.savefig(savename+".jpg")
        plt.savefig(savename+".pdf")
        plt.close()


    def plot_xNv (self, refv = None):
        """
        Plot the x(t) and v(t) of the wave packet.
        """
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
    """
    A class that compiles the result of a time evolution and provides certain visualization functions.
    2D version.

    Attributes similar to EvoDat1D, except that the dimension L has become two parameters, L and W.
    Furthermore, here it is sometimes advantagenous to keep the array in cupy format.
    Therefore, the initialization function has a return_cp_when_possible argument.
    """

    def from_evolve (Ham:LatticeHam, T, dT, init, name, takedT = None, return_cp_when_possible = False):

        ed = EvoDat2D()

        if USE_GPU and return_cp_when_possible:
            ncp = cp
            ed.isCupy = True
        else:
            ncp = np
            ed.isCupy = False

        ed.norms, ed.res = Ham.time_evolve(T, dT, init, return_idnorm=False, return_cp_when_possible = return_cp_when_possible)
        ed.L = Ham.bcs[0]
        ed.W = Ham.bcs[1]
        _,_, ed.idof, ed.T = ncp.shape(ed.res)
        ed.times = ncp.array([dT*i for i in range(ed.T)])
        if takedT is not None:
            ed.times = ed.times[::int(takedT/dT)]
            ed.norms = ed.norms[::int(takedT/dT)]
            ed.res = ed.res[:,:,::int(takedT/dT)]
            ed.T = len(ed.times)
        ed.res = ncp.reshape(ed.res, [ed.L, ed.W, ed.idof, ed.T])
        ed.name = name

        return ed
    
    def from_evolve_m (model:HamModel, L, W, *args, **kwargs):
        return EvoDat2D.from_evolve(model([L,W]), *args, **kwargs)

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

        if USE_GPU and return_cp_when_possible:
            ed.norms = cp.array(ed.norms)
            ed.times = cp.array(ed.times)
            ed.res = cp.array(ed.res)
            ed.isCupy = True
        else:
            ed.isCupy = False

        return ed

    def save (self, filename = None):

        if filename is None:
            filename = self.name + ".npz"
        
        np.savez(filename, name = self.name, L = self.L, W = self.W, idof = self.idof, T = self.T,
                 times = self.getTimes(force_numpy=True), norms = self.getNorms(force_numpy=True), res = self.getRes(force_numpy=True))

    def getNorms (self, force_numpy = True):
        return cast_numpy(self.norms) if force_numpy else self.norms

    def getRes (self, force_numpy = True):
        return cast_numpy(self.res) if force_numpy else self.res

    def getTimes (self, force_numpy = True):
        return cast_numpy(self.times) if force_numpy else self.times

    def animate (self, lower_panel_init, lower_panel_update, name, color_as_phase = False, cuts = None, point_scaling_exponent = 1, t_interval = 0.3):
        """
        Animate the wave packet evolution in 2D space.
        Has two panels, with upper panel plotting wave packet profile, and lower panel plotting some other information, which can be simultaneously animated.
        Requires ffmpeg to run. Install ffmpeg and enter its path into Config.py.
        
        Parameters:
        lower_panel_init: a function that takes an axis and returns a list of plotted objects.
            Corresponds to the initialization plot of the lower panel.
        lower_panel_update: a function that takes an axis and an integer t (index of time),
            does some plot at time t, and returns a list of plotted objects in the lower panel.
        name: a string to save the name of the animation.
        color_as_phase: Default False. If True, the color of the wave function profile corresponds to the phase of the wave function.
            If False, corresponds to the amplitude.
        cuts: Default None. If provided, should be a tuple of two numpy array, corresponding to the indices to take in x and y directions.
        point_scaling_exponent: Default 1. Determines how sensitive the size of a representative dot is to the magnitude of the wave function.
            A smaller value would make parts of the wave function with smaller amplitude more visible.
        t_interval: Default 0.3. How long (in seconds) would a unit t interval be in the output video.
        """

        if not isfile(FFMPEG_PATH):
            raise Exception(f"ffmpeg path '{FFMPEG_PATH}' is invalid!")
        
        fig = plt.figure(figsize=SQUARE_SHOW_FIGSIZE, layout="constrained")

        # The figure will be split into two parts, 
        map_ax = plt.axes([0.1, 0.5, 0.9, 0.45])
        norm_ax = plt.axes([0.1, 0.05, 0.8, 0.42])

        # lp_init plots things on the lower panel that are persistent throughout the animation
        nmp = lower_panel_init(norm_ax)
        sct = None
        # nmp keeps track of all the objects plotted in the norm_ax
        # sct keeps track of the scatter plot in the map_ax
        # Both have to be removed and re-plotted at each update step.

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
        datas = np.linalg.norm(self.getRes(force_numpy=True), axis=2)
        if cuts is not None:
            datas = datas[cuts[0], cuts[1], :]
        datas = datas / np.max(datas, axis=(0,1))

        # Create a color bar
        if color_as_phase:
            cmap = plt.get_cmap("twilight")
            cbar = fig.colorbar(mappable=cm.ScalarMappable(Normalize(-np.pi, np.pi, True), cmap), ax=map_ax)
            cbar.set_label("Phase")
        else:
            cmap = plt.get_cmap("YlGn")
            cbar = fig.colorbar(mappable=cm.ScalarMappable(Normalize(0, 1, True), cmap), ax=map_ax)
            cbar.set_label("Rel. Amplitude")
        map_ax.set_title("Relative Amplitude")

        # Maximal size for representative point for the wave packet
        max_sz = round((3*fig.dpi/max(Lc, Wc))**2)

        def getClSz (dat):
            flatdat = dat.flatten()
            szs = 1/((1-np.log(np.abs(flatdat)+1e-10))**point_scaling_exponent)
            if color_as_phase:
                cls = cmap(np.angle(flatdat)/(2*np.pi)+0.5)
            else:
                cls = cmap(szs/max_sz)
            return (cls, szs)
        
        def update (t):
            # At each step, redo the plot.
            nonlocal sct, nmp
            tind = min(enumerate(self.times), key=lambda x:abs(x[1]-t))[0]
            dat = datas[:,:,tind]
            cls, szs = getClSz(dat)
            if sct is not None:
                sct.remove()
            sct = map_ax.scatter(xs, ys, s=szs, c=cls)
            for obj in nmp:
                obj.remove()
            nmp = lower_panel_update(norm_ax, tind)
            return sct, *nmp
        
        rets = update(0)
        sct = rets[0]
        nmp = rets[1:]

        anim = animation.FuncAnimation(fig, lambda i: update(self.times[i]),
                                frames=len(self.times), interval=t_interval*1000*(self.times[1]-self.times[0]), blit=True)
        plt.rcParams['animation.ffmpeg_path'] = FFMPEG_PATH
        anim.save(self.name + f"_Anim{name}{'_'+str(cuts) if cuts is not None else ''}.mp4")
        plt.close(fig)

    def animate_with_norm (self, **kwargs):
        """
        Animate with the lower panel showing the real-time growth rate of the total amplitude.
        """
            
        norm_datas = np.real(np.exp(self.getNorms()))
        norm_datas = sgradient(self.getTimes(), norm_datas)

        def lower_panel_init (norm_ax):
            norm_ax.plot(self.getTimes(), norm_datas, color="C0")
            norm_ax.set_xlabel("t")
            norm_ax.set_yscale("log")
            norm_ax.set_ylabel("Growth Rate")
            return []

        def lower_panel_update (norm_ax, t):
            return [norm_ax.scatter([self.getTimes()[t]], [norm_datas[t]], color="C1")]

        self.animate(lower_panel_init, lower_panel_update, "", **kwargs)

    def animate_with_curves (self, plotx, plotdatas, name, legends=None, xlabel="", ylabel="", title="", **kwargs):
        """
        Plots customary curves in the lower panel.

        Parameters:
        plotx: An array giving the x-axis of the lower panel.
        plotdatas: a list of 2D arrays of shape [len(plotx), T]. Each gives a curve to plot.
        legends: Defualt None. If given, should be a list of strings of the same length as plotdatas.
        xlabel: Default "". The x-axis label of the lower panel plot.
        ylabel: Default "". The y-axis label of the lower panel plot.
        title: Default "". The title of the lower panel plot.
        """

        def lower_panel_init (norm_ax):
            nmp = []
            for i,data in enumerate(plotdatas):
                nmp.append(norm_ax.plot(plotx, data[:,0], color=f"C{i}", label=legends[i] if legends is not None else f"Line {i+1}")[0])
            if legends is not None:
                norm_ax.legend()
            norm_ax.set_xlabel(xlabel)
            norm_ax.set_ylabel(ylabel)
            norm_ax.set_title(title)
            return nmp

        def lower_panel_update (norm_ax, t):
            return [norm_ax.plot(plotx, data[:,t], color=f"C{i}",
                        label=legends[i] if legends is not None else f"Line {i+1}")[0]
                        for i,data in enumerate(plotdatas)]

        self.animate(lower_panel_init, lower_panel_update, name, **kwargs)

    def project (self, axis = 0):
        """
        Return an EvoDat1D object corresponding to the projection
        axis = 0 means project out x axis (i.e., project to y axis)
        axis = 1 means project out y axis
        """
        ed = EvoDat1D()

        ed.idof = self.idof
        ed.T = self.T
        ed.times = self.getTimes(force_numpy=True)

        if axis == 0:
            ed.name = self.name + "ProjY"
            ed.L = self.W
            ed.res = np.linalg.norm(self.getRes(force_numpy=True), axis=0)
        else:
            ed.name = self.name + "ProjX"
            ed.L = self.L
            ed.res = np.linalg.norm(self.getRes(force_numpy=True), axis=1)

        ed.idnorm = np.zeros(np.shape(ed.res))
        delnorms = np.linalg.norm(ed.res, axis=(0,1))
        ed.res /= delnorms
        ed.norms = self.getNorms(force_numpy=True) + np.log(delnorms)
        ed.idnorm = np.zeros(np.shape(ed.res))

        return ed

    def plot_normNgrowth (self, pt=None, refs = []):

        ptstr = ""

        if pt is None:
            ttimes = self.getTimes(force_numpy=True)
            tnorms = self.getNorms(force_numpy=True)
        elif -self.L <= pt[0] < self.L and -self.W <= pt[1] < self.W:
            ptnorm = np.linalg.norm(self.getRes(force_numpy=True)[pt[0],pt[1],:,:], axis=0)
            choose = ptnorm > 1e-5
            ttimes = np.array(self.getTimes(force_numpy=True))[choose]
            tnorms = self.getNorms(force_numpy=True)[choose] + np.log(ptnorm[choose])
            ptstr = "_pt{}".format(pt)
        else:
            raise Exception("pt {} out of range for L,W = ({},{})".format(pt, self.L, self.W))

        title = "L = {}, W = {}".format(self.L, self.W)
        name = self.name+"Growth"+"_"+ptstr
        EvoDat1D._plot_growth(ttimes, tnorms, name, title, refs)