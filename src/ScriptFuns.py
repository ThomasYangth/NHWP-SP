import itertools
from os import mkdir
import re
import numpy as np
import sympy as sp
from math import floor, ceil
from numpy import transpose as Tp
import scipy
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from matplotlib import cm, animation
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider, Button
from random import random
from time import time
import scipy.integrate as si
#import scipy
from scipy.integrate import ode
from scipy import optimize
from os.path import isfile, isdir
from os import listdir
from HamClass.HamLib import ETI2DRot, ETI3D
from HamClass.HamClass import LatticeHam, EvoDat1D, EvoDat2D
from HamClass.WaveFuns import GaussianWave

# Rotated ETI 2D
def SlideRot (M, l, bc, B = [0,0], dens = 24, kdens = 72, KorIPR = 0, xlog = False, Sz = 0):
    thetas = [i/dens for i in range(dens+1)]
    name = "K"
    if KorIPR == 1:
        name = "IPR"
    elif KorIPR == 2:
        name = "BiIPR"
    dirname = "SlideRot_M{}_L{}_bc{}{}{}{}_{}".format(M, l, bc, "_XLog" if xlog else "", "_Sz{}".format(Sz) if Sz != 0 else "", "_B{}".format(B) if B != [0,0] else "", name)
    try:
        mkdir(dirname)
    except FileExistsError:
        dirnameo = dirname
        i = 1
        dirname = dirnameo + "_{}".format(i)
        while isdir(dirname):
            i = i + 1
            dirname = dirnameo + "_{}".format(i)
        mkdir(dirname)
    images = []
    for theta in thetas:
        E2D = ETI2DRot (M, l, bc, theta, B, Sz=Sz)
        E2D.name = dirname + "\\T{:.3f}".format(theta)
        if KorIPR == 1:
            E2D.complex_Spectrum_IPR(kdens=kdens)
        elif KorIPR == 2:
            E2D.complex_Spectrum_biIPR(kdens=kdens)
        else:
            E2D.complex_Spectrum_k(kdens=kdens, xlog = xlog)
        seekname = [dirname+"\\"+fn for fn in listdir(dirname) if fn.startswith("T{:.3f}".format(theta)) and fn.endswith("jpg")]
        if len(seekname) == 0:
            raise Exception("File not found at theta = {:.3f}".format(theta))
        elif len(seekname) > 1:
            print("Too name names found:")
            print(seekname)
            raise Exception("Too name names found at theta = {:.3f}".format(theta))
        images.append(mpimg.imread(seekname[0]))
    
    fig = plt.figure(figsize=(8,8))
    map_ax = plt.axes([0.1, 0.2, 0.9, 0.7])
    slider_ax = plt.axes([0.1, 0.02, 0.7, 0.05])
    bar = Slider(slider_ax, 'theta/2pi', 0, 1, valinit=0, valstep=1/dens)
    imshow = map_ax.imshow(images[0])
    
    def update (t, animate=False):
        tind = min(enumerate(thetas), key=lambda x:abs(x[1]-t))[0]
        imshow.set_data(images[tind])
        if animate:
            bar.eventson = False
            bar.set_val(t)
            bar.eventson = True
            return [imshow]
        else:
            fig.canvas.draw_idle()

    bar.on_changed(update)

    
    anim = animation.FuncAnimation(fig, lambda i: update(thetas[i], animate=True),
                            frames=len(thetas), interval=int(10000/len(thetas)), blit=True, repeat = False)
    plt.rcParams['animation.ffmpeg_path'] ='D:\\Downloads\\ffmpeg-5.0-essentials_build\\bin\\ffmpeg.exe'
    anim.save(dirname+"_Anim.mp4")
    plt.close()
    #plt.show()

def SlideSweep (ModelFun, para_range, para_step, para_name, save_name, kdens = 20, KorIPR = 0):
    plow, phigh = para_range
    paras = [plow+i*para_step for i in range(int((phigh-plow)/para_step)+1)]
    name = "K"
    if KorIPR == 1:
        name = "IPR"
    elif KorIPR == 2:
        name = "BiIPR"
    elif KorIPR == 3:
        name = "RB"
    dirname = save_name + "_" + name
    try:
        mkdir(dirname)
    except FileExistsError:
        dirnameo = dirname
        i = 1
        dirname = dirnameo + "_{}".format(i)
        while isdir(dirname):
            i = i + 1
            dirname = dirnameo + "_{}".format(i)
        mkdir(dirname)
    images = []
    for p in paras:
        Ham = ModelFun(p)
        Ham.name = dirname + "\\P{:.3f}".format(p)
        if KorIPR == 1:
            Ham.complex_Spectrum_IPR(kdens=kdens)
        elif KorIPR == 2:
            Ham.complex_Spectrum_biIPR(kdens=kdens)
        elif KorIPR == 3:
            Ham.plotRealBand(kdens=kdens)
        else:
            Ham.complex_Spectrum_k(kdens=kdens)
        seekname = [dirname+"\\"+fn for fn in listdir(dirname) if fn.startswith("P{:.3f}".format(p)) and fn.endswith("jpg")]
        if len(seekname) == 0:
            raise Exception("File not found at para = {:.3f}".format(p))
        elif len(seekname) > 1:
            print("Too many names found:")
            print(seekname)
            raise Exception("Too name names found at theta = {:.3f}".format(p))
        images.append(mpimg.imread(seekname[0]))
    
    fig = plt.figure(figsize=(8,8))
    map_ax = plt.axes([0.1, 0.2, 0.9, 0.7])
    slider_ax = plt.axes([0.1, 0.02, 0.7, 0.05])
    bar = Slider(slider_ax, para_name, plow, phigh, valinit=plow, valstep=para_step)
    imshow = map_ax.imshow(images[0])
    
    def update (p, animate=False):
        tind = min(enumerate(paras), key=lambda x:abs(x[1]-p))[0]
        imshow.set_data(images[tind])
        if animate:
            bar.eventson = False
            bar.set_val(p)
            bar.eventson = True
            return [imshow]
        else:
            fig.canvas.draw_idle()

    bar.on_changed(update)

    
    anim = animation.FuncAnimation(fig, lambda i: update(paras[i], animate=True),
                            frames=len(paras), interval=int(10000/len(paras)), blit=True, repeat = False)
    plt.rcParams['animation.ffmpeg_path'] ='D:\\Downloads\\ffmpeg-5.0-essentials_build\\bin\\ffmpeg.exe'
    anim.save(dirname+"_Anim.mp4")
    plt.close()

def ETI3DLoop (M, l, B, d, a, dir, ks, kdens = 50):
    LH = ETI3D(M, l, B, d, a, [0,0,0])
    Hk = LH.realize()
    Ers = []
    Eis = []
    cols = []
    cmap = cm.get_cmap('RdYlGn')
    for k in [2*np.pi*i/kdens for i in range(kdens)]:
        k1 = ks[:]
        k1.insert(dir-1, k)
        Hthis = Hk(k1)
        w, _ = np.linalg.eig(Hthis)
        for E in w:
            Ers.append(np.real(E))
            Eis.append(np.imag(E))
            cols.append(cmap(k/(2*np.pi)))
    plt.figure()
    plt.scatter(Ers, Eis, c=cols)
    plt.xlabel("Re(E)")
    plt.ylabel("Im(E)")
    plt.title("k{} Winding, for (k{}={}pi, k{}={}pi)".format(dir, 1+(dir==1), format(ks[0]/np.pi,".2f"), 3-(dir==3), format(ks[1]/np.pi,".2f")))
    cbar = plt.colorbar(mappable=cm.ScalarMappable(Normalize(0, 2*np.pi, True), cmap))
    cbar.set_label("k{}".format(dir))
    name = "k{} Winding, for (k{}={}pi, k{}={}pi)".format(dir, 1+(dir==1), format(ks[0]/np.pi,".2f"), 3-(dir==3), format(ks[1]/np.pi,".2f")) + "ETI3D(M{}_L{}_B{}_d{}_a{})".format(M,l,B,d,format(a/np.pi,".2f"))
    plt.savefig(name+".pdf")
    plt.savefig(name+".jpg")

def ETI3DLoopOBC (M, l, B, d, a, dir, ks, olen = 20):
    bc = [0,0]
    bc.insert(dir-1, olen)
    LH = ETI3D(M, l, B, d, a, bc)
    Hk = LH.realize()
    ReE = []
    ImE = []
    IPRs = []
    w, v = np.linalg.eig(Hk(ks))
    for i, E in enumerate(w):
        Ev = v[:,i]
        IPR1 = np.sum(np.abs(Ev[:int(np.size(v)/2)])**4)
        IPR2 = np.sum(np.abs(Ev[int(np.size(v)/2):])**4)
        ReE.append(np.real(E))
        ImE.append(np.imag(E))
        IPRs.append((IPR1+IPR2)*(1 if IPR1 > IPR2 else -1))
    plt.figure()
    cmap = cm.get_cmap('RdYlGn')
    imin = np.min(IPRs)
    imax = np.max(IPRs)
    IPRs = (np.array(IPRs)-np.min(IPRs))/(max(np.max(IPRs)-np.min(IPRs),1e-3))
    cols = [cmap(ipr) for ipr in IPRs]
    plt.scatter(ReE, ImE, c=cols)
    plt.xlabel("Re(E)")
    plt.ylabel("Im(E)")
    plt.title("k{} OBC, for (k{}={}pi, k{}={}pi)".format(dir, 1+(dir==1), format(ks[0]/np.pi,".2f"), 3-(dir==3), format(ks[1]/np.pi,".2f")))
    cbar = plt.colorbar(mappable=cm.ScalarMappable(Normalize(imin, imax, True), cmap))
    cbar.set_label("IPR")
    name = "k{} OBC, for (k{}={}pi, k{}={}pi)".format(dir, 1+(dir==1), format(ks[0]/np.pi,".2f"), 3-(dir==3), format(ks[1]/np.pi,".2f")) + "ETI3D(M{}_L{}_B{}_d{}_a{})".format(M,l,B,d,format(a/np.pi,".2f"))
    plt.savefig(name+".pdf")
    plt.savefig(name+".jpg")

def ETI3DPvO (M, l, B, d, a, dir, ks):
    ETI3DLoopOBC(M, l, B, d, a, dir, ks)
    ETI3DLoop(M, l, B, d, a, dir, ks)

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

def Chern2D (Ham, bcs, intDOF = 1, interdens = 20):
    H = Ham.realize()
    pbcdims = [i for i, x in enumerate(bcs) if x == 0]
    deriHs = [Ham.realize(kders=getIns([0],d,1)) for d in pbcdims]
    """
    X = positionOP(obcdims,0,intDOF)
    Y = positionOP(obcdims,1,intDOF)
    Z = positionOP(obcdims,2,intDOF)
    Hx = np.dot(X,H)-np.dot(H,X)
    Hy = np.dot(Y,H)-np.dot(H,Y)
    Hz = np.dot(Z,H)-np.dot(H,Z)
    """
    if len(pbcdims) == 0:
        Hi = np.linalg.inv(H)
        Hx = Ham.realize(xcmts=[1,0])/bcs[0]
        Hy = Ham.realize(xcmts=[0,1])/bcs[1]
        return 1j*np.trace(recDot([Hi, Hx, Hi, Hy]))
    else:
        def integrand(*ks):
        #for ks in itertools.product(*[[i*2*np.pi/interdens for i in range(interdens)]]*len(pbcdims)):
            ks = list(ks)
            Ht = H(ks)
            Hi = np.linalg.inv(Ht)
            m1 = 1
            for i in [0,1]:
                m1 = np.dot(m1, Hi)
                try:
                    ind = pbcdims.index(i)
                    m1 = np.dot(m1, deriHs[ind](ks))
                except ValueError:
                    m1 = np.dot(m1, (-1j)*Ham.realize(xcmts=getIns([0],i,1))(ks)/bcs[i])
            return np.trace(m1)*(-1/(16*np.pi**2))#*((2*np.pi/interdens)**(len(pbcdims)))
        # Use standard integration
        return MCIntegrate(integrand, len(pbcdims), -np.pi, np.pi)

def Chern3D (Ham, bcs, intDOF = 1, interdens = 20):
    H = Ham.realize()
    pbcdims = [i for i, x in enumerate(bcs) if x == 0]
    deriHs = [Ham.realize(kders=getIns([0,0],d,1)) for d in pbcdims]
    """
    X = positionOP(obcdims,0,intDOF)
    Y = positionOP(obcdims,1,intDOF)
    Z = positionOP(obcdims,2,intDOF)
    Hx = np.dot(X,H)-np.dot(H,X)
    Hy = np.dot(Y,H)-np.dot(H,Y)
    Hz = np.dot(Z,H)-np.dot(H,Z)
    """
    if len(pbcdims) == 0:
        Hi = np.linalg.inv(H)
        Hx = Ham.realize(xcmts=[1,0,0])/bcs[0]
        Hy = Ham.realize(xcmts=[0,1,0])/bcs[1]
        Hz = Ham.realize(xcmts=[0,0,1])/bcs[2]
        return 1j*np.trace(recDot([Hi, Hx, Hi, Hy, Hi, Hz])-recDot([Hi, Hx, Hi, Hz, Hi, Hy]))
    else:
        def integrand(*ks):
        #for ks in itertools.product(*[[i*2*np.pi/interdens for i in range(interdens)]]*len(pbcdims)):
            ks = list(ks)
            Ht = H(ks)
            Hi = np.linalg.inv(Ht)
            m1 = 1
            for i in [0,1,2]:
                m1 = np.dot(m1, Hi)
                try:
                    ind = pbcdims.index(i)
                    m1 = np.dot(m1, deriHs[ind](ks))
                except ValueError:
                    m1 = np.dot(m1, (-1j)*Ham.realize(xcmts=getIns([0,0],i,1))(ks)/bcs[i])
            m2 = 1
            for i in [0,2,1]:
                m2 = np.dot(m2, Hi)
                try:
                    ind = pbcdims.index(i)
                    m2 = np.dot(m2, deriHs[ind](ks))
                except ValueError:
                    m2 = np.dot(m2, (-1j)*Ham.realize(xcmts=getIns([0,0],i,1))(ks)/bcs[i])
            return np.trace(m1-m2)*(-1/(8*np.pi**2))#*((2*np.pi/interdens)**(len(pbcdims)))
        # Use standard integration
        return MCIntegrate(integrand, len(pbcdims), -np.pi, np.pi)
        """
        if len(pbcdims) == 1:
            return si.quad(integrand, -np.pi. np.pi)
        elif len(pbcdims) == 2:
            return si.dblquad(integrand, -np.pi, np.pi, -np.pi, np.pi)
        elif len(pbcdims) == 3:
            return si.tplquad(integrand, -np.pi, np.pi, -np.pi, np.pi, -np.pi, np.pi)
        else:
            raise Exception("Too many integration dimensions!")
        """

def fitWP (Ham, GaussInit, T, dT, fit_axes, savename, res = None, method=0):
    times = np.array([dT*i for i in range(floor(T/dT)+1)])
    bc = Ham.bcs
    xss = [[i for i in range(d)] for d in bc]
    if res is None:
        _, res = Ham.time_evolve(T, dT, GaussInit)
    onsite_amps = np.linalg.norm(res, axis=len(bc))**2
    xs_mesh = np.meshgrid(*xss, indexing="ij")
    xs_mesh_flat = [a.flatten() for a in xs_mesh]
    test_mesh = np.ones(bc)
    norm_factor = np.sum((test_mesh.T*onsite_amps.T).T, axis=tuple([i for i in range(len(bc))]))
    if method == 0:
        avg_xs = [np.sum((mesh.T*onsite_amps.T).T, axis=tuple([i for i in range(len(bc))]))/norm_factor for mesh in xs_mesh]
    else:
        amps_reshape = np.reshape(onsite_amps, [np.prod(bc),len(times)])
        maxinds = np.argmax(amps_reshape, axis=0)
        max_xs = [[xs_mesh_flat[i][ind] for ind in maxinds] for i in range(len(bc))]
        avg_xs = max_xs
    avg_f = avg_xs[fit_axes-1]
    v = np.polyfit(times, avg_f, deg=1)[0]
    if method == 0:
        fw = open(savename+"_Avg.txt", 'w+')
    else:
        fw = open(savename+"_Max.txt", 'w+')
    fw.write("WavePacket Fit\n")
    fw.write("Fit Velocity: {}\n".format(v))
    fw.write("t\t"+"".join(["x{}\t".format(i+1) for i in range(len(bc))])+"\n")
    for i in range(len(times)):
        fw.write("{}\t".format(times[i]))
        fw.write("".join(["{}\t".format(avg_xs[j][i]) for j in range(len(bc))])+"\n")
    fw.close()
    plt.figure()
    cmap = plt.get_cmap("viridis")
    plt.scatter(avg_xs[0],avg_xs[1], c=cmap(times/T))
    plt.colorbar(mappable=cm.ScalarMappable(Normalize(0, T, True), cmap))
    plt.title("Center Trajectory vs T (Color)")
    if method == 0:
        plt.savefig(savename+"_Avg.jpg")
    else:
        plt.savefig(savename+"_Max.jpg")
    plt.close() 
    return v

def select_data (datas, norms, axis=0, sellen=10, left=True):
    dshape = np.shape(datas)
    T = dshape[-1]
    bc = dshape[:-2]
    datas = np.linalg.norm(datas, axis=-2)
    take_axis_len = bc[axis]
    if sellen >= take_axis_len:
        print("Select_data: Axis {} original length {} larger than intended selection length {}.".format(axis, take_axis_len, sellen))
        return datas, norms
    if left:
        indices = [i for i in range(sellen)]
    else:
        indices = [i for i in range(take_axis_len-sellen, take_axis_len)]
    newbc = np.copy(bc)
    newbc[axis] = sellen
    datasel = np.take(datas, axis=axis+1, indices=indices)
    sel_norms = np.linalg.norm(np.reshape(datasel, [np.product(newbc),T]), axis=1)
    norms += np.log(sel_norms)
    return np.transpose(np.transpose(datasel)/sel_norms), norms, newbc

def fitWPSel (Ham, GaussInit, T, dT, fit_axes, sel_axis, sel_len, sel_dir, savename, res = None, norms = None, method=0):
    times = np.array([dT*i for i in range(floor(T/dT)+1)])
    if res is None or norms is None:
        norms, res = Ham.time_evolve(T, dT, GaussInit)
    
    res, norms, bc = select_data(res, norms, axis=sel_axis, sellen=sel_len, left=sel_dir)
    xss = [[i for i in range(d)] for d in bc]
    xs_mesh = np.meshgrid(*xss, indexing="ij")
    xs_mesh_flat = [a.flatten() for a in xs_mesh]
    test_mesh = np.ones(bc)
    onsite_amps = res**2
    if method == 0:
        avg_xs = [np.sum(onsite_amps*mesh, axis=tuple([i+1 for i in range(len(bc))])) for mesh in xs_mesh]
    else:
        amps_reshape = np.reshape(onsite_amps, [len(times),np.prod(bc)])
        maxinds = np.argmax(amps_reshape, axis=1)
        max_xs = [[xs_mesh_flat[i][ind] for ind in maxinds] for i in range(len(bc))]
        avg_xs = max_xs
    avg_f = avg_xs[fit_axes-1]
    v = np.polyfit(times, avg_f, deg=1)[0]
    if method == 0:
        fw = open(savename+"_Avg.txt", 'w+')
    else:
        fw = open(savename+"_Max.txt", 'w+')
    fw.write("WavePacket Fit\n")
    fw.write("Fit Velocity: {}\n".format(v))
    fw.write("t\t"+"".join(["x{}\t".format(i+1) for i in range(len(bc))])+"\n")
    for i in range(len(times)):
        fw.write("{}\t".format(times[i]))
        fw.write("".join(["{}\t".format(avg_xs[j][i]) for j in range(len(bc))])+"\n")
    fw.close()
    plt.figure()
    cmap = plt.get_cmap("viridis")
    plt.scatter(avg_xs[0],avg_xs[1], c=cmap(times/T))
    plt.colorbar(mappable=cm.ScalarMappable(Normalize(0, T, True), cmap))
    plt.title("Center Trajectory vs T (Color)")
    if method == 0:
        plt.savefig(savename+"_Avg.jpg")
    else:
        plt.savefig(savename+"_Max.jpg")
    plt.close()

    plt.figure()
    norms_der = np.gradient(norms)/dT
    plt.plot(times, norms_der)
    plt.xlabel("t")
    plt.ylabel("Growth Rate")
    plt.title("Growth Rate of Restricted Wave Packet")
    plt.savefig(savename+"_Growth.jpg")
    plt.close()

    return v

def my_minimize (fun, k0, bounds, tol=1e-5):
    minF = fun(k0)
    minK = k0

    Bsizes = [b[1]-b[0] for b in bounds]

    if np.min(Bsizes) < tol:
        return k0

    mesh_size = 10
    for ks in itertools.product(*[[b[0]+(b[1]-b[0])*i/mesh_size for i in range(mesh_size+1)] for b in bounds]):
        fval = fun(ks)
        if fval < minF:
            minF = fval
            minK = ks
    return my_minimize (fun, minK, [(minK[i]-Bsizes[i]/mesh_size, minK[i]+Bsizes[i]/mesh_size) for i in range(len(Bsizes))], tol)

def reshape_datas (datas, norms, sel_axis=0, left=True, range=10):
    dshape = np.shape(datas)
    T = dshape[1]
    bcs = dshape[2:]
    datas = np.linalg.norm(datas, axis=1)
    #datas_sel = np.take(datas,  , axis=sel_axis+1)

def plotTimeProfile (HamF, Hamname, L, T, dT=0.01, left=True, region_sel = None):
    #T = 100
    #dT = 0.01
    Ham = HamF([L])
    if left:
        init = Ham.GaussianWave((0,), 5, (0,))
        name_addition = "Left"
    else:
        #Ham = WTXModel(-0.8j, 1.2j, 0.05j, 0.35j, 0.35, [L])
        init = Ham.GaussianWave((L-1,), 5, (0,))
        name_addition = "Right"
    times = [dT*i for i in range(floor(T/dT)+1)]
    norms, res = Ham.time_evolve(T, dT, init)
    if region_sel is not None:
        res = res[region_sel[0]:region_sel[1],:,:]
        name_addition += "_Sel{}".format(region_sel)
        add_norms = np.linalg.norm(res[:,0,:], axis=0)
        norms += np.log(add_norms)
        res /= add_norms
    plt.figure()
    plt.plot(times, norms)
    plt.xlabel("t")
    plt.ylabel("Norm")
    plt.title("L = {}".format(L))
    plt.savefig("{}Evolve_{}_L{}_T{}.jpg".format(Hamname, name_addition, L, T))
    plt.close()
    ngrowth = np.gradient(norms)/dT
    plt.figure()
    plt.plot(times, ngrowth)
    plt.xlabel("t")
    plt.ylabel("Growth Rate")
    plt.title("L = {}".format(L))
    plt.savefig("{}Growth_{}_L{}_T{}.jpg".format(Hamname, name_addition, L, T))
    plt.close()
    LatticeHam._plot_evo_1D(res, times, norms, savename="{}Profile_{}_L{}_T{}".format(Hamname, name_addition, L, T))

def evolve_1D_and_save (HamF, Hamname, L, T, dT=0.01):
    Ham = HamF([L])
    init = Ham.GaussianWave((int(L/2),), 5, (0,))
    evodat = EvoDat1D(Ham, T, dT, init, "{}_L{}_T{}".format(Hamname, L, T))
    evodat.save()
    return evodat

def plot_and_fit_1D (HamF, Hamname, L, T, ks=[0], initpos = 0.5, dT=0.01, force_evolve = False, VKG = None):
    Ham = HamF([L])
    if VKG is None:
        VKG = Ham.VKGMMA()
    spg, v, mxg = VKG
    for k in ks:
        init = Ham.GaussianWave((int(L*initpos),), 5, [(k,)])
        FName = "{}_L{}_T{}_k{}".format(Hamname, L, T, k)
        if initpos != 0.5:
            FName += "_ip{:.2f}".format(initpos)
        try:
            if force_evolve:
                raise Exception()
            evodat = EvoDat1D.read(FName+".txt")
        except Exception as e:
            evodat = EvoDat1D.from_evolve(Ham, T, dT, init, FName, takedT = 0.1)
            evodat.save()
        evodat.plot_normNgrowth(refg = [spg, mxg])
        evodat.plot_normNgrowth(pt=2, refg = [spg, mxg])
        evodat.plot_normNgrowth(pt=-3, refg = [spg, mxg])
        evodat.plot_profile()
        evodat.plot_xNv(refv = v)

def plot_and_fit_1D_Edge (HamF, Hamname, L, T, OBCL = None, dT=0.01, force_evolve = False, output=print, must_spec = 0, force_ip = None, sps = None, k = None, x0 = 5):
    if OBCL is None:
        OBCL = L
    Ham = HamF([OBCL])
    FName = "{}_SP_L{}_T{}".format(Hamname, L, T)
    if k is not None:
        FName += "_{}".format(k)
    if x0 is not None:
        FName += "_w{}".format(x0)
    Ham.name = FName
    im, sp, lgim, lgsp = Ham.max_SP_GBZ (0, output, must_spec = must_spec, L = OBCL, sps = sps)
    Ham.bcs = [L]
    if sp > 1:
        initpos = 3
    elif sp < 1:
        initpos = L-4
    else:
        raise Exception("No Skin Effect in this model!")
    if force_ip is not None:
        initpos = force_ip
    if k is None:
        k0 = 0
    elif k == "SPL":
        k0 = np.imag(np.log(lgsp[0]))
    elif k == "SPV":
        k0 = np.imag(np.log(sp))
    else:
        k0 = 0
    init = Ham.GaussianWave((initpos,), x0, [(0,)])
    try:
        if force_evolve:
            raise Exception()
        evodat = EvoDat1D.read(FName+".txt")
    except Exception as e:
        evodat = EvoDat1D.from_evolve(Ham, T, dT, init, FName, takedT = 0.1)
        evodat.save()
    maxiter = 5000
    while True:
        try:
            GBZspec,_ = scipy.sparse.linalg.eigs(scipy.sparse.csr_matrix(HamF([100]).realize()), which="LI", k=1, maxiter = maxiter)
            GBZtop = np.imag(GBZspec)
            break
        except:
            maxiter *= 2
        if maxiter > 1e8:
            print("Too many iterations! Switching to normal diagonalization")
            w,_ = np.linalg.eig(HamF([100]).realize())
            GBZtop = np.max(np.imag(w))
    
    evodat.plot_normNgrowth(ref =[([im], "Legal SP"), (lgim, "Illegal SP"), (GBZtop, "GBZ Top")])
    evodat.plot_normNgrowth(pt=initpos, ref =[([im], "Legal SP"), (lgim, "Illegal SP"), (GBZtop, "GBZ Top")])
    evodat.plot_profile()

def plot_velocities_1D (Ham:LatticeHam, L, T, dT=0.01, force_evolve = False, output=print, ip = None, comp=False):
    Ham.bcs = [L]
    Hamname = Ham.name
    FName = "{}_Vel_L{}_T{}".format(Hamname, L, T)
    if ip is None:
        ip = int(L/2)
    else:
        FName += "_ip{}".format(ip)
        if ip < 0:
            ip = L-ip
    init = Ham.GaussianWave((ip,), 1, [(0,)])
    tkdT = 0.1
    try:
        if force_evolve:
            raise Exception()
        evodat = EvoDat1D.read(FName+".txt")
    except Exception:
        evodat = EvoDat1D.from_evolve(Ham.time_evolve, T, dT, init, FName, takedT = tkdT)
        evodat.save()

    evodat.plot_profile()
    allnorms = np.linalg.norm(evodat.res,axis=1)
    lastprof = np.log(allnorms[:,-1])+evodat.norms[-1]
    midind = int(T/(2*tkdT))
    midprof = np.log(allnorms[:,midind])+evodat.norms[midind]
    lastprof = lastprof[::2]
    midprof = midprof[int(ip/2):int(ip/2)+len(lastprof)]
    lyaps = (lastprof-midprof+np.log(2))/(T/2)
    vs = (2*np.array([i for i in range(len(lastprof))])-ip)/T
    if comp:
        prec = 20
        t1 = time()
        mmalyaps = Ham.GrowthsMMA(vs[0],vs[-1],prec)
        print("Mathematica calculation time: {}s".format(time()-t1))
    plt.figure()
    plt.plot(vs, lyaps)
    if comp:
        plt.plot([((prec-i)*vs[0]+i*vs[-1])/prec for i in range(prec+1)], mmalyaps)
    plt.xlabel("v")
    plt.ylabel("lambda")
    if comp:
        plt.legend(["Numerical", "Theory"])
    plt.title("Lyapnov exponents fitted at T={}, L={}".format(T, L))
    plt.savefig(FName+".jpg")
    plt.close()

def plot_and_fit_2D_Edge (HamF, Hamname, L, W, T, Tplot=None, edge="x-", k = (0,0), dT=0.2, force_evolve = False):

    tkdT = 0.25

    Ham = HamF([L,W])
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

def plot_velocities_2D (Ham:LatticeHam, L, W, T, Tplot=None, ip = None, dT=0.2, force_evolve = False):

    tkdT = 0.25
    Ham.bcs = [L,W]

    FName = "{}_L{}W{}_T{}".format(Ham.name, L, W, T)

    if ip is None:
        ip = (int(L/2), int(W/2))
    else:
        FName += "_{}".format(ip)
    
    init = Ham.GaussianWave(ip, 1)

    try:
        if force_evolve:
            raise Exception()
        evodat = EvoDat2D.read(FName+".txt")
    except Exception as e:
        evodat = EvoDat2D.from_evolve(Ham, L, W, T, dT, init, FName, takedT=tkdT)
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

    vxs = (2*np.array([i for i in range(len(datsp[0]))])-ip[0])/T
    vys = (2*np.array([i for i in range(len(datsp[1]))])-ip[1])/T
    
    plt.figure()
    
    plt.plot(vs, lyaps)
    plt.xlabel("v")
    plt.ylabel("lambda")
    plt.title("Lyapunov exponents fitted on edge {} at T={}".format(edge, T))
    plt.savefig(FName+".jpg")
    plt.close()

def plot_and_fit_2D (HamF, Hamname, L, W, T, initpos = (0.5,0.5), ks=[(0,0)], kdir=0, dT=0.01, force_evolve = False, VKG = None):
    Ham = HamF(L,W)
    for k in ks:
        init = Ham.GaussianWave((int(L*initpos[0]),int(W*initpos[1])), 5, k)
        FName = "{}_k{}_L{}W{}_T{}".format(Hamname, k, L, W, T)
        if initpos != (0.5,0.5):
            FName += "_ip{}".format(initpos)
        try:
            if force_evolve:
                raise Exception()
            evodat = EvoDat2D.read(FName+".txt")
        except Exception as e:
            evodat = EvoDat2D.from_evolve(Ham, L, W, T, dT, init, FName, takedT=1)
            evodat.save()
        evodat.animate()
        evodat.plot_profile(0)
        evodat.plot_profile(1)
        evodat.plot_normNgrowth()

def plotfun (fun, pt, val, dens, pdens, savename, rng = None):
    x0, y0 = pt
    if val < fun(x0,y0)-0.1:
        print("Saddle point {:.3f}+{:.3f}i, with lambda {:.3f} is not even the largest on its own z!".format(x0,y0,val))
        return
    up = max(0.5, 2*abs(val))
    dn = -up
    if rng is None:
        rng = 2*np.sqrt(x0**2+y0**2)
    step = 2*rng/dens
    sstp = step/pdens
    xs = np.array([-rng+step*i for i in range(dens+1)])
    ys = np.array([-rng+step*i for i in range(dens+1)])
    zs = np.maximum(np.minimum(np.array([fun(x,y) for y in ys for x in xs]),up),dn).reshape(dens+1,dens+1)
    print("Saddle point {:.3f}+{:.3f}i done zs construction".format(x0,y0))

    bdx = []
    bdy = []
    for i in range(dens):
        for j in range(dens):
            if len(set([np.sign(zs[j,i]-val),np.sign(zs[j+1,i]-val),np.sign(zs[j,i+1]-val),np.sign(zs[j+1,i+1]-val)])) > 1:
                #print("Searching in block: {},{}".format(xs[i],ys[j]))
                #print("The four signs: {}".format([np.sign(zs[j,i]-val),np.sign(zs[j+1,i]-val),np.sign(zs[j,i+1]-val),np.sign(zs[j+1,i+1]-val)]))
                tzs = np.array([fun(xs[i]+kx*sstp,ys[j]+ky*sstp)-val for kx in range(pdens+1) for ky in range(pdens+1)]).reshape(pdens+1,pdens+1)
                for k in range(pdens+1):
                    #print("k = {}".format(k))
                    tx = xs[i]+sstp*k
                    #print("Two points on x direction: {:.2f}, {:.2f}".format(fun(tx,ys[j])-val, fun(tx,ys[j+1])-val))
                    if tzs[k,0]*tzs[k,-1] <= 0:
                        ty = ys[j]+sstp*(np.abs(tzs[k,:]).flatten().argmin())
                        bdx.append(tx)
                        bdy.append(ty)
                        #print("Got point from x direction: {},{}".format(tx,ty))
                    ty = ys[j]+sstp*k
                    #print("Two points on y direction: {:.2f}, {:.2f}".format(fun(xs[i],ty)-val, fun(xs[i+1],ty)-val))
                    if tzs[0,k]*tzs[-1,k] <= 0:
                        tx = xs[i]+sstp*(np.abs(tzs[:,k]).flatten().argmin())
                        bdx.append(tx)
                        bdy.append(ty)
                        #print("Got point from y direction: {},{}".format(tx,ty))
    
    print("Saddle point {:.3f}+{:.3f}i done bzd construction".format(x0,y0))

    plt.figure()
    plt.imshow(zs[::-1,:], interpolation="bilinear", extent=[-rng,rng,-rng,rng])
    plt.colorbar()
    plt.scatter(bdx, bdy, 5, c="C1")
    plt.scatter([0,x0], [0,y0], c="C2")
    plt.title("f_{:.3f}".format(val))
    plt.savefig(savename + "_Profile.jpg")

    nzs = (zs<val).astype(int)
    plt.figure()
    plt.contourf(xs, ys, nzs, [-1,0,1], colors=["white","cyan"])
    plt.scatter(bdx, bdy, 5, c="C1")
    plt.scatter([0,x0], [0,y0], c="C0")
    plt.title("f < {:.2f}".format(val))
    plt.savefig(savename + "_Region.jpg")

def analyze_1D_saddlepoints (Ham, Hamname):
    spz = Ham.zs[0]
    l = sp.symbols("l")

    def sing (expr):
        return expr.has(sp.oo, -sp.oo, sp.zoo, sp.nan)

    def cancelneg (expr, var):
        while sing(expr.subs(var, 0)):
            expr = sp.expand(expr*var)
        return expr
    
    def solve_sp (expr, var):
        eq = sp.poly_from_expr(expr, var)[0]
        coeffs = [complex(z) for z in eq.all_coeffs()]
        return np.roots(coeffs)
    
    char0 = Ham.Hmat.charpoly(l).as_expr()
    char = cancelneg(char0, spz)
    charp = cancelneg(sp.diff(char0, spz), spz)
    sps = solve_sp(sp.polys.resultant(char, charp, spz), l)
    print(sps)
    splist = []
    for Sp in sps:
        zs = solve_sp(char.subs(l, Sp), spz)
        newzs = []
        for z in zs:
            if np.abs(complex(charp.subs([(spz, z), (l, Sp)]))) > 1e-5:
                continue
            if len(newzs)>0 and np.min(np.abs(newzs-z)) < 1e-5:
                continue
            newzs.append(z)
        for z in newzs:
            splist.append((z, Sp))
    print(splist)
    for z,s in splist:
        plotfun(lambda x,y:10 if x==0 and y==0 else np.max(np.imag(solve_sp(char.subs(spz, x+1j*y), l))), (np.real(z),np.imag(z)), np.imag(s), 30, 5, Hamname+"_s={0.imag:.3f}".format(s))