import sys
sys.path.append("..")

from src.py.EvoPlotScripts import *
from src.py.HamLib import *

def runX():

    model = GenMulti({2:0.0995338 - 1.4427j, 1:1.02306 + 0.577343j ,0:-1.14803 - 0.485631j, -1:-1.80091 + 0.196978j, -2:-0.0842039 - 1.83081j}, 1, "ModelX")
    plot_pointG_1D_tslope(model, 200, 50, ip=0, force_evolve=True)
    plot_pointG_1D_tslope(model, 200, 50, ip=-1, force_evolve=True)
    plot_pointG_1D_tslope(model, 200, 50, ip=100, force_evolve=True)

def runX_WF():
    model = GenMulti({2:0.0995338 - 1.4427j, 1:1.02306 + 0.577343j ,0:-1.14803 - 0.485631j, -1:-1.80091 + 0.196978j, -2:-0.0842039 - 1.83081j}, 1, "ModelX")
    plot_pointG_1D_WF(model, 200, 50, 0, 0, 10, iprad=5, takets=[0, 0.2, 0.5, 1, 3, 5, 10], force_evolve=True)

def runHN():

    model = GenMulti({1:1j, -1:1}, 1, "HN")
    plot_pointG_1D_tslope(model, 200, 50, ip=100, force_evolve=True)

def runModel3():
    model = GenMulti({1:-(0.0455 + 0.9305j),
          -1:0.3925 + 2.0955j,
          2:0.112 + 0.0955j,
          -2:-(0.088 - 1.0845j)}, 1, "Model3")
    plot_pointG_1D_WF(model, 200, 50, 0, 0, 10, iprad=3, takets=[0, 1, 3, 5, 10, 20, 30], force_evolve=True)

def getModel2A():

    ts0 = {
        -2:-0.194+0.316j,
        -1:-0.744-0.634j,
        0: 0.315-0.55j,
        1: 0.217+0.434j,
        2: -0.346-0.488j
    }

    tsm = {
        -1:-0.276-0.596j,
        0: 0.777+0.996j,
        1: -0.715-0.37j,
    }

    tsp = {
        -1:0.591-0.818j,
        0: 0.008+0.979j,
        1: -0.317+0.464j
    }

    return GenHN2D(tsp, ts0, tsm, "M2A")

def run2DEdge():

    model = getModel2A()

    sz1 = 500
    sz2 = 1

    plot_and_compare_2D_Edge(model, sz1, sz2, 50, kspan=0.05, k=0, force_evolve=True, Ns=sz1, edge="x+")

def run2Dtslope():

    model = getModel2A()
    sz1 = 50
    sz2 = 50
    plot_pointG_2D_tslope(model, sz1, sz2, 20, ip=(int(sz1/2), int(sz2/2)))
    plot_pointG_2D_tslope(model, sz1, sz2, 20, ip=(0, int(sz2/2)))
    plot_pointG_2D_tslope(model, sz1, sz2, 20, ip=(0,0))

def run2Dexpo():

    model = getModel2A()
    sz1 = 50
    sz2 = 50
    plot_pointG_2D_exponent(model, sz1, sz2, 20, ip=(int(sz1/2), int(sz2/2)), sel_mode="inst")
    plot_pointG_2D_exponent(model, sz1, sz2, 20, ip=(0, int(sz2/2)), sel_mode="inst")
    plot_pointG_2D_exponent(model, sz1, sz2, 20, ip=(0,0), sel_mode="inst")

def runExoticModel():
    ts = {0:(-0.752 - 0.922j), -1: -(0.214 - 0.03j), 1:(0.282 + 0.625j)}
    model = GenMulti(ts, 1, "from2D")
    plot_pointG_1D_WF(model, 200, 50, 0, 0, 10, iprad=5, takets=[0, 1, 5, 10, 20, 30, 50], force_evolve=True)
    plot_pointG_1D_WF(model, 200, 50, 1, 0, 10, iprad=5, takets=[0, 1, 5, 10, 20, 30, 50], force_evolve=True)

def run2DexpoC():

    model = MODEL_2D_C
    sz1 = 50
    sz2 = 50
    plot_pointG_2D_exponent(model, sz1, sz2, 20, ip=(int(sz1/2), int(sz2/2)), sel_mode="inst")
    plot_pointG_2D_exponent(model, sz1, sz2, 20, ip=(0, int(sz2/2)), sel_mode="inst")
    plot_pointG_2D_exponent(model, sz1, sz2, 20, ip=(0,0), sel_mode="inst")
    plot_pointG_2D_tslope(model, sz1, sz2, 20, ip=(int(sz1/2), int(sz2/2)))
    plot_pointG_2D_tslope(model, sz1, sz2, 20, ip=(0, int(sz2/2)))
    plot_pointG_2D_tslope(model, sz1, sz2, 20, ip=(0,0))

def run2DexpoD():

    model = MODEL_2D_D
    sz1 = 50
    sz2 = 50
    plot_pointG_2D_exponent(model, sz1, sz2, 20, ip=(int(sz1/2), int(sz2/2)), sel_mode="inst")
    plot_pointG_2D_exponent(model, sz1, sz2, 20, ip=(0, int(sz2/2)), sel_mode="inst")
    plot_pointG_2D_exponent(model, sz1, sz2, 20, ip=(0,0), sel_mode="inst")
    plot_pointG_2D_tslope(model, sz1, sz2, 20, ip=(int(sz1/2), int(sz2/2)))
    plot_pointG_2D_tslope(model, sz1, sz2, 20, ip=(0, int(sz2/2)))
    plot_pointG_2D_tslope(model, sz1, sz2, 20, ip=(0,0))

def run1Dmodel_everything (model, L, T, comp=1):

    print("##################################")
    print("NOW RUNNING:::")
    print(model.name)
    print("##################################")

    hL = int(L/2)
    model.OBCPBCSpec_1D()
    print("######### LEFT EXPONENT #########")
    plot_pointG_1D_exponent(model, L, T, ip=0, comp=comp)
    print("######### MIDDLE EXPONENT #########")
    plot_pointG_1D_exponent(model, L, T, ip=hL, comp=comp)
    print("######### RIGHT EXPONENT #########")
    plot_pointG_1D_exponent(model, L, T, ip=-1, comp=comp)
    print("######### LEFT SLOPE #########")
    plot_pointG_1D_tslope(model, L, T, ip=0)
    print("######### MIDDLE SLOPE #########")
    plot_pointG_1D_tslope(model, L, T, ip=hL)
    print("######### RIGHT SLOPE #########")
    plot_pointG_1D_tslope(model, L, T, ip=-1)
    if model.int_dim == 1:
        print("######### LEFT WAVE FUNCTION #########")
        plot_pointG_1D_WF(model, L, T, 0, 0, 10, iprad=5, takets = [0, T/10, T/5, T*0.4, T*0.6, T*0.8, T])
        print("######### RIGHT WAVE FUNCTION #########")
        plot_pointG_1D_WF(model, L, T, 1, 0, 10, iprad=5, takets = [0, T/10, T/5, T*0.4, T*0.6, T*0.8, T])
    else:
        print("######### MIDDLE SPIN VECTOR #########")
        plot_pointG_1D_vec(model, L, T, ip=hL, start_t=15, error_log=True)

def runModelc(depth = 0):

    L = 500
    T = 25

    #plot_pointG_1D_WF_left(MODEL_1D_C, L, T, 0, depth, 15, takets = [0, T/10, T/5, T*0.4, T*0.6, T*0.8, T], precision=20, force_evolve=True)
    #plot_pointG_1D_WF_left(MODEL_1D_C, L, T, 1, depth, 15, takets = [0, T/10, T/5, T*0.4, T*0.6, T*0.8, T], precision=20, force_evolve=True)

    #plot_pointG_1D_WF(MODEL_1D_C, L, T, 0, depth, 15, takets = [T/10, T/5, T*0.4, T*0.6, T*0.8, T], force_evolve=True)
    #plot_pointG_1D_WF(MODEL_1D_C, L, T, 1, depth, 15, takets = [T/10, T/5, T*0.4, T*0.6, T*0.8, T], force_evolve=True)

    #plot_pointG_1D_WF_onsite(MODEL_1D_A, L, T, 0, depth, 15, takets = [0, T/10, T/5, T*0.4, T*0.6, T*0.8, T])
    #plot_pointG_1D_WF_onsite(MODEL_1D_A, L, T, 1, depth, 15, takets = [0, T/10, T/5, T*0.4, T*0.6, T*0.8, T])

    for i in range(5):
        #plot_pointG_1D_tslope(MODEL_1D_A, L, T, ip=i)
        plot_pointG_1D_tslope(MODEL_1D_C, L, T, ip=i, iprad=0.0001, force_evolve=1, start_t=1, precision=20)
        plot_pointG_1D_tslope(MODEL_1D_C, L, T, ip=-(i+1), iprad=0.0001, force_evolve=1, start_t=1, precision=20)


def run1DModels():

    L = 500
    T = 30
    T2 = 50

    run1Dmodel_everything(MODEL_1D_A, L, T)
    MODEL_1D_Av1.OBCPBCSpec_1D()
    MODEL_1D_Av2.OBCPBCSpec_1D()

    run1Dmodel_everything(MODEL_1D_B, L, T, comp=2)
    run1Dmodel_everything(MODEL_1D_C, L, T)
    run1Dmodel_everything(MODEL_1D_D, L, T2)
    run1Dmodel_everything(MODEL_1D_Ev1, L, T)
    run1Dmodel_everything(MODEL_1D_Ev2, L, T)

def YaoWangSpecs():

    #YaoWang2B(1.5, 1.2, 0.2+0.2j, 0.5, name = "1Ev3").OBCPBCSpec_1D()
    #YaoWang2B(1.5, 1, 0.2-0.2j, 0.5, name = "1Ev4").OBCPBCSpec_1D()
    #YaoWang2B(1.5, 1, 0.2+0.2j, 0.2, name = "1Ev5").OBCPBCSpec_1D()
    model = YaoWang2B(1+1j, 1, 0.2+0.2j, 0.2, name = "1Ev6")
    run1Dmodel_everything(model, 50, 30)


def doHNHerm():
    ts = {1:1, -1:1}
    Ham = GenMulti(ts, 1, "HNH")
    L = 500
    plot_pointG_1D_tcompare(Ham, L, 30, ip=int(L/2))
    plot_pointG_1D_tcompare(Ham, L, 30, ip=0)

def do_model12():

    ts = {-2: (-0.1102-0.0173j),
           -1: (0.3189-0.3244j),
             1: (-0.1912-1.8998j),
               2: (0.3856+0.3460j)}

    Ham = GenMulti(ts,1,"my12")

    Ham.OBCPBCSpec_1D(obclen = 50)
    L = 500

    plot_pointG_1D_tcompare(Ham, L, 30, ip=int(L/2), start_t=2)
    plot_pointG_1D_tcompare(Ham, L, 30, ip=0, start_t=2)
    #plot_pointG_1D_tslope(Ham, L, 50, ip=2)
    #plot_pointG_1D_tslope(Ham, L, 50, ip=5)
    #plot_pointG_1D_tslope(Ham, L, 100, ip=10)
    #plot_pointG_1D_tslope(Ham, L, 30, ip=20)
    #plot_pointG_1D_tslope(Ham, L, 30, ip=50)

    #plot_pointG_1D_spsimp(Ham, L, 30, ip=int(L/2), sel_mode="inst", comp=1, start_t=5)
    #plot_pointG_1D_spsimp(Ham, L, 30, ip=0, sel_mode="inst", comp=1, start_t=5)
    #plot_pointG_1D_spsimp(Ham, L, 30, ip=L-1, sel_mode="inst", comp=1, start_t=5)

def do_model12_d1():

    ts = {-2: (-0.1102-0.0173j),
           -1: (0.3189-0.4244j),
             1: (-0.1912-1.8998j),
               2: (0.3856+0.3460j)}

    Ham = GenMulti(ts,1,"my12d1")

    Ham.OBCPBCSpec_1D(obclen = 50)
    L = 500

def do_model12_d2():

    ts = {-2: (-0.1102-0.0173j),
           -1: (0.3189-0.5244j),
             1: (-0.1912-1.8998j),
               2: (0.3856+0.3460j)}

    Ham = GenMulti(ts,1,"my12d2")

    Ham.OBCPBCSpec_1D(obclen = 50)
    L = 500

    
def do_very_comp_model():

    ts = {
        -3:np.array([[-0.276-0.596j,  0.008+0.979j], [ 0.796+0.607j, -0.816-0.601j]]),
        -2:np.array([[-0.194+0.316j,  0.777+0.996j], [-0.679-0.857j, -0.769+0.998j]]),
        -1:np.array([[-0.744-0.634j, -0.54 -0.474j], [-0.107+0.129j,  0.34 -0.109j]]),
        0:np.array([[ 0.315-0.55j , -0.715-0.37j ], [ 0.5  +0.816j, -0.282+0.583j]]),
        1:np.array([[ 0.217+0.434j,  0.367+0.554j], [-0.177+0.179j, -0.694-0.544j]]),
        2:np.array([[-0.346-0.488j, -0.317+0.464j], [-0.191-0.558j,  0.437+0.947j]]),
        3:np.array([[ 0.591-0.818j,  0.053+0.446j], [-0.857+0.755j, -0.074+0.907j]])
    }

    Ham = GenMulti(ts,2,"my5")

    #Ham.OBCPBCSpec_1D(obclen = 100)
    L = 500

    plot_pointG_1D_vec(Ham, L, 40, ip=int(L/2), error_log=True, start_t=15)
    #plot_pointG_1D_vec(Ham, L, 30, ip=0, error_log=True)

    #plot_pointG_1D_spsimp(Ham, L, 30, ip=int(L/2), sel_mode="inst", comp=1, start_t=5)
    #plot_pointG_1D_spsimp(Ham, L, 50, ip=0, sel_mode="inst", comp=1, start_t=5, ploterr=True)
    #plot_pointG_1D_spsimp(Ham, L, 50, ip=L-1, sel_mode="inst", comp=1, start_t=5)

def do6_YaoWang():

    model1 = YaoWang2B(0.5, 1, 0.2+0.2j, 0.5)
    model1.OBCPBCSpec_1D(obclen=100)

    model2 = YaoWang2B(1.5, 1, 0.2+0.2j, 0.5)
    model2.OBCPBCSpec_1D(obclen=100)

    L = 500

    plot_pointG_1D_spsimp(model1, L, 30, ip=int(L/2), sel_mode="inst", comp=1, start_t=5, ploterr=True)
    plot_pointG_1D_spsimp(model1, L, 50, ip=0, sel_mode="inst", comp=1, start_t=5, ploterr=True)
    plot_pointG_1D_spsimp(model1, L, 50, ip=L-1, sel_mode="inst", comp=1, start_t=5, ploterr=True)

    plot_pointG_1D_spsimp(model2, L, 30, ip=int(L/2), sel_mode="inst", comp=1, start_t=5, ploterr=True)
    plot_pointG_1D_spsimp(model2, L, 50, ip=0, sel_mode="inst", comp=1, start_t=5, ploterr=True)
    plot_pointG_1D_spsimp(model2, L, 50, ip=L-1, sel_mode="inst", comp=1, start_t=5, ploterr=True)

def do_model3():

    ts = {1:-(0.0455 + 0.9305j),
          -1:0.3925 + 2.0955j,
          2:0.112 + 0.0955j,
          -2:-(0.088 - 1.0845j)}
    #plot_pointG_1D(GenMulti(ts,1,"my3"), L, 40, force_evolve=True, ip=-1, plotre=False, alt=-0.294404)
    #plot_pointG_1D(GenMulti(ts,1,"my3"), L, 40, force_evolve=False, ip=-1, plotre=False)
    #plot_pointG_1D(GenMulti(ts,1,"my3"), L, 40, force_evolve=False, ip=-1, plotre=False, alt=-0.294404, sel_mode="half")

    Ham = GenMulti(ts,1,"my3")

    #Ham.OBCPBCSpec_1D(obclen = 50)
    L = 500

    #plot_pointG_1D_spsimp(Ham, L, 30, ip=int(L/2), sel_mode="inst", comp=1, start_t=5)
    plot_pointG_1D_spsimp(Ham, L, 30, ip=0, sel_mode="inst", comp=1, start_t=5)
    plot_pointG_1D_spsimp(Ham, L, 30, ip=L-1, sel_mode="inst", comp=1, start_t=5)

def do_model3_tslope():

    ts = {1:-(0.0455 + 0.9305j),
          -1:0.3925 + 2.0955j,
          2:0.112 + 0.0955j,
          -2:-(0.088 - 1.0845j)}
    #plot_pointG_1D(GenMulti(ts,1,"my3"), L, 40, force_evolve=True, ip=-1, plotre=False, alt=-0.294404)
    #plot_pointG_1D(GenMulti(ts,1,"my3"), L, 40, force_evolve=False, ip=-1, plotre=False)
    #plot_pointG_1D(GenMulti(ts,1,"my3"), L, 40, force_evolve=False, ip=-1, plotre=False, alt=-0.294404, sel_mode="half")

    Ham = GenMulti(ts,1,"my3")

    #Ham.OBCPBCSpec_1D(obclen = 50)
    L = 500

    #plot_pointG_1D_spsimp(Ham, L, 30, ip=int(L/2), sel_mode="inst", comp=1, start_t=5)
    plot_pointG_1D_tslope(Ham, L, 30, ip=0)

def get_oscillating_model_PH():

    ts = {
    1:-1.5j+0.01,
     2:-0.1j,
     -1:0.5j+0.01,
     -2:-0.1j}
    
    Ham = GenMulti(ts,1,"oscil_ph")
    #Ham.OBCPBCSpec_1D()
    
    L = 300
    T = 30

    plot_pointG_1D_spsimp(Ham, L, T, ip=0, sel_mode="inst", comp=2, start_t=5, iprad=0.1)
    #plot_pointG_1D_spsimp(Ham, L, 30, ip=int(L/2), sel_mode="inst", comp=2, start_t=5)