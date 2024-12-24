from ..py.EvoPlotScripts import plot_pointG_1D_tslope, plot_and_compare_2D_Edge
from ..py.HamLib import GenMulti, GenHN2D

def runX():

    model = GenMulti({2:0.0995338 - 1.4427j, 1:1.02306 + 0.577343j ,0:-1.14803 - 0.485631j, -1:-1.80091 + 0.196978j, -2:-0.0842039 - 1.83081j}, 1, "ModelX")
    plot_pointG_1D_tslope(model, 200, 50, ip=0, force_evolve=True)
    plot_pointG_1D_tslope(model, 200, 50, ip=-1, force_evolve=True)
    plot_pointG_1D_tslope(model, 200, 50, ip=100, force_evolve=True)

def runHN():

    model = GenMulti({1:1j, -1:1}, 1, "HN")
    plot_pointG_1D_tslope(model, 200, 50, ip=100, force_evolve=True)

def run2D():

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

    model = GenHN2D(tsp, ts0, tsm, "2Da")
    sz1 = 150
    sz2 = 70

    plot_and_compare_2D_Edge(model, sz1, sz2, 20, force_evolve=False, Ns=sz1)
