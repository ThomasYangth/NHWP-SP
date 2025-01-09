import sys
sys.path.append("..")

from src.py.EvoPlotScripts import *
from src.py.HamLib import *

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


def run2Dmodel_exp_slope (model, L, W, T):

    print("##################################")
    print("NOW RUNNING:::")
    print(model.name)
    print("##################################")

    hL = int(L/2)
    hW = int(W/2)

    print("######### BULK EXPONENT #########")
    plot_pointG_2D_exponent(model, L, W, T, ip=(hL, hW), sel_mode="inst")
    print("######### EDGE EXPONENT #########")
    plot_pointG_2D_exponent(model, L, W, T, ip=(0, hW), sel_mode="inst")
    print("######### CORNER EXPONENT #########")
    plot_pointG_2D_exponent(model, L, W, T, ip=(0,0), sel_mode="inst")
    print("######### BULK SLOPE #########")
    plot_pointG_2D_tslope(model, L, W, T, ip=(hL, hW))
    print("######### EDGE SLOPE #########")
    plot_pointG_2D_tslope(model, L, W, T, ip=(0, hW))
    print("######### CORNER SLOPE #########")
    plot_pointG_2D_tslope(model, L, W, T, ip=(0,0))


def run2Dmodel_edge_eff (model, sz1, sz2, T, edges = ["x-"], kspan = 0.05, k = 0):

    print("##################################")
    print("NOW RUNNING:::")
    print(model.name)

    for edge in edges:
        print(f"######### {edge} EDGE EFFECTIVE THEORY #########")

        if edge.startswith("x"):
            L = sz1
            W = sz2
        else:
            L = sz2
            W = sz1
        plot_and_compare_2D_Edge(model, L, W, T, kspan=kspan, k=k, Ns=sz1, edge=edge, snapshots=[0, T/5, T*2/5, T*3/5, T*4/5, T])


def run2DModels():

    L = 200
    W = 80
    S = 120
    T = 50

    run2Dmodel_exp_slope(MODEL_2D_A, S, S, T)
    run2Dmodel_exp_slope(MODEL_2D_B, S, S, T)
    run2Dmodel_exp_slope(MODEL_2D_C, S, S, T)
    run2Dmodel_edge_eff(MODEL_2D_A, L, W, T, edges=["x-","x+","y-","y+"], kspan=10)
    run2Dmodel_edge_eff(MODEL_2D_B, L, W, T, edges=["x-","x+","y-","y+"], kspan=10)
    run2Dmodel_edge_eff(MODEL_2D_C, L, W, T, edges=["x-","x+","y-","y+"], kspan=10)