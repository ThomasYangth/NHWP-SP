from .HamClass import LatticeHam, LatticeTerm, HamModel
import numpy as np

def Gen1D (ts:dict, bands = 1, name="", ran = None):
    """
    Generates a 1D model.

    Parameters:
        ts: dict
            A directionary of {hopping: amplitude}. For example, {1:2j, -1:-3} is the model H(z)=2j*z-3/z.
        bands: int
            Number of bands, default 1. If 1-band, "amplitudes" in ts should be floats; otherwise, should be bands x bands matrices.
        name: string
            Name of the model.
        ran: None
            If given, add an on-site random potential of amplitude ran to the model. "None" means no random potential.

    Returns:
        HamModel
    """

    if name == "":
        name = f"Unnamed1D{bands}B"

    term = []
    for i in ts.keys():
        if i == 0 and ran is not None:
            term.append((ts[0],"",ran)) # Random on-site potential
        else:
            term.append((ts[i],"exp({}k)".format(i)))
    return HamModel(1, bands, LatticeTerm.parseList(term), name)

def Gen2D (tss:dict, name="", bands = 1):
    """
    Generates a 2D model.

    Parameters:
        ts: dict
            A directionary of dictionaries, with the structure {y-hopping: {x-hopping: amplitude}}.
            For example, {-1:{1:2j}} is the model H(z_x,z_y)=2j*z_x/z_y.
        bands: int
            Number of bands, default 1. If 1-band, "amplitudes" in ts should be floats; otherwise, should be bands x bands matrices.
        name: string
            Name of the model.

    Returns:
        HamModel
    """

    if name == "":
        name = f"Unnamed2D{bands}B"

    term = []
    for yhop in tss.keys():
        xterms = tss[yhop]
        for xhop in xterms.keys():
            term.append((xterms[xhop], (f"exp({xhop}kx)" if xhop!=0 else "") + (f"exp({yhop}ky)" if yhop!=0 else "")))

    return HamModel(2, bands, LatticeTerm.parseList(term), name)

MODEL_1D_HN = Gen1D({1:1j, -1:1}, name="1HN")
MODEL_1D_A = Gen1D({-2: (-0.1102-0.0173j), -1: (0.3189-0.3244j), 1: (-0.1912-1.8998j), 2: (0.3856+0.3460j)}, name="1A")
MODEL_1D_Av1 = Gen1D({-2: (-0.1102-0.0173j), -1: (0.3189-0.4244j), 1: (-0.1912-1.8998j), 2: (0.3856+0.3460j)}, name="1Av1")
MODEL_1D_Av2 = Gen1D({-2: (-0.1102-0.0173j), -1: (0.3189-0.5244j), 1: (-0.1912-1.8998j), 2: (0.3856+0.3460j)}, name="1Av2")
MODEL_1D_B = Gen1D({1:-1.5j+0.01, 2:-0.1j, -1:0.5j+0.01, -2:-0.1j}, name="1B")
MODEL_1D_C = Gen1D({1:-(0.0455 + 0.9305j), -1:0.3925 + 2.0955j, 2:0.112 + 0.0955j, -2:-(0.088 - 1.0845j)}, name="1C")
MODEL_1D_D = Gen1D({
    -3:np.array([[-0.276-0.596j,  0.008+0.979j], [ 0.796+0.607j, -0.816-0.601j]]),
    -2:np.array([[-0.194+0.316j,  0.777+0.996j], [-0.679-0.857j, -0.769+0.998j]]),
    -1:np.array([[-0.744-0.634j, -0.54 -0.474j], [-0.107+0.129j,  0.34 -0.109j]]),
    0:np.array([[ 0.315-0.55j , -0.715-0.37j ], [ 0.5  +0.816j, -0.282+0.583j]]),
    1:np.array([[ 0.217+0.434j,  0.367+0.554j], [-0.177+0.179j, -0.694-0.544j]]),
    2:np.array([[-0.346-0.488j, -0.317+0.464j], [-0.191-0.558j,  0.437+0.947j]]),
    3:np.array([[ 0.591-0.818j,  0.053+0.446j], [-0.857+0.755j, -0.074+0.907j]])
        }, bands=2, name="1D")

# The classical SSH-ish model proposed by S-Y Yao and Z Wang, which has a topological transition
def YaoWang2B (t1, t2, t3, g, name=""):
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    return HamModel(1, 2, LatticeTerm.parseList([(t1*sx,""),((t2+t3)*sx,"cos(k)"),((t2-t3)*sy,"sin(k)"),(0.5j*g*sy,"")]), name if name!="" else f"YW_{t1}_{t2}_{t3}_{g}")

MODEL_1D_Ev1 = YaoWang2B(0.5, 1, 0.2+0.2j, 0.5, name = "1Ev1")
MODEL_1D_Ev2 = YaoWang2B(1+1j, 1, 0.2+0.2j, 0.2, name = "1Ev2")

MODEL_2D_A = Gen2D({
    0: {
        -2:-0.194+0.316j,
        -1:-0.744-0.634j,
        0: 0.315-0.55j,
        1: 0.217+0.434j,
        2: -0.346-0.488j
    },
    -1: {
        -1:-0.276-0.596j,
        0: 0.777+0.996j,
        1: -0.715-0.37j,
    },
    1: {
        -1:0.591-0.818j,
        0: 0.008+0.979j,
        1: -0.317+0.464j
    }
}, name = "2A")

MODEL_2D_B = Gen2D({
    0: {
        -2:-0.194+0.316j,
        -1:-0.715-0.37j,
        0: 0.315-0.55j,
        1: 0.217+0.434j,
        2:0.591-0.818j
    },
    -1: {
        -1:-0.317+0.464j,
        0: 0.777+0.996j,
        1: -0.744-0.634j,
    },
    1:{
        -1:-0.346-0.488j,
        0:-0.276-0.596j ,
        1: 0.008+0.979j
    }
}, name = "2B")

MODEL_2D_C = Gen2D({
    -2: {0: 0.796121 - 0.468468j},
    -1: {-1:-0.143185 - 1.13573j, 0:-1.25526 + 0.991317j, 1:0.165271 - 0.461177j},
    0: {-2:-0.163037 + 0.361071j, -1:0.909651 - 0.112658j, 0:-0.678558 - 0.386062j, 1:-0.0523665 + 1.53487j, 2:1.27515 + 0.331704j},
    1: {-1:-1.22052 - 0.903354j, 0:-1.15259 + 0.938529j, 1:-0.917016 + 0.474096j},
    2: {0:0.334645 + 0.928897j}
}, name="2C")

"""
MODEL_2D_D = Gen2D({
    0:{
        -1:np.array([[-0.744-0.634j, -0.54 -0.474j], [-0.107+0.129j,  0.34 -0.109j]]),
        0:np.array([[ 0.315-0.55j , -0.715-0.37j ], [ 0.5  +0.816j, -0.282+0.583j]]),
        1:np.array([[ 0.217+0.434j,  0.367+0.554j], [-0.177+0.179j, -0.694-0.544j]])
    },
    -1:{
        -1:np.array([[-0.276-0.596j,  0.008+0.979j], [ 0.796+0.607j, -0.816-0.601j]]),
        0:np.array([[-0.194+0.316j,  0.777+0.996j], [-0.679-0.857j, -0.769+0.998j]]),
        1:np.array([[ 0.315-0.55j , -0.715-0.37j ], [ 0.5  +0.816j, -0.282+0.583j]]),
    },
    1:{
        -1:np.array([[ 0.591-0.818j,  0.053+0.446j], [-0.857+0.755j, -0.074+0.907j]]),
        0:np.array([[-0.276-0.596j,  0.008+0.979j], [ 0.796+0.607j, -0.816-0.601j]]),
        1:np.array([[-0.346-0.488j, -0.317+0.464j], [-0.191-0.558j,  0.437+0.947j]])
    }
}, bands=2, name="2D")
"""