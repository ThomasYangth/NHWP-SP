from HamClass.HamClass import LatticeHam, LatticeTerm, HamModel
import numpy as np

def ETI3D (M, l, B, d, a, bc):
    s0 = np.array([[1,0],[0,1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    s = {"0":s0, "x":sx, "y":sy, "z":sz, 0:sx, 1:sy, 2:sz}
    dirs = ["x","y","z"]
    terms = [LatticeTerm(np.kron(sz,s0), "cos(k{})".format(dir)) for dir in dirs]
    terms.append(LatticeTerm(-M*np.kron(sz,s0), ""))
    terms = terms + [LatticeTerm(l*np.kron(sx,s[dir]), "sin(k{})".format(dir)) for dir in dirs]
    terms = terms + [LatticeTerm(np.sin(a)*B[i]*np.kron(s0,s[i]), "") for i in [0,1,2]]
    terms = terms + [LatticeTerm(np.cos(a)*B[i]*np.kron(sz,s[i]), "") for i in [0,1,2]]
    terms.append(LatticeTerm(np.kron(sx, s0)*1j*d, ""))

    return LatticeHam(3, 4, terms, bc, "ETI_a{:.3f}_bc{}".format(a/np.pi,bc)) #"ETI3DLowRes(M{}_L{}_B{}_d{}_a{}_bc{})".format(M,l,B,d,format(a/np.pi,".2f"),bc))

def TI3DChiral (M, l, bc):
    s0 = np.array([[1,0],[0,1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    s = {"0":s0, "x":sx, "y":sy, "z":sz, 0:sx, 1:sy, 2:sz}
    dirs = ["x","y","z"]
    terms = [LatticeTerm(s0, "cos(k{})".format(dir)) for dir in dirs]
    terms.append(LatticeTerm(-M*s0, ""))
    terms = terms + [LatticeTerm(-1j*l*s[dir], "sin(k{})".format(dir)) for dir in dirs]

    return LatticeHam(3, 2, terms, bc, "TI3DChiral(M{}_L{}_bc{})".format(M,l,bc))
            
def Sato3D (bc):
    s0 = np.array([[1,0],[0,1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    s = {"0":s0, "x":sx, "y":sy, "z":sz, 0:sx, 1:sy, 2:sz}
    dirs = ["x","y","z"]
    terms = [LatticeTerm(sx, "sin(kx)"), LatticeTerm(sy, "sin(kz)"), LatticeTerm(-1j*s0, "sin(ky)"), LatticeTerm(2*sz, "")]
    terms = terms + [LatticeTerm(-1*sz, "cos(k{})".format(dir)) for dir in dirs]

    return LatticeHam(3, 2, terms, bc, "Sato3D(bc{})".format(bc))
            
def ETI2D (M, l, t, B = [0,0]):
    s0 = np.array([[1,0],[0,1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    s = {"0":s0, "x":sx, "y":sy, "z":sz, 0:sx, 1:sy, 2:sz}
    dirs = ["x","y"]
    terms = [LatticeTerm(l*s[dir], "sin(k{})".format(dir)) for dir in dirs]
    terms = terms + [LatticeTerm(-1j*t*s0, "cos(k{})".format(dir)) for dir in dirs]
    terms.append(LatticeTerm(1j*M*s0, ""))
    terms.append(LatticeTerm(B[0]*sx,""))
    terms.append(LatticeTerm(B[1]*sy,""))

    return HamModel(2, 2, terms, "ETI2D(M{}l{}t{}{})".format(M,l,t,"_B{}".format(B) if B != [0,0] else ""))

def ETI2DDouble (M, l, t, bc, B = [0,0]):
    s0 = np.array([[1,0],[0,1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    s = {"0":s0, "x":sx, "y":sy, "z":sz, 0:sx, 1:sy, 2:sz}
    dirs = ["x","y"]
    terms = [LatticeTerm(l*np.kron(sx,s0), "sin(kx)".format(dir)), LatticeTerm(l*1j*np.kron(s0,s0), "sin(ky)".format(dir))]
    terms = terms + [LatticeTerm(-np.kron(sy,s0), "cos(kx)".format(dir)) for dir in dirs]
    terms.append(LatticeTerm(M*np.kron(sy,s0), ""))
    terms.append(LatticeTerm(t*np.kron(sx, sx), ""))
    terms.append(LatticeTerm(B[0]*np.kron(sx,s0),""))
    terms.append(LatticeTerm(B[1]*np.kron(sy,s0),""))

    return LatticeHam(2, 4, terms, bc, "ETI2D_DBL(M{}_L{}_t{}_bc{}{})".format(M,l,t,bc,"_B{}".format(B) if B != [0,0] else ""))
            
def TI2D (M, l, bc, B = [0,0]):
    s0 = np.array([[1,0],[0,1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    s = {"0":s0, "x":sx, "y":sy, "z":sz, 0:sx, 1:sy, 2:sz}
    dirs = ["x","y"]
    terms = [LatticeTerm(l*s[dir], "sin(k{})".format(dir)) for dir in dirs]
    terms = terms + [LatticeTerm(-sz, "cos(k{})".format(dir)) for dir in dirs]
    terms.append(LatticeTerm(M*sz, ""))
    terms.append(LatticeTerm(B[0]*sx,""))
    terms.append(LatticeTerm(B[1]*sy,""))

    return LatticeHam(2, 2, terms, bc, "TI2D(M{}_L{}_bc{}{})".format(M,l,bc,"_B{}".format(B) if B != [0,0] else ""))

def ETI2DMx (M, l, bc, B = [0,0], E=0, Gamma=1):
    s0 = np.array([[1,0],[0,1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    s = {"0":s0, "x":sx, "y":sy, "z":sz, 0:sx, 1:sy, 2:sz}
    dirs = ["x","y"]
    terms = [LatticeTerm(sx, "sin(kx)"), LatticeTerm(1j*Gamma*s0, "sin(ky)")]
    terms = terms + [LatticeTerm(-sy, "cos(k{})".format(dir)) for dir in dirs]
    terms.append(LatticeTerm(M*sy, ""))
    terms.append(LatticeTerm(B[0]*sx,""))
    terms.append(LatticeTerm(1j*B[1]*s0,""))
    terms.append(LatticeTerm(-E*s0, ""))

    return LatticeHam(2, 2, terms, bc, "ETI2DMx(M{}_L{}_bc{}{}{}_E{})".format(M,l,bc,"_B{}".format(B) if B != [0,0] else "","_G{}".format(Gamma) if Gamma != 1 else "",E))

def ETI2DMxMG (M, l, bc, B = [0,0], E=0, Gamma=1):
    s0 = np.array([[1,0],[0,1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    s = {"0":s0, "x":sx, "y":sy, "z":sz, 0:sx, 1:sy, 2:sz}
    dirs = ["x","y"]
    terms = [LatticeTerm(sx, "sin(kx)"), LatticeTerm(1j*Gamma*s0, "sin(ky)")]
    terms = terms + [LatticeTerm(-sy, "cos(k{})".format(dir)) for dir in dirs]
    terms.append(LatticeTerm(M*sy, ""))
    terms.append(LatticeTerm(B[0]*sx,"cos(2kx)cos(ky)"))
    terms.append(LatticeTerm(1j*B[1]*s0,""))
    terms.append(LatticeTerm(-E*s0, ""))

    return LatticeHam(2, 2, terms, bc, "ETI2DMxMG(M{}_L{}_bc{}{}{}_E{})".format(M,l,bc,"_B{}".format(B) if B != [0,0] else "","_G{}".format(Gamma) if Gamma != 1 else "",E))
            
def ETI2DRot (M, l, theta, B = [0,0], Sz = 0):
    s0 = np.array([[1,0],[0,1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    s = {"0":s0, "x":sx, "y":sy, "z":sz, 0:sx, 1:sy, 2:sz}
    dirs = ["x","y"]
    a = (1+2*np.cos(2*np.pi*theta))/3
    b = (1+2*np.cos(2*np.pi*theta-2*np.pi/3))/3
    c = (1+2*np.cos(2*np.pi*theta+2*np.pi/3))/3
    def rc (i):
        if i % 3 == 0:
            return a
        elif i % 3 == 1:
            return c
        else:
            return b
    sn = [sx, sy, 1j*s0]
    terms = sum([LatticeTerm.parseList([(l*rc(i)*sn[i], "sin(kx)"), (B[0]*rc(i)*sn[i], "cos(kx)"), (l*rc(i+1)*sn[i], "sin(ky)"), (B[1]*rc(i+1)*sn[i], "cos(ky)"), (rc(i+2)*sn[i]*M, ""), (-rc(i+2)*sn[i], "cos(kx)"), (-rc(i+2)*sn[i], "cos(ky)")]) for i in range(3)], [])
    terms.append(LatticeTerm(1j*Sz*sz, ""))

    return HamModel(2, 2, terms, "ETI2DRot(M{}l{}theta{:.2f}{}{})".format(M,l,theta,"_B{}".format(B) if B != [0,0] else "","_Sz{}".format(Sz) if Sz != 0 else ""))

def ETI2DRotHerm (M, l, bc, theta, E = 0, B = [0,0]):
    s0 = np.array([[1,0],[0,1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    s = {"0":s0, "x":sx, "y":sy, "z":sz, 0:sx, 1:sy, 2:sz}
    a = (1+2*np.cos(2*np.pi*theta))/3
    b = (1+2*np.cos(2*np.pi*theta-2*np.pi/3))/3
    c = (1+2*np.cos(2*np.pi*theta+2*np.pi/3))/3
    def rc (i):
        if i % 3 == 0:
            return a
        elif i % 3 == 1:
            return c
        else:
            return b
    terms = sum([LatticeTerm.parseList([(l*rc(i)*s[i], "sin(kx)"), (B[0]*rc(i)*s[i], "cos(kx)"), (l*rc(i+1)*s[i], "sin(ky)"), (B[1]*rc(i+1)*s[i], "cos(ky)"), (rc(i+2)*s[i]*M, ""), (-rc(i+2)*s[i], "cos(kx)"), (-rc(i+2)*s[i], "cos(ky)")]) for i in range(3)], [])
    terms.append(LatticeTerm(-E*sz, ""))

    return LatticeHam(2, 2, terms, bc, "ETI2DRotHerm(M{}_L{}_bc{}_theta{:.2f}{}{})".format(M,l,bc,theta,"_B{}".format(B) if B != [0,0] else "","_E{}".format(E) if E != 0 else ""))

def ETI2DMxSz (M, l, bc, B = [0,0,0], E=0, Gamma=1):
    s0 = np.array([[1,0],[0,1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    s = {"0":s0, "x":sx, "y":sy, "z":sz, 0:sx, 1:sy, 2:sz}
    dirs = ["x","y"]
    terms = [LatticeTerm(sx, "sin(kx)"), LatticeTerm(1j*Gamma*s0, "sin(ky)")]
    terms = terms + [LatticeTerm(-sy, "cos(k{})".format(dir)) for dir in dirs]
    terms.append(LatticeTerm(M*sy, ""))
    terms.append(LatticeTerm(B[0]*sx,"cos(2kx)cos(ky)"))
    terms.append(LatticeTerm(1j*B[2]*sz, "cos(ky)cos(kx)"))
    terms.append(LatticeTerm(1j*B[1]*s0,""))
    terms.append(LatticeTerm(-E*s0, ""))

    return LatticeHam(2, 2, terms, bc, "ETI2DMxSz(M{}_L{}_bc{}{}{}_E{})".format(M,l,bc,"_B{}".format(B) if B != [0,0] else "","_G{}".format(Gamma) if Gamma != 1 else "",E))

def ETI2DMxHerm (M, l, bc, B = [0,0], E=0):
    s0 = np.array([[1,0],[0,1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    s = {"0":s0, "x":sx, "y":sy, "z":sz, 0:sx, 1:sy, 2:sz}
    dirs = ["x","y"]
    terms = [LatticeTerm(sx, "sin(kx)"), LatticeTerm(sz, "sin(ky)")]
    terms = terms + [LatticeTerm(-sy, "cos(k{})".format(dir)) for dir in dirs]
    terms.append(LatticeTerm(M*sy, ""))
    terms.append(LatticeTerm(B[0]*sx,""))
    terms.append(LatticeTerm(B[1]*sy,""))
    terms.append(LatticeTerm(-E*s0, ""))

    return LatticeHam(2, 2, terms, bc, "ETI2DMxHerm(M{}_L{}_bc{}{}_E{})".format(M,l,bc,"_B{}".format(B) if B != [0,0] else "",E))

def ETI2DMxDBL (M, l, bc, g = 0, B = [0,0]):
    s0 = np.array([[1,0],[0,1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    s = {"0":s0, "x":sx, "y":sy, "z":sz, 0:sx, 1:sy, 2:sz}
    dirs = ["x","y"]
    terms = [LatticeTerm(np.kron(s0,sx), "sin(kx)"), LatticeTerm(1j*np.kron(s0,s0), "sin(ky)")]
    terms = terms + [LatticeTerm(-np.kron(s0,sy), "cos(k{})".format(dir)) for dir in dirs]
    terms.append(LatticeTerm(M*np.kron(s0,sy), ""))
    terms.append(LatticeTerm(g*np.kron(sx,sx), ""))
    terms.append(LatticeTerm(B[0]*np.kron(s0,sx),""))
    terms.append(LatticeTerm(B[1]*np.kron(s0,sy),""))

    return LatticeHam(2, 4, terms, bc, "ETI2DMxDBL(M{}_L{}_bc{}_g{}{})".format(M,l,bc,g,"_B{}".format(B) if B != [0,0] else ""))
            
def ETI2DMogai (M, l, bc):
    s0 = np.array([[1,0],[0,1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    s = {"0":s0, "x":sx, "y":sy, "z":sz, 0:sx, 1:sy, 2:sz}
    dirs = ["x","y"]
    terms = [LatticeTerm(l*s[dir], "sin(k{})".format(dir)) for dir in dirs]
    terms = terms + [LatticeTerm(-1j*s0, "cos(kx)".format(dir)), LatticeTerm(-1j*s0, "cos(2ky)".format(dir))]
    terms.append(LatticeTerm(M*1j*s0, ""))

    return LatticeHam(2, 2, terms, bc)

def ETI2Dblabla (M, l, bc):
    s0 = np.array([[1,0],[0,1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    s = {"0":s0, "x":sx, "y":sy, "z":sz, 0:sx, 1:sy, 2:sz}
    dirs = ["x","y"]
    terms = [LatticeTerm(l*s[dir], "sin(k{})".format(dir)) for dir in dirs]
    terms.append(LatticeTerm(0.1*s, "cos(ky)"))
    terms = terms + [LatticeTerm(-1j*s0, "cos(k{})".format(dir)) for dir in dirs]
    terms.append(LatticeTerm(1j*M*s0, ""))

    return LatticeHam(2, 2, terms, bc)

def NoriModel (m, S, bc):
    s0 = np.array([[1,0],[0,1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    s = {"0":s0, "x":sx, "y":sy, "z":sz, 0:sx, 1:sy, 2:sz}
    dirs = ["x","y"]
    terms = [LatticeTerm(sz,"sin(kx)"), LatticeTerm(-1j*S*sz,"sin(ky)"), LatticeTerm(m*sx, "")]

    return LatticeHam(2, 2, terms, bc)

def SSH2D (omega, nu, gamma, bc):
    s0 = np.array([[1,0],[0,1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    terms = LatticeTerm.parseList([(1j*gamma*np.kron(s0,sz),""), (omega*np.kron(s0,sx),""), (nu*np.kron(s0,sx),"cos(kx)"),
        (nu*np.kron(s0,sy),"sin(kx)"), (omega*np.kron(sx,sz),""), (nu*np.kron(sx,sz),"cos(ky)"), (nu*np.kron(sy,sz),"sin(ky)")])
    return LatticeHam(2, 4, terms, bc)

def SSH (Bx, By, m, bc):
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    terms = LatticeTerm.parseList([(Bx*sx,""), (sx,"cos(kx)"), (By*sy,""), (sy,"sin(kx)"), (m*sz, "")])
    return LatticeHam(1, 2, terms, bc)

def NHTM3D (m, l, b, bc):
    s0 = np.array([[1,0],[0,1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    s = {"0":s0, "x":sx, "y":sy, "z":sz, 0:sx, 1:sy, 2:sz}
    dirs = ["x","y","z"]
    terms = [LatticeTerm(-s0, "cos(k{})".format(dir)) for dir in dirs]
    terms.append(LatticeTerm(m*s0, ""))
    terms = terms + [LatticeTerm(1j*l*s[dir], "sin(k{})".format(dir)) for dir in dirs]
    terms.append(LatticeTerm(2j*l*b*sx, "cos(kz)"))
    return LatticeHam(3, 2, terms, bc, "NHTM3D_m{}_l{}_b{}_bc{}".format(m, l, b, bc))

def HNModel (tL, tR, bc, wL=0, wR=0, wS=0):
    return LatticeHam(1, 1, LatticeTerm.parseList([(tL+tR,"cos(k)",wL),(1j*(tR-tL),"sin(k)",wR),(0,"",wS)]), bc, "HN_L{}_{}_R{}_{}_S{}_bc{}".format(tL,wL,tR,wR,wS,bc))

def GDSEMinimal (bc, p = 1, q = 0, tx=1, ty=1):
    # The model cos(kx)+icos(ky), with axes x and y rotated by an angle arctan(q/p).
    return LatticeHam(2, 1, LatticeTerm.parseList([(tx,"cos({}kx)cos({}ky)".format(p,q)), (-tx,"sin({}kx)sin({}ky)".format(p,q)), 
        (1j*ty,"cos({}kx)cos({}ky)".format(q,p)), (1j*ty,"sin({}kx)sin({}ky)".format(q,p))]), bc, "GDSEMin_p{}q{}_tx{}_ty{}_bc{}".format(p, q, tx, ty, bc))

def BiNHTM_EP (m, l, b, eps, epk, bc):
    s0 = np.array([[1,0],[0,1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sp = (sx+1j*sy)/2
    sm = (sx-1j*sy)/2
    sz = np.array([[1,0],[0,-1]])
    s = {"0":s0, "x":sx, "y":sy, "z":sz, 0:sx, 1:sy, 2:sz}
    dirs = ["x","y","z"]
    terms = [LatticeTerm(-np.kron(s0,s0), "cos(k{})".format(dir)) for dir in dirs]
    terms.append(LatticeTerm(m*np.kron(s0,s0), ""))
    terms = terms + [LatticeTerm(1j*l*np.kron(s0,s[dir]), "sin(k{})".format(dir)) for dir in dirs]
    terms.append(LatticeTerm(2j*l*b*np.kron(s0,sx), "cos(kz)"))
    terms.append(LatticeTerm(eps*np.kron(sp, s0)*np.cos(epk[0]), "sin(kx)"))
    terms.append(LatticeTerm(-eps*np.kron(sp, s0)*np.sin(epk[0]), "cos(kx)"))
    terms.append(LatticeTerm(1j*eps*np.kron(sp, s0)*np.cos(epk[1]), "sin(ky)"))
    terms.append(LatticeTerm(-1j*eps*np.kron(sp, s0)*np.sin(epk[1]), "cos(ky)"))
    terms.append(LatticeTerm(eps*np.kron(sm, s0), ""))
    return LatticeHam(3, 4, terms, bc, "BiNHTM_eps{}_epk{}_bc{}".format(eps, epk, bc))

def EdgeDisp1 (bc):
    return LatticeHam(2, 1, LatticeTerm.parseList([(1,"cos(ky)"), (1j, "sin(ky)"), (0.1, "cos(2ky)"), (1, "cos(kx)"), (0.5, "sin(kx)cos(ky)")]), bc, "ED1_bc{}".format(bc))

def EdgeDisp2 (bc):
    return LatticeHam(2, 1, LatticeTerm.parseList([(1,"cos(kx)"), (1, "cos(ky)"), (1j, "sin(kx)sin(ky)")]), bc, "ED2_bc{}".format(bc))

def EdgeDisp3 (bc):
    # The Prototypical Corner Skin Effect model in the Higher Dimensional universal NHSE paper
    return LatticeHam(2, 1, LatticeTerm.parseList([(5,"cos(kx)"), (5,"cos(2kx)"), (-1j,"sin(kx))"), (-3j,"sin(2kx)"), (5,"cos(ky)"), (1j,"sin(ky)")]), bc, "ED3_bc{}".format(bc))

def MovableEP (m, mu, alpha, beta, bc):
    s0 = np.array([[1,0],[0,1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    s1 = (s0+sz)/2
    s2 = (s0-sz)/2
    s = {"0":s0, "x":sx, "y":sy, "z":sz, 0:sx, 1:sy, 2:sz}
    terms = [LatticeTerm(m*np.kron(s0, sx), "")]
    terms += [LatticeTerm(-np.kron(s0,sx), "cos(k{})".format(dir)) for dir in ["x","y","z"]]
    terms.append(LatticeTerm(np.kron(s0, sy), "sin(ky)"))
    terms.append(LatticeTerm(0.5*np.kron(sz, sz), "sin(kz)"))
    terms.append(LatticeTerm(1j*np.kron(s1, s0), "sin(kx)"))
    terms.append(LatticeTerm(1j*mu*np.kron(s2, s0), ""))
    terms.append(LatticeTerm(0.5j*np.kron(sx, s0), "sin(kz)"))
    terms.append(LatticeTerm(-0.5*np.kron(sx, s0), "sin(kx)"))
    terms.append(LatticeTerm(-alpha*np.kron(sx, s0), ""))
    terms.append(LatticeTerm(1j*beta*np.kron(sy, s0)," sin(ky)"))
    return LatticeHam(3, 4, terms, bc, "MEP_m{}_mu{}_a{}_b{}_bc{}".format(m, mu, alpha, beta, bc))

def GDSEMinimal (bc, p = 1, q = 0, tx=1, ty=1):
    # The model cos(kx)+icos(ky), with axes x and y rotated by an angle arctan(q/p).
    return LatticeHam(2, 1, LatticeTerm.parseList([(tx,"cos({}kx)cos({}ky)".format(p,q)), (-tx,"sin({}kx)sin({}ky)".format(p,q)), 
        (1j*ty,"cos({}kx)cos({}ky)".format(q,p)), (1j*ty,"sin({}kx)sin({}ky)".format(q,p))]), bc, "GDSEMin_p{}q{}_tx{}_ty{}_bc{}".format(p, q, tx, ty, bc))

def WTXModel (t1L, t1R, t2L, t2R, kap, bc):
    return LatticeHam(1, 1, LatticeTerm.parseList([(t1L+t1R,"cos(k)"),(1j*(t1R-t1L),"sin(k)"),(t2L+t2R,"cos(2k)"),(1j*(t2R-t2L),"sin(2k)"),(-1j*kap,"")]), bc, "WTX_bc{}".format(bc))

def CPSkin1 (a, b, c):
    return HamModel(2, 1, LatticeTerm.parseList([(1,"cos(kx)"),(1,"cos(ky)"),(1j*a+c,"sin(ky)"),(b,"sin(kx)sin(ky)")]), "CPS1_a{}b{}c{}".format(a, b, c))

def CPSkin2 (a, b, c, bc):
    return LatticeHam(2, 1, LatticeTerm.parseList([(1/(2*c),"exp(ky)"),(a/(2*c),"exp(kx)exp(ky)"),(b/(2*c),"exp(-kx)exp(ky)"),(c/2,"exp(-ky)")]), bc, "CPS2_a{}b{}c{}_bc{}".format(a, b, c, bc))

def CPSkin1edge (a, b, c, bc):
    s0 = np.array([[1,0],[0,1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    return LatticeHam(1, 2, LatticeTerm.parseList([(s0,"cos(k)"),(sx,""),((1j*a-c)*sy,""),(-b*sy,"sin(k)")]), bc, "CPS1e_a{}b{}c{}_bc{}".format(a, b, c, bc))

def CPSkin3 (a, b, c):
    return HamModel(2, 1, LatticeTerm.parseList([(1,"cos(kx)"),(1,"cos(ky)"),(1j*a,"sin(ky)"),(1j*b,"cos(kx)sin(ky)"),(c,"sin(kx)cos(ky)")]), "CPS3_a{}b{}c{}".format(a, b, c))

def CPSkin3edge (a, b, c, bc):
    s0 = np.array([[1,0],[0,1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    return LatticeHam(1, 2, LatticeTerm.parseList([(s0,"cos(k)"),(sx,""),(c*sx,"sin(k)"),(-1j*a*sy,""),(-1j*b*sy,"cos(k)")]), bc, "CPS3e_a{}b{}c{}_bc{}".format(a, b, c, bc))

def CPSkin30 (a, b, c, bc):
    return LatticeHam(2, 1, LatticeTerm.parseList([(1,"cos(ky)"),(1j*a,"sin(ky)"),(1j*b,"cos(kx)sin(ky)"),(c,"sin(kx)cos(ky)")]), bc, "CPS3_a{}b{}c{}_bc{}".format(a, b, c, bc))

def CPSkin3edge0 (a, b, c, bc):
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    return LatticeHam(1, 2, LatticeTerm.parseList([(sx,""),(c*sx,"sin(k)"),(-1j*a*sy,""),(-1j*b*sy,"cos(k)")]), bc, "CPS3e0_a{}b{}c{}_bc{}".format(a, b, c, bc))

def SqrtModel (a, b, termNo, bc):

    from scipy.special import comb

    now_coef = 1
    terms = [LatticeTerm(1,"")]
    for n in range(1,termNo+1):
        now_coef *= (3-2*n)/(2*n)
        for m in range(n+1):
            terms.append(LatticeTerm(now_coef*comb(n,m)*a**m*b**(n-m), "exp({}k)".format(2*m-n)))
    return LatticeHam(1, 1, terms, bc, "Sqr0_a{}b{}_{}T_bc{}".format(a, b, termNo, bc))

def YaoWang2B (t1, t2, t3, g):
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    return HamModel(1, 2, LatticeTerm.parseList([(t1*sx,""),((t2+t3)*sx,"cos(k)"),((t2-t3)*sy,"sin(k)"),(0.5j*g*sy,"")]), f"YW_{t1}_{t2}_{t3}_{g}")

def YaoWangQWZ (gamma, m):
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    return HamModel(2, 2, LatticeTerm.parseList([(1j*gamma*sx,""),(sx,"sin(kx)"),(1j*gamma*sy,""),(sy,"sin(ky)"),((m+1j*gamma)*sz,""),(-sz,"cos(kx)"),(-sz,"cos(ky)")]), "YWqwz_{}_{}".format(gamma, m))

def WTXModel2B (t1L, t1R, t2L, t2R, kap, c, r, bc):
    s0 = np.array([[1,0],[0,1]])
    sz = np.array([[1,0],[0,-1]])
    sx = np.array([[0,1],[1,0]])
    lst = [((t1L+t1R)*s0,"cos(k)"),(1j*(t1R-t1L)*s0,"sin(k)"),((t2L+t2R)*s0,"cos(2k)"),(1j*(t2R-t2L)*s0,"sin(2k)"),(-1j*kap*s0,"")]
    lst += [((t1L+t1R)*r*sz,"cos(k)"),(1j*(t1R-t1L)*r*sz,"sin(k)"),((t2L+t2R)*r*sz,"cos(2k)"),(1j*(t2R-t2L)*r*sz,"sin(2k)"),(-1j*kap*r*sz,"")]
    lst += [(c*sx,"")]
    return LatticeHam(1, 2, LatticeTerm.parseList(lst), bc, "WTX2b_bc{}".format(bc))

def HN2B (tL, tR, c, r, bc):
    s0 = np.array([[1,0],[0,1]])
    sz = np.array([[1,0],[0,-1]])
    sx = np.array([[0,1],[1,0]])
    lst = [((tL+tR)*s0,"cos(k)"),(1j*(tL-tR)*s0,"sin(k)"),((tL+tR)*r*sz,"cos(k)"),(1j*(tL-tR)*r*sz,"sin(k)"),(c*sx,"")]
    return LatticeHam(1, 2, LatticeTerm.parseList(lst), bc, "HN2b_bc{}".format(bc))

def ThreeModel (ts, bc):
    t1L, t1R, t2L, t2R, t3L, t3R = ts
    #return LatticeHam(1, 1, LatticeTerm.parseList([(t1L+t1R,"cos(k)"),(1j*(t1R-t1L),"sin(k)"),(t2L+t2R,"cos(2k)"),(1j*(t2R-t2L),"sin(2k)"),(t3L+t3R,"cos(3k)"),(1j*(t3R-t3L),"sin(3k)")]), bc, "WTX_bc{}".format(bc))
    return LatticeHam(1, 1, LatticeTerm.parseList([(t1L+t1R,"cos(k)"),(1j*(t1L-t1R),"sin(k)"),(t2L+t2R,"cos(2k)"),(1j*(t2L-t2R),"sin(2k)"),(t3L+t3R,"cos(3k)"),(1j*(t3L-t3R),"sin(3k)")]), bc, "WTX_bc{}".format(bc))

def Gen2B (tL, tR, tE, addn, bc):
    return LatticeHam(1, 2, LatticeTerm.parseList([(tL+tR,"cos(k)"), (1j*(tL-tR),"sin(k)"), (tE, "")]), bc, "G2B_{}".format(addn))

def InvSP2B (a, c, e, bc):
    return LatticeHam(1, 2, LatticeTerm.parseList([(np.array([[2,0],[0,1+a]]),"cos(k)"), (np.array([[0,0],[0,1j*(1-a)]]),"sin(k)"),
                                                  (np.array([[0,c],[np.conj(c),0]]),""), (np.array([[e,0],[0,0]]),"exp(2k)")]), bc, "Inv2B_a{}c{}e{}".format(a,c,e))

def genRandom2D (rng, bc, name):
    ts = np.random.randn(2*rng+1,2*rng+1,2)
    ts = ts[:,:,0]+1j*ts[:,:,1]
    lst = [(ts[i,j],"exp({}kx)exp({}ky)".format(i-rng,j-rng)) for i in range(2*rng+1) for j in range(2*rng+1)]
    return LatticeHam(2, 1, LatticeTerm.parseList(lst), bc, name)

def GenMulti (ts:dict, bands, addn="", ran = None):
    term = []
    for i in ts.keys():
        if i == 0 and ran is not None:
            term.append((ts[0],"",ran))
        else:
            term.append((ts[i],"exp({}k)".format(i)))
    return HamModel(1, bands, LatticeTerm.parseList(term), "G{}B_{}".format(bands, addn))

def GenHN2D (tsp:dict, ts0:dict, tsm:dict, addn="", ran = None, bands=1):
    term = []
    for i in tsp.keys():
        term.append((tsp[i],"exp({}kx)exp(ky)".format(i)))
    for i in ts0.keys():
        if i == 0 and ran is not None:
            term.append((ts0[i],"",ran))
        else:
            term.append((ts0[i],"exp({}kx)".format(i)))
    for i in tsm.keys():
        term.append((tsm[i],"exp({}kx)exp(-1ky)".format(i)))
    return HamModel(2, bands, LatticeTerm.parseList(term), "GHN_{}".format(addn))

def GenHN2DY (tsp:dict, ts0:dict, tsm:dict, addn="", ran = None):
    term = []
    for i in tsp.keys():
        term.append((tsp[i],"exp({}ky)exp(kx)".format(i)))
    for i in ts0.keys():
        if i == 0 and ran is not None:
            term.append((ts0[i],"",ran))
        else:
            term.append((ts0[i],"exp({}ky)".format(i)))
    for i in tsm.keys():
        term.append((tsm[i],"exp({}ky)exp(-1kx)".format(i)))
    return HamModel(2, 1, LatticeTerm.parseList(term), "GHNY_{}".format(addn))

def GenHN2Dedge (tsp:dict, ts0:dict, tsm:dict, bc, addn=""):
    term = []
    for i in tsp.keys():
        term.append((np.array([[0,2*tsp[i]],[0,0]]),"exp({}kx)".format(i)))
    for i in ts0.keys():
        term.append((ts0[i]*np.eye(2),"exp({}kx)".format(i)))
    for i in tsm.keys():
        term.append((np.array([[0,0],[2*tsm[i],0]]),"exp({}kx)".format(i)))
    return LatticeHam(1, 2, LatticeTerm.parseList(term), bc, "GHN_{}".format(addn))
       
def ChernTunedX (M, l, g, theta):
    s0 = np.array([[1,0],[0,1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    terms = [LatticeTerm(l*sx, "sin(kx)"), LatticeTerm(l*sy, "sin(ky)")]
    terms = terms + [LatticeTerm(M*sz, ""), LatticeTerm(-sz, "cos(kx)"), LatticeTerm(-sz, "cos(ky)")]
    terms = terms + [LatticeTerm(g*1j*np.cos(theta)*sx, "cos(kx)"), LatticeTerm(g*1j*np.sin(theta)*sx, "sin(kx)")]
    return HamModel(2, 2, terms, "ChernTunedX_M{}l{}g{}t{:.2f}pi".format(M, l, g, theta/np.pi))

def OppoEdge2D (a, b, c, theta):
    term = [(-1,"cos(kx)"), (-1,"cos(ky)"), (1j*a,"sin(ky)"), (1j*b/np.sqrt(2),"cos(kx)sin(ky)"),
            (1j*b/np.sqrt(2),"sin(kx)sin(ky)"), (0.5j*c,"cos(ky)"), (0.5j*c*np.cos(theta),"cos(kx)cos(ky)"),
            (-0.5j*c*np.sin(theta),"sin(kx)cos(ky)")]
    return HamModel(2, 1, LatticeTerm.parseList(term), f"OppoEdge_a{a}b{b}c{c}th{theta}")