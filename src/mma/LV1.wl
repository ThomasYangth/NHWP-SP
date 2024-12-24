(* ::Package:: *)

(* ::Text:: *)
(*This is the package for evaluating saddle points and growth rates of Hamiltonians in one dimension.*)


BeginPackage["LV1`"];

SPS1::usage="SPS1[Heq_,v_:0] = Table[{z,w}]";
SolvePath1::usage="SolvePath1[Heq_,z0_,H0_,T_,dir_,v_:0,idir_:-1,FindBZI_:0] = {z,w}";
Flow1::usage="Flow1[Heq_,z0_,w0_,T_,dir_,v_:0,idir_:-1] = SolvePath1[..][T]";
SPFlows1::usage="SPFlows1[Hmat_,v_,print_:True]";
GetGrowthNew1::usage="GetGrowthNew1[Hmat_,v_:0,print_:True]";
GrowthList1::usage="Table[GetGrowthNew1[Hmat,v],{v,vs}]";
PlotGrowth1::usage="PlotGrowth1[Hmat_,vrng_,prec_:20]";
TableBZItsc::usage="TableBZItsc[Heq_,sps_,v_:0,BZrad_:1] = List of whether saddle points have non-zero winding";
FindBZItsc::usage="FindBZItsc[Heq_,sps_,v_:0,BZrad_:1] = {positions of non-zero winding}";

Begin["`Private`"];


(* ::Text:: *)
(*Functions for handling negative-degree polynomials.*)


(*Judge whether expr[var] is singular at var=0*)
Judge1[expr_,var_]:=Quiet[With[{tval=expr/.var->0},tval===ComplexInfinity||tval===Indeterminate]]
(*Find the highest negative exponent of var in Fun*)
NegDegree1[Fun_,var_]:=Module[{i,Nowf},i=0;Nowf=Fun;While[Judge1[Nowf,var],Nowf=Expand[Nowf*var];i+=1];i]
(*Turns a Laurent polynomial Fun[var] into a polynomial*)
CancelNeg1[Fun_,var_]:=Expand[Fun*var^NegDegree1[Fun,var]]


(* ::Text:: *)
(*Some random functions to be used*)


(*Get the angle of a complex number*)
Ang1[z_]:=z/Abs[z]
(*Returns the growth rate of a saddle point*)
Lam1[sp_,v_:0]:=N[Im[sp[[2]]]+v*Log[Abs[sp[[1]]]]]


(* ::Text:: *)
(*The function SP1 finds the saddle point of a Hamiltonian at a given velocity.*)


SPS1[Heq_,v_:0]:=
Module[{z,l},
	SortBy[
		N[{z,l}/.
			(*We suppress the rantz error message, which would pop up when the
			  Hamiltonian includes exact coefficients*)
			Quiet[
				Solve[Heq[z,l]==0&&z*D[Heq[z,l],z]-I*v*D[Heq[z,l],l]==0,{z,l}],
			Solve::ratnz]
		],
		-Lam1[#,v]& (*The saddle points are sorted in descending
										  order of lambda*)
	]
]


(* ::Text:: *)
(*Finds the upward- or downward-flowing path starting from a saddle point*)
(*# Input*)
(*Heq -- A function of [z,l], the characteristic polynomial of the Hamiltonian*)
(*z0 -- The initial point of flow*)
(*H0 -- The H value at the initial point, should satisfy Heq[z0,H0]=0*)
(*T -- The length of the path, in terms of the difference of the final and initial values of lambda*)
(*dir -- +1 or -1, corresponding to two possible directions of the flowing*)
(*v -- The velocity*)
(*idir -- +1 or -1, corresponding to increasing or decreasing lambda*)
(*# Returns*)
(*{z[s],w[s]}, the flow function parametrized by 0<=s<=T*)


SolvePath1[Heq_,z0_,H0_,T_,dir_,v_:0,idir_:-1]:=
Module[{z,w,mdz,mdw,zarg,fz,fw,fd0,rt0,eps,tol},

	(*Partial derivative of Heq w.r.t. z*)
	fz[z_,w_]:=Module[{zz,ww},D[Heq[zz,ww],zz]/.{zz->z,ww->w}];
	(*Partial derivative of Heq w.r.t. w*)
	fw[z_,w_]:=Module[{zz,ww},D[Heq[zz,ww],ww]/.{zz->z,ww->w}];
	(*High-Order Partial derivative of Heq, at the initial point*)
	fd0[zo_,wo_]:=Module[{zz,ww},D[Heq[zz,ww],{zz,zo},{ww,wo}]/.{zz->z0,ww->H0}];
	
	
	(*Since the derivative is singular at the saddle point, the flow
	  must begin at some point that deviates a little from the saddle.
	  eps is an initial guess for the amount of this deviation. We then
	  rescale eps such that Heq[z+Delta z,H+Delta H]=tol, if we approximate
	  Delta H to second order in Delta z*)
	eps=0.01;tol=10^-10;
	(*See Supplementary Materials for detailed derivations*)
	rt0=(fd0[2,0]-fd0[0,2]v^2/z0^2-2I fd0[1,1]v/z0)/fd0[0,1]+I v/z0^2;
	zarg=Sqrt[I/rt0]*dir*Sqrt[-idir]; (*Select the correct direction*)
	(* If we are not even at a saddle point, just set mdz=0 *)
	mdz = If[Abs[fd0[1,0]]<eps^2*Abs[fd0[0,1]],
		eps*CubeRoot[tol/
			Abs[Heq[z0+eps*zarg,H0+idir*I*eps^2/2-I v( eps zarg/z0-(eps zarg/z0)^2/2)]]
		],
		0
	];
	
	(*Now to find the {z,w} tuple, both are functions of a parameter s, which
	  corresponds to the flowing path. s ranges from ~0 to T.*)
	Check[
		{z,w}/.NDSolve[
			{
				(*See Supplementary Materials for detailed derivations*)
				z'[s]==-idir*I/(fz[z[s],w[s]]/fw[z[s],w[s]]-I*v/z[s]),
				w'[s]==idir*I/(1 - I*v/z[s]*fw[z[s],w[s]]/fz[z[s],w[s]]),
				z[mdz^2/2]==z0+mdz*zarg,
				w[mdz^2/2]==H0+idir*I*mdz^2/2-I*v*(mdz*zarg/z0-(mdz*zarg/z0)^2/2)
			},
			{z,w},{s,T}
		][[1]], (*NDSolve will return a list, its [[1]] is the {z,w} we want*)
		Print["ERR::Error in solving differential equation, probably with variable definitions"];
		Abort[],
		ReplaceAll::reps
	]
]

(*Gets the final {z,w}, at time T, from the functions return by SolvePath1*)
Flow1[Heq_,z0_,w0_,T_,dir_,v_:0,idir_:-1]:=
Module[{z,w,s,ds,zf,wf,sps,i},
	z = z0;
	w = w0;
	s = 0;
	i = 1; (*Iteration count*)
	While[True,
		{zf,wf} = SolvePath1[Heq,z,w,T-s,dir,v,idir];(*Try to flow to s=T*)
		ds = (zf@Domain[])[[1,2]];(*Find out how much we actually flowed*)
		z = N[zf[ds]]; w = N[wf[ds]];
		s += ds;
		If[s>=T,
			Break[], (*If we did reach s=T, we can exit the loop*)
			If[!ValueQ[sps], sps=SPS1[Heq,v]];
			{z,w} = Flatten[Nearest[sps,{z,w}]]; (*If the flow is terminated early, it should have
										ran into a saddle point*)
			i++;
		];
		If[i>10,
			Print["ERR::Too many ran-into-saddle iterations"];
			Abort[]
		]
	];
	{z,w}
]


(* ::Text:: *)
(*Functions that finds whether or not the flowing path touches the Brillouin Zone*)


(*The newest BZ intersection finder, based on flowing the spectrum in two directions and
  checking whether they lie in the same or opposite sides of the BZ. Does not require
  Reaping in the process of SolvePath1.*)
  
(* TableBZItsc returns a boolean list, with Length equal to the number of saddle points,
   indicating whether each saddle point intersects the BZ non-trivially*)
TableBZItsc[Heq_,sps_,v_:0,BZrad_:1]:=
Module[{l,PBCtop,lam},
	(*Find the largest Im on the PBC Spectrum of the Hamiltonian*)
	PBCtop = NMaximize[
				Max[Im[l/.Solve[Heq[BZrad*Exp[I*k],l]==0,l]]]+v*Log[BZrad],{k,0,2Pi}
			][[1]] + 0.1;
	Table[
		lam = Lam1[sps[[i]],v]; (*The lambda value of this saddle point*)
		If[lam < PBCtop,
		
			(*If lambda < PBCtop, we flow the saddle point upwards*)
			(Plus@@Table[
				Sign[
					Abs[
						(*Flow the saddle point upwards, until its imaginary part is
						  bigger than the top of the PBC spectrum*)
						Flow1[
							Heq, sps[[i,1]], sps[[i,2]],
							PBCtop - lam,
							dir, v, 1
						][[1]] (*Pick the z[T]*)
					] - BZrad (*And see whether z[T] is larger or smaller than BZrad*)
				],
				{dir,{1,-1}}
			]) == 0, (*We check whether Sign[z[T]-BZrad] is the same for either
				     flow directions; if it's the same, this point is invalid;
				     if it's opposite, this point is valid*)
				     
			False (*If lambda >= PBCtop, it is invalid*)
		],
		{i,Length[sps]}
	]
]

(* FindBZItsc returns the list of indices of BZ-intersecting saddle points *)
FindBZItsc[Heq_,sps_,v_:0,BZrad_:1]:=
	Flatten[
		Position[
			TableBZItsc[Heq,sps,v,BZrad],
			True (*Pick out the saddle points that intersects the BZ non-trivially*)
		]
	]
	
Windings[Heq_,sps_,v_:0,BZrad_:1]:=
Module[{l,PBCtop,lam},
	(*Find the largest Im on the PBC Spectrum of the Hamiltonian*)
	PBCtop = NMaximize[
				Max[Im[l/.Solve[Heq[BZrad*Exp[I*k],l]==0,l]]]+v*Log[BZrad],{k,0,2Pi}
			][[1]] + 0.1;
	Table[
		lam = Lam1[sps[[i]],v]; (*The lambda value of this saddle point*)
		If[lam < PBCtop,
		
			(*If lambda < PBCtop, we flow the saddle point upwards*)
			Plus@@Table[
				dir*Sign[
					Abs[
						(*Flow the saddle point upwards, until its imaginary part is
						  bigger than the top of the PBC spectrum*)
						Flow1[
							Heq, sps[[i,1]], sps[[i,2]],
							PBCtop - lam,
							dir, v, 1
						][[1]] (*Pick the z[T]*)
					] - BZrad (*And see whether z[T] is larger or smaller than BZrad*)
				],
				{dir,{1,-1}}
			]/2, (*We check whether Sign[z[T]-BZrad] is the same for either
				     flow directions; if it's the same, this point is invalid;
				     if it's opposite, this point is valid*)
				     
			0 (*If lambda >= PBCtop, it is invalid*)
		],
		{i,Length[sps]}
	]
]


(* ::Text:: *)
(*Get the growth rate of Hmat[z] at velocity v*)


GetGrowthNew1[Hmat_,v_:0,print_:True]:=
Module[{Heq, sps, sppos},
	Heq=Evaluate@CharacteristicPolynomial[Hmat[#1],#2]&;
	sps=SPS1[Heq,v]; (*Get saddle points and sort by lambda*)
	sppos=FindBZItsc[Heq,sps,v,1]; (*Find the BZ intersection status of the saddle points*)
	If[Length[sppos]==0,
		Print["ERR::No saddle points are found valid at velocity "<>ToString[v]];
		Abort[]
	];
	If[print,
		(*If print is True, print the saddle point information*)
		Print["Saddle Points:"];
		Print[Grid[
			Join[{{"z","H[z]","lambda","valid"}},
				Table[{sps[[i,1]],sps[[i,2]],Lam1[sps[[i]],v],MemberQ[sppos,i]},{i,Length[sps]}]
			],
			Frame->All	
		]],
		(*If print is False, return the growth rate at this point*)
		Lam1[sps[[Min[sppos]]],v]
	]
]

GetH2[Heq_,sp_,v_:0]:=
	Module[{z,l,fH,fzz,fHz},
		fH = D[Heq[z,l],l]/.{z->sp[[1]],l->sp[[2]]};
		fzz = D[Heq[z,l],{z,2}]/.{z->sp[[1]],l->sp[[2]]};
		fHz = D[Heq[z,l],z,l]/.{z->sp[[1]],l->sp[[2]]};
		z = sp[[1]];
		-fzz/fH + I*v*(fHz/fH-1/z)/z
	]

(*This function prints out the list of saddle points along with their lambda
  and BZ-intersecting property*)
SPFlows1[Hmat_,v_:0,print_:True]:=
Module[{Heq,sps,result,windings},
	Heq=Evaluate@CharacteristicPolynomial[Hmat[#1],#2]&;
	sps=SPS1[Heq,v];
	windings = Windings[Heq,sps,v,1];
	result = Join[
			Table[{sp[[1]],sp[[2]],Lam1[sp,v]},{sp,sps}],
			Table[{a!=0},{a,windings}],
			Table[{-Sqrt[I/(-GetH2[Heq,sps[[i]],v])]*windings[[i]]},{i,Length[sps]}],
			Table[With[
				{HmE = Hmat[sp[[1]]]-IdentityMatrix[Dimensions[Hmat[sp[[1]]]][[1]]]*sp[[2]]},
				{
				NullSpace[HmE,Tolerance->10^(-12)],
				NullSpace[Transpose[HmE],Tolerance->10^(-12)]
				}]
				,{sp,sps}],
			2
		];
	If[print,
	
		Print["Saddle Points:"];
		Print[Grid[
			Join[
				{{"z","E","lambda","wind","1/Sqrt[I H''[z]]","v"}},
				result
			],
			Frame->All
		]],
		
		result
	]
]

(*Get a list of growth rates*)
GrowthList1[Hmat_,vs_]:=Table[GetGrowthNew1[Hmat,v,False],{v,vs}]

(*Plot growth rates*)
PlotGrowth1[Hmat_,vrng_,prec_:20]:=
Module[{vs,gs},
	vs=Table[(vrng[[1]]*(prec-i)+vrng[[2]]*i)/prec,{i,0,prec}];
	gs=GrowthList1[Hmat,vs];
	Plot[ListInterpolation[gs,{vrng}][v],{v,vrng[[1]],vrng[[2]]}]
]

End[];
EndPackage[];
