(* ::Package:: *)

(* ::Text:: *)
(*This is the package for evaluating saddle points and growth rates of Hamiltonians in more than one dimensions.*)
(*In principle it can work for any dimension, but in practice it is only tested for two dimensions.*)


BeginPackage["LVn`"];

SPS::usage="SPS[Heq_,dim_,v_:0] = Table[{zs,w}]";
SolvePath::usage="SolvePath[Heq_,zs0_,w0_,T_,dir_,v_:0,idir_:-1,evaluate_:-1] = Table[{zs,l} for dir] (dir can be table)";
Flow::usage="Flow[Heq_,zs0_,w0_,T_,dir_,v_:0,idir_:-1] = SolvePath[Heq,zs0,w0,T,dir,v,idir,T]";
SPFlows::usage="SPFlows[Heq_,dim_,v_:0,prec_:20]";
GetGrowth::usage="GetGrowth[Heq_,dim_,vels_,prec_:10] = lambda(vels)";
GrowthList::usage = "GrowthList[Heq_,dim_,vs_(list of {vx,vy})] = {lambdas}";
GrowthBulk::usgae = "GrowthBulk[Heq_,dim_,vxl_,vxh_,vyl_,vyh_,intvs_:20] = {vs,lambds}";
GrowthEdge::usage = "GrowthEdge[Heq_,dim_,edge_,vl_,vh_,intvs_:20] = {vs(1D),lambdas}";

Begin["`Private`"]


(* ::Text:: *)
(*Functions for handling negative-degree polynomials.*)


(*Judge whether expr[var] is singular at var=0*)
Judge[expr_,var_]:=Quiet[With[{tval=expr/.var->0},tval===ComplexInfinity||tval===Indeterminate]]
(*Find the highest negative exponent of var in Fun*)
NegDegree[Fun_,var_]:=Module[{i,Nowf},i=0;Nowf=Fun;While[Judge[Nowf,var],Nowf=Expand[Nowf*var];i+=1];i]
(*Turns a multi-variable Laurent polynomial Fun[vars] into a polynomial*)
CancelNegs[Fun_,vars_]:=Module[{Nowf},Nowf=Fun;For[i=1,i<=Length[vars],i++,Nowf=Expand[Nowf*vars[[i]]^NegDegree[Nowf,vars[[i]]]]];Nowf]


(* ::Text:: *)
(*Some random functions*)


(*Creates a list of symbols {z1,z2,...,zn}*)
Zs[dim_]:=Table[Symbol["z"<>ToString[i]],{i,dim}]
(*Similar for k*)
Ks[dim_]:=Table[Symbol["k"<>ToString[i]],{i,dim}]
(*Get the angle of a complex number*)
Ang[z_]:=z/Abs[z]
(*Returns the lambda growth rate for a saddle point*)
Lam[sp_,v_:0]:=N[Im[sp[[2]]]+Plus@@(v*Log[Abs[sp[[1]]]])]


(* ::Text:: *)
(*The SPS functions finds the saddle point of a Hamiltonian at a given velocity.*)


SPS[Heq_,dim_,v_:0]:=
Module[{vels,HamDs,resz,resz1,zs,l},
	zs = Zs[dim];
	Table[Clear[z],{z,zs}]; (*Reset all z variables*)
	vels = If[Head[v]===List, v, ConstantArray[v,dim]];
	SortBy[
		N[{zs,l} /.
		(*We suppress the rantz error message, which would pop up when the
		  Hamiltonian includes exact coefficients*)
			Quiet[
				Solve[
					(*Solve Heq[z,l]=0 and Heq'[z,l]=v for each z*)
					Heq[zs,l]==0 && 
						And @@ Table[
							zs[[i]]*D[Heq[zs,l],zs[[i]]]-I*vels[[i]]*D[Heq[zs,l],l]==0,
							{i,dim}
						],
					zs ~Join~ {l}
				],
				Solve::ratnz
			]
		],
		-Lam[#, vels]& (*The saddle points are sorted in descending
				   order of lambda*)
	]
]


(* ::Text:: *)
(*SolvePath finds the flowing path starting from a saddle point.*)


(*Returns the Hessian matrix of a function*)
Hess[f_,xs_]:=Table[Grad[k,xs],{k,Grad[f@@xs,xs]}]

SolvePath[Heq_,zs0_,w0_,T_,dir_,v_:0,idir_:-1,evaluate_:-1]:=
Module[{zs,vels,l,w,eps,tol,dim,Hzs,Hw,dzs,fd0,fw0,fww0,
	Lzzs,Rlzzs,eigv,vs,vecs,vec,thirdorder,mdz,mdw,h0,s,GetSol},
	
	eps=0.01;tol=10^-6;
	dim=Length[zs0];
	vels=If[Head[v]===List,v,ConstantArray[v,dim]];
	zs=Zs[dim];
	Table[Clear[z],{z,zs}];

	Hzs=Grad[Heq[zs,l],zs];
	Hw=D[Heq[zs,l],l];
	dzs=Table[-Hzs[[i]]/Hw+I vels[[i]]/zs[[i]],{i,Length[zs]}];
	
	(*Gets the derivatives of Heq at the original points*)
	fd0[inds__]:=
		With[{varfun=Function[i,If[i>0,zs[[i]],l]]}, (*varfun maps i=0 to l, and i>0 to zi*)
			(D@@(
				{Heq[zs,l]} ~Join~
				With[{c=Counts[{inds}]},
					Table[{varfun[k],c[k]},{k,Keys[c]}]
				] (*This turns the inds list into a series of derivation rules
					For example, {0}->{{l,1}}, {0,0,1}->{{l,2},{z1,1}}*)
			))
			/.Table[zs[[i]]->zs0[[i]],{i,dim}]/.{l->w0} (*Substitute the value zs0 and w0*)
		];
	fw0=fd0[0];
	fww0=fd0[0,0];
	
	(*Get the effective Hessian; See Supplementary Materials for derivation*)
	Lzzs = Array[
		Function[{i,j},
			-1/2*(fd0[i,j]/fw0 - fww0*vels[[i]]*vels[[j]]/(fw0*zs0[[i]]*zs0[[j]])
			 - I*fd0[0,i]*vels[[j]]/(fw0*zs0[[j]])-I*fd0[0,j]*vels[[i]]/(fw0*zs0[[i]])
			 + If[i==j, I*vels[[i]]/(zs0[[i]]^2),0])
		],
		{dim,dim}
	];
	Rlzzs = ArrayFlatten[{{Im[Lzzs],Re[Lzzs]},{Re[Lzzs],-Im[Lzzs]}}];
	{eigv,vs} = Eigensystem[Rlzzs];
	(*Find the ascending directions*)
	vecs = SortBy[
			Transpose[Join[{eigv},Transpose[vs/Sqrt[Abs[eigv]]]]],
			-N[#[[1]]]& (*Sort in descending order of eigenvalues*)
		]
		[[;;dim,2;;]]; (*Get the normalized eigenvectors with positive eigenvalues*)
	
	(*This functions get the path for a single given direction*)
	GetSol[tdir_]:=Module[{res},
		(*Get the flowing direction*)
		vec=tdir . vecs;
		vec=Sqrt[I idir]*(vec[[;;dim]]+I vec[[dim+1;;]]);
		mdz=eps;
		res = 
			If[evaluate < 0,
				{zs,w}, (*If evaluate < 0, just return the functions*)
				{Table[z[evaluate],{z,zs}],w[evaluate]} (*Otherwise, return the value*)
			]
			/.
			NDSolve[
				Join[
					Table[
						(zs[[i]])'[s] ==
							I*idir*((Conjugate[dzs[[i]]]/(Norm[dzs]^2))
								/.Table[z->z[s],{z,zs}]/.{l->w[s]}),
						{i,dim}
					],
					Table[
						zs[[i]][mdz^2] == zs0[[i]]+mdz*vec[[i]],
						{i,dim}
					],
					{w'[s] == 
						(-I*idir*
							Plus@@Table[
								Hzs[[i]]*Conjugate[dzs[[i]]]/(Norm[dzs]^2),
								{i,dim}
							] / Hw
						)
						/.Table[z->z[s],{z,zs}]/.{l->w[s]},
					 w[mdz^2] == 
						w0 + idir*I*mdz^2
						 - I*Plus@@(vels*(mdz*vec/zs0-(mdz*vec/zs0)^2/2))
					}
				],
				zs ~Join~ {w},
				{s,T}
			][[1]];
		res (*Returns res*)
	];
	
	(*This enables evaluating dir as a list for directions*)
	If[ArrayDepth[dir]==1,
		GetSol[dir],
		Table[GetSol[tdir],{tdir,dir}]
	]
]

(*This function flows a saddle point for a fixed distance T*)
Flow[Heq_,dim_,zs0_,w0_,T_,dir_,v_:0,idir_:-1]:=
	If[ArrayDepth[dir]==1,
	
		Module[{zs,w,s,ds,zf,wf,sps,i},
			zs = zs0;
			w = w0;
			s = 0;
			i = 1; (*Iteration count*)
			While[True,
				{zf,wf} = SolvePath[Heq,zs,w,T-s,dir,v,idir];(*Try to flow to s=T*)
				ds = (wf@Domain[])[[1,2]];(*Find out how much we actually flowed*)
				zs = Table[N[f[ds]],{f,zf}]; w = N[wf[ds]];
				s += ds;
				If[s>=T,
					Break[], (*If we did reach s=T, we can exit the loop*)
					If[!ValueQ[sps], sps=SPS[Heq,dim,v]];
					(*If the flow is terminated early, it should have ran into a saddle point*)
					{zs,w} = sps[[
						MinimalBy[
							Range[Length[sps]],
							Norm[Flatten[sps[[i]]-{zs,w}]]
						]
					]];
					i++;
				];
				If[i>10,
					Print["ERR::Too many ran-into-saddle iterations"];
					Abort[]
				]
			];
			{zs,w}
		],
		
		Table[Flow[Heq,dim,zs0,w0,T,tdir,v,idir],{tdir,dir}]
	]


MapNested[fun_,list_]:=Map[If[Head[#]===List,#,fun[#]]&,list,Infinity]


(* ::Text:: *)
(*Converts spherical coordinate into Cartesian coordinate, in arbitrary dimensions*)


SphereToCarteSingle[phis_]:=
Module[{n,sinlist,i},
	n=Length[phis]+1;
	sinlist=ConstantArray[1,n];
	For[i=1,i<n,i++,
		sinlist[[i+1]]=sinlist[[i]]*Sin[phis[[i]]]
	];
	Table[N[sinlist[[i]]*If[i==n,1,Cos[phis[[i]]]]],{i,1,n}]
]

SphereToCarte[phis_]:=
	If[ArrayDepth[phis]==1,
		SphereToCarteSingle[phis],
		Table[SphereToCarteSingle[phi],{phi,phis}]
	]


(* ::Text:: *)
(*Finds the winding number of a given saddle point, they key function of this script.*)


(*Put x1-x2 into the Interval [-Pi,Pi], and return its sign*)
DifSign[x1_,x2_]:=Sign[Mod[x1-x2,2Pi,-Pi]]

(*This function currently works ONLY IN TWO DIMENSIONS*)
IntegrateWinding[vfun_,dim_,prec_:10,tol_:Pi/5,angrange_:{0,2Pi}]:=
	Module[{redo,phis,philist,tocpx,phases,df,difis,len,returnphases},
	
		redo=10^-3;
		phis=Table[Symbol["\[Phi]"<>ToString[i]],{i,dim-1}];
		
		(*Generate a list of phis*)
		philist=Table[{angrange[[1]]+(angrange[[2]]-angrange[[1]])*ind/prec},{ind,0,prec}];
		
		(*Get Log[Abs[zfinal]] for the list of phis*)
		tocpx=#1+I #2&;
		phases=Table[N[Arg[tocpx@@Log[Abs[res]]]],{res,vfun[philist]}];
		
		(*Find the points at which the phase jumps abruptly*)
		df=Differences@phases;
		difis=Flatten@Position[df,x_/;Abs[x]>tol];
		
		len=Length[phases];
		returnphases={};
		(*For each jumping point, find the jumping direction*)
		Module[{j,i,wind,retp},
			For[j=1,j<=Length[difis],j++,
				i = difis[[j]];
				
				(*If df[i]>2Pi-tol, it is clearly a +2Pi jump, and vice versa*)
				If[df[[i]]>2Pi-tol,
					phases = phases-(2Pi*UnitStep[#-(i+1)]&/@Range[len]);
					Continue[]
				];
				If[df[[i]]<-(2Pi-tol),
					phases = phases+(2Pi*UnitStep[#-(i+1)]&/@Range[len]);
					Continue[]
				];
				
				(*If Abs[df[[i]]]<2Pi-tol, the direction is not immediately clear*)
				If[philist[[i+1,1]] - philist[[i,1]] < redo,
					(*If the searching interval is already small enough,
					  use a best guess based on the current jump amplitude*)
					phases = phases + (2Pi * UnitStep[#-(i+1)]&/@Range[len] *
						With[
							(*philist[[i,1]] and philist[[i+1,1]] are the two points closest to the current jumping point
							  We shift them outwards for a distance Sqrt[redo], and look for the slope*)
							{newr = Table[
								N[Arg[tocpx@@Log[Abs[res]]]],
								{res,
									vfun[{{philist[[i,1]]-Sqrt[redo]},{philist[[i+1,1]]+Sqrt[redo]}}]
								}
							]},
							With[
								{fd = 
									(*Average over the slope to the right of philist[[i+1]] and to the left of philist[[i]]*)
									Sign[DifSign[newr[[2]],phases[[i+1]]] + DifSign[phases[[i]],newr[[1]]]]
								},
								(*If fd==0 (i.e. slopes on the two sides are opposite),
									or fd==Sign[df[[i]]] (i.e. slopes on the two sides contradict with the central jump),
									we conclude that there is no jump. If all three jumps are consistent, take this jump.*)
								fd*(1-fd*Sign[df[[i]]])/2
							]
						]
					),
					
					(*Otherwise, we further refine the interval to look for the correct jump amplitude*)
					{wind,retp} = IntegrateWinding[vfun,dim,prec,tol,{N[philist[[i,1]]],N[philist[[i+1,1]]]}];
					(*wind is the total winding within the smaller interval, we account for this in the new phases*)
					phases = phases + (2Pi*Round[wind-df[[i]]/(2Pi)]*UnitStep[#-(i+1)]&/@Range[len]);
					(*We add all the refined researching points to the returnphases list*)
					returnphases = returnphases ~Join~
						Table[{tp[[1]],tp[[2]]+phases[[i]]-retp[[1,2]]},{tp,retp[[2;;-2]]}];
				]
			]
		];
		
		(*Construct a list of all known {phi,phase}*)
		returnphases = 
			Table[{philist[[i,1]],phases[[i]]},{i,Length[philist]}]
				~Join~ returnphases;
		returnphases = SortBy[returnphases,N[#[[1]]]&]; (*And sort in increasing order of phi*)
	
		{(phases[[-1]]-phases[[1]])/(2Pi), returnphases} (*Return the winding number and the returnphases list*)
	]

(*Finds the top of the PBC spectrum*)
FindPBCTop[Heq_,dim_,prec_:10]:=
	Module[{fun,rans,ks,startk,top},
		ks=Ks[dim];
		fun[tks_]:=Module[{l},Max[Im[l]/.Solve[Heq[Exp[I*tks],l]==0,l]]]; (*The Im[H[k]] function*)
		(*Generate a list of random starting points*)
		rans = Table[
			With[{tks=RandomReal[{0,2Pi},dim]},
				{tks,fun[tks]}],
			{i,prec^dim}
		];
		(*Select the maximum among these random points as a trial solution*)
		startk = MaximalBy[rans,#[[2]]&][[1,1]];
		(*Invoke FindMax to find the exact top*)
		top=FindMaximum[{fun[ks],And@@Table[0<=k<2Pi,{k,ks}]},Transpose[{ks,startk}]];
		top[[1]]
	]


(* ::Text:: *)
(*Functions that create outputs*)


SPFlows[Heq_,dim_,v_:0,prec_:10,print_:True,zlvlgiven_:0,maxno_:0]:=
	Module[{vels,sps,zlvl,zs,ks,res},
	
		vels = If[Head[v]===List,v,ConstantArray[v,dim]];
		sps = SPS[Heq,dim,vels];
		zs = Zs[dim];
		zlvl = If[zlvlgiven==0, FindPBCTop[Heq,dim,prec]+1, zlvlgiven];

		maxn = If[maxno==0, Length[sps], maxno];
		curno = 0;
		
		(*Get a list of {windings,phase_information}*)
		res = Table[
			If[zlvl-Lam[sp,vels]>0 && curno < maxn, (*Do the integration only if curno < maxno*)
				intg = IntegrateWinding[
					Map[#[[1]]&,
						Flow[Heq,dim,sp[[1]],sp[[2]],zlvl-Lam[sp,vels],SphereToCarte[#],vels,1]
					]&,
					dim, prec
				];
				If[intg[[1]] != 0, curno++];
				intg,
				{0,"N/A"} (*If lambda >= zlvl, there is nothing to find*)
			],
			{sp,sps}
		];
		
		If[print,
			(*If print is True, print the saddle point information*)
			Print["Saddle Points:"];
			Print[Grid[
				Join[
					{{"z","H[z]","lambda","wind","phases"}},
					Join[
						(*z, H[z], lambda*)
						Table[{sp[[1]],sp[[2]],Lam[sp,vels]},{sp,sps}],
						(*winding and phases*)
						Table[
							{
								a[[1]],
								If[Head[a[[2]]]===String,
									"N/A",
									ListPlot[a[[2]],PlotRange->All,PlotMarkers->{Automatic, Medium},Joined->True]
								]
							},
							{a,res}
						],
						2
					]
				],
				Frame->All
			]],
			(*Otherwise, return the saddle points and their validities*)
			Join[
				(*z, H[z], lambda*)
				Table[{sp[[1]],sp[[2]],Lam[sp,vels]},{sp,sps}],
				(*windings*)
				{Part[#,1]}&/@res,
				2
			]
		]
	]

GetGrowth[Heq_,dim_,vels_,prec_:10,zlvl_:0]:=
	Module[{res,effpos},
		res = SPFlows[Heq,dim,vels,prec,False,zlvl];
		effpos = FirstPosition[
			Table[Round[thisres[[4]]]!=0,{thisres,res}]
			,True
		][[1]];
		If[effpos=="NotFound",
			Print["ERR:No valid saddle point is found for velocity "<>StringRiffle[vels,{"(",",",")"}]];
			Abort[]
		];
		res[[effpos,3]]
	]
	
GrowthList[Heq_,dim_,vs_]:=
	With[{zlvl=FindPBCTop[Heq,dim,20]+1},
		Map[GetGrowth[Heq,dim,#,10,zlvl]&,vs]
	]
	
GrowthBulk[Heq_,dim_,vxl_,vxh_,vyl_,vyh_,intvs_:20]:=
	Module[{vxs,vys,vs},
		vxs=Table[vxl+(vxh-vxl)*i/intvs,{i,0,intvs}];
		vys=Table[vyl+(vyh-vyl)*i/intvs,{i,0,intvs}];
		vs=Flatten[Table[{vx,vy},{vx,vxs},{vy,vys}],1];zlvl=FindPBCTop[Heq,dim,20]+1;
		{vs,GrowthList[Heq,dim,vs]}
	]
	
GrowthEdge[Heq_,dim_,edge_,vl_,vh_,intvs_:20]:=
	Module[{vs,zlvl},
		If[!MemberQ[{"x","y"},ToLowerCase[edge]],
			Print["ERR::'edge' must be either 'x' or 'y'"];
			Abort[]
		];
		vs = Table[vl+(vh-vl)*i/intvs,{i,0,intvs}];
		vs = Table[If[ToLowerCase[edge]=="x",{v,0},{0,v}],{v,vs}];
		{vs,GrowthList[Heq,dim,vs]}
	]

End[]
EndPackage[]

