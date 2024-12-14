#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Text:: *)
(*This is the WolframScript file that serves as a connection between the Mathematica script and the Python codes. This script can be called directly from the command line.*)


(* ::Text:: *)
(*Add the current directory to Mathematica path, and load the LV1.wl and LVn.wl script files.*)


dir=If[$InputFileName!="",DirectoryName[$InputFileName],NotebookDirectory[]];
If[MemberQ[$Path,dir],0,AppendTo[$Path,dir]];


(* ::Text:: *)
(*If there is no input parameters, or -help is called, we output the help string.*)


If[Length[$ScriptCommandLine]==1||$ScriptCommandLine[[2]]=="-help",
	Print[Import["README.txt"]];
	Exit[]
]


Get["LV1.wl"];
Get["LVn.wl"];
Get["Eigenfun.wl"];


ExportJSON[obj_]:=
Print[
		ExportString[
			obj,
			"ExpressionJSON",
			"Compact"->True
		]
	]


(* ::Text:: *)
(*These are a list of parameters that are potentially used. They are set to some initial values, and may be modified by, for example, -dim 2 in the command line input.*)


unitary = False;
dim = 1;
prec = 20;
edge = 0;
v = 0;

ClearAll[z,l,f];

ToMat[f_]:=If[Head@f===List,f,{{f}}];
Hamexp = ToMat[ToExpression[$ScriptCommandLine[[2]]]];
InitHam[]:=If[dim==1,
	Hmat[z_]:=Evaluate@Hamexp;Heq[z_,l_]:=Evaluate@CharacteristicPolynomial[Hmat[z],l],
	Hmat[z1_,z2_]:=Evaluate@Hamexp;Quiet[Heq[zs_,l_]:=Evaluate@CharacteristicPolynomial[Hmat[zs[[1]],zs[[2]]],l],Part::partd]]

Fsps[flow_]:=
Module[{sps,pos,res},
	InitHam[];
	If[dim==1,
		(*One-dimensional case*)
		Print[
			ExportString[
				If[flow,
					LV1`SPFlows1[Hmat, v, False],
					LV1`SPS1[Heq,v]
				],
				"ExpressionJSON", "Compact"->True
			]
		],
		
		(*Higher-dimensional case*)
		Print[
			ExportString[
				If[flow,
					If[dim == 2,
						LVn`SPFlows[Heq, 2, v, 10, False],
						Print["ERR::-spflow now only supported for dim = 1 or 2"];Abort[]
					],
					LVn`SPS[Heq,dim,v]
				],
				"ExpressionJSON", "Compact"->True
			]
		]
	]
]

Flv[vbounds_]:=
Module[{vs,vs1,vs2},
	Switch[dim,
	
		1,
			Switch[Length[vbounds],
				1,
					vs1 = Table[Abs[vbounds[[1]]]*(2i-prec)/prec, {i,0,prec}],
				2,
					vs1 = Table[((prec-i)*vbounds[[1]]+i*vbounds[[2]])/prec, {i,0,prec}],
				_,
					Print["ERR::Give the v range by calling -lv! Usage: -lv v means [-v,v], and -lv v1 v2 means [v1,v2]"];
					Abort[]
			],
		
		2,
			Switch[Length[vbounds],
				1,
					vs1 = Table[Abs[vbounds[[1]]]*(2i-prec)/prec, {i,0,prec}];
					vs2 = vs1,
				2,
					vs1 = Table[((prec-i)*vbounds[[1]]+i*vbounds[[2]])/prec, {i,0,prec}];
					vs2 = vs1,
				4,
					vs1 = Table[((prec-i)*vbounds[[1]]+i*vbounds[[2]])/prec, {i,0,prec}];
					vs2 = Table[((prec-i)*vbounds[[3]]+i*vbounds[[4]])/prec, {i,0,prec}],
				_,
					Print["ERR::Give the v range by calling -lv! Usage: -lv v means [-v,v] for both directions; -lv v1 v2 means [v1,v2] for both directions; -lv v1 v2 v3 v4 means [v1,v2] for x-direction, and [v3,v4] for y-direction."];
					Abort[]
			],
			
		_,
			Print["ERR::-lv now only supported for dim = 1 or 2."];
			Abort[]
	];
	InitHam[];
	If[dim == 1,
		Print[ExportString[LV1`GrowthList1[Hmat,vs1],"ExpressionJSON","Compact"->True]],
		If[edge=="x"||edge=="1",
			vs = Table[{v,0},{v,vs1}],
		If[edge=="y"||edge=="2",
			vs = Table[{0,v},{v,vs1}],
		If[edge==0,
			vs = Flatten[Table[{vx,vy},{vx,vs1},{vy,vs2}],1],
			Print["ERR::'edge' must be either 'x'/'1' or 'y'/'2'"];Abort[]
		]]];
		Print[ExportString[LVn`GrowthList[Heq,2,vs],"ExpressionJSON","Compact"->True]]
	]
]

Fhk[krange_]:=
Module[{ks,k2s,Hmat1,Heq1},
	ks = Switch[Length[krange],
		0, Table[2*Pi*i/prec,{i,0,prec}]-Pi,
		1, Table[2*krange[[1]]*i/prec,{i,0,prec}]-krange[[1]],
		2, Table[(krange[[2]]-krange[[1]])*i/prec,{i,0,prec}]+krange[[1]],
		_, Print["ERR::Options for -hk must be either [k_max] or [k_min] [k_max]"];Abort[]
	];
	InitHam[];
	If[dim==1,
		(*One Dimension*)
		Print[ExportString[
			Table[
				MaximalBy[
					l/.NSolve[Heq[Exp[I*k],l]==0,l],
					Im
				][[1]],
				{k,ks}
			]
			,"ExpressionJSON","Compact"->True
		]]
		,
		(*Two Dimensions*)
		If[edge==0,
		
			k2s = Flatten[Table[{kx,ky},{kx,ks},{ky,ks}],1];
			Print[ExportString[
				Table[
					MaximalBy[
						l/.NSolve[Heq[Exp[I*k],l]==0,l],
						Im
					][[1]],
					{k,k2s}
				]
				,"ExpressionJSON","Compact"->True
			]]
			
			,
			
			If[edge=="x"||edge=="1",
				Hmat1[k_] := (Evaluate@Hmat[Exp[I*k],#]&);
				Heq1[k_] := (Evaluate@Heq[{Exp[I*k],#1},#2]&),
			If[edge=="y"||edge=="2",
				Hmat1[k_] := (Evaluate@Hmat[#,Exp[I*k]]&);
				Heq1[k_] := (Evaluate@Heq[{#1,Exp[I*k]},#2]&),
				(*If edge doesn't match any options*)
				Print["ERR::'edge' must be either 'x'/'1' or 'y'/'2'"];Abort[]
			]];
			
			Print[ExportString[
				Table[
					MaximalBy[
						Select[SPFlows1[Hmat1[k],0,False],#[[4]]&],
						Im[#[[3]]]&
					][[1,2]],
					{k,ks}
				]
				,"ExpressionJSON","Compact"->True
			]]
			
		];
	]
]

Efun[E_,xpairs_]:=
(
InitHam[];
H[z_]:=Evaluate@(Hmat[z][[1,1]]);
ExportJSON[
	Table[
		If[Length[xs]!=2 || Sign[xs[[1]]]!=Sign[xs[[2]]],
			Print["ERR:Invalid x pair "<>ToString[xs]];
			Abort[]
		];
		If[xs[[1]] > 0,
			Eigenfun`EigenfunctionL[H, E, xs[[1]], xs[[2]]],
			Eigenfun`EigenfunctionR[H, E, -xs[[1]], -xs[[2]]]
		]
	,
	{xs,xpairs}
	]
]
)

EfunD[En_,xpairs_]:=
Module[{evar,MyND},
InitHam[];
Needs["NumericalCalculus`"];
H[z_]:=Evaluate@(Hmat[z][[1,1]]);
MyND[fun_,x_]:=
	(
	ClearAll[eps];
	eps = 10^-5;
	(fun[x+eps]-fun[x-eps])/(2*eps)
	);
ExportJSON[
	Table[
		If[Length[xs]!=2 || Sign[xs[[1]]]!=Sign[xs[[2]]],
			Print["ERR:Invalid x pair "<>ToString[xs]];
			Abort[]
		];
		If[xs[[1]] > 0,
			MyND[(Evaluate@Eigenfun`EigenfunctionL[H, #, xs[[1]], xs[[2]]]&), En],
			MyND[(Evaluate@Eigenfun`EigenfunctionR[H, #, -xs[[1]], -xs[[2]]]&), En]
			(*N[Limit[D[Eigenfun`EigenfunctionL[H, evar, xs[[1]], xs[[2]]],evar],evar->E]],
			N[Limit[D[Eigenfun`EigenfunctionR[H, evar, -xs[[1]], -xs[[2]]],evar],evar->E]]*)
		]
	,
	{xs,xpairs}
	]
]
]


nowind = 3;

While[True,
If[Length[$ScriptCommandLine]<nowind,Print["Specify function!"];Break[]];
Switch[$ScriptCommandLine[[nowind]],
	"-dim",dim=ToExpression[$ScriptCommandLine[[nowind+1]]];nowind=nowind+2,
	"-prec",prec=ToExpression[$ScriptCommandLine[[nowind+1]]];nowind=nowind+2,
	"-edge",edge=$ScriptCommandLine[[nowind+1]];nowind=nowind+2,
	"-v",v=ToExpression[$ScriptCommandLine[[nowind+1]]];nowind=nowind+2,
	"-unitary",unitary=True;nowind=nowind+1,
	"-sps",Fsps[False];Break[],
	"-spflow",Fsps[True];Break[],
	"-lv",Flv[Flatten[ToExpression/@($ScriptCommandLine[[nowind+1;;]])]];Break[],
	"-hk",Fhk[Flatten[ToExpression/@($ScriptCommandLine[[nowind+1;;]])]];Break[],
	"-ef",Efun[ToExpression[$ScriptCommandLine[[nowind+1]]],ToExpression[$ScriptCommandLine[[nowind+2]]]];Break[],
	"-efd",EfunD[ToExpression[$ScriptCommandLine[[nowind+1]]],ToExpression[$ScriptCommandLine[[nowind+2]]]];Break[],
	_,Print["No matching command '"<>$ScriptCommandLine[[nowind]]<>"'!"];Break[]]
]
