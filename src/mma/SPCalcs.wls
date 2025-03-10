#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Text:: *)
(*This is the WolframScript file that serves as a connection between the Mathematica script and the Python codes. This script can be called directly from the command line.*)


(* ::Text:: *)
(*Add the current directory to Mathematica path, and load the LV1.wl and LVn.wl script files.*)


dir=If[$InputFileName!="",DirectoryName[$InputFileName],NotebookDirectory[]];
If[MemberQ[$Path,dir],0,AppendTo[$Path,dir]];


(* ::Text:: *)
(*Different systems use different conventions for the command line arguments. For wolframscript, the command line arguments are encoded in $ScriptCommandLine, with index starting at 2. For math -script, arguments are in $CommandLine, with index starting at 4.*)


(* ::Text:: *)
(*Use this for math -script*)


(*CommandLine = $CommandLine;
StartIndex = 4;*)


(* ::Text:: *)
(*Use this for wolframscript*)


(*CommandLine = $ScriptCommandLine;
StartIndex = 2;*)


(* ::Text:: *)
(*Import from file*)


ConfigFile = FileNameJoin[{ParentDirectory[ParentDirectory[dir]], "Config.json"}];
If[!FileExistsQ[ConfigFile], Print["ERR::Config file '"<>ConfigFile<>"' not found!"];Abort[]];
ConfigVals = Import[ConfigFile];
CommandLine = ToExpression["MMA_CL_NAME"/.ConfigVals];
StartIndex = "MMA_CL_INDEX"/.ConfigVals;


(* ::Text:: *)
(*If there is no input parameters, or -help is called, we output the help string.*)


If[Length[CommandLine]<=StartIndex||CommandLine[[StartIndex]]=="-help",
	Print["Got Command Line:"];
	Print[CommandLine];
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


dim = 1;
prec = 20;
edge = 0;
v = 0;
k = Null;
spcDebug = False;

ClearAll[z,l,f];

ToMat[f_]:=If[Head@f===List,f,{{f}}];
Hamexp = ToMat[ToExpression[CommandLine[[StartIndex]]]];
InitHam[processK_:True]:=If[dim==1,
	Hmat[z_]:=Evaluate@Hamexp;Heq[z_,l_]:=Evaluate@CharacteristicPolynomial[Hmat[z],l],
	If[processK && NumericQ[k],
		(* If k is provided, reduce the 2D Hamiltonian to a 1D one *)
		dim = 1;
		If[edge==0 || edge=="x",
			Hmat[z_]:=Evaluate@Hamexp/.{z1->Exp[I k],z2->z},
			Hmat[z_]:=Evaluate@Hamexp/.{z1->z,z2->Exp[I k]}
		];
		Heq[z_,l_]:=Evaluate@CharacteristicPolynomial[Hmat[z],l] ,
	Hmat[z1_,z2_]:=Evaluate@Hamexp;Quiet[Heq[zs_,l_]:=Evaluate@CharacteristicPolynomial[Hmat[zs[[1]],zs[[2]]],l],Part::partd]
	]
]

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

FspfEff[nsp_]:=
Module[{sps,pos,res},
	InitHam[];
	spf = If[dim==1,
		(*One-dimensional case*)
			LV1`SPFlows1[Hmat, v, False, nsp],
			(*Higher-dimensional case*)
			If[dim == 2,
				LVn`SPFlows[Heq, 2, v, 10, False, 0, nsp],
				Print["ERR::-spflow now only supported for dim = 1 or 2"];Abort[]
			]
		];
	effsps = {};
	haveSp = 0;
	For[i=1, haveSp < nsp && i<=Length[spf], i++,
		With[{spinfo = spf[[i]]},
			If[spinfo[[4]] != 0,
				AppendTo[effsps, spinfo]; haveSp++
			]
		]
	];
	Print[
		ExportString[effsps,
			"ExpressionJSON", "Compact"->True
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

Efun[e_,xpairs_]:=
(
InitHam[];
H = Evaluate@(Hmat[#][[1,1]])&;
If[AllTrue[xpairs, #>0&, 2],
	ExportJSON[EigenfunctionL[H, e, xpairs]],
	If[AllTrue[xpairs, #<0&, 2],
		ExportJSON[EigenfunctionR[H, e, xpairs]],
		Print["ERR:Invalid x pair "<>ToString[xpairs]];Abort[]
	]
]
)

EfunBatchK[kdens_,xpairs_] :=
(
	InitHam[False]; (*Initialize the Hamiltonian in its original 2D form*)
	If[spcDebug, Print["Hamiltonian Initialized"]];
	ks = Table[2*Pi*i/kdens,{i,0,kdens-1}];
	EigenF = If[AllTrue[xpairs, #>0&, 2], EigenfunctionL,
		If[AllTrue[xpairs, #<0&, 2], EigenfunctionR,
			Print["ERR:Invalid x pair "<>ToString[xpairs]];Abort[]
		]
	];
	HF[k_] := If[edge==0 || edge=="x",
				Function[z,Evaluate@Hmat[Exp[I k],z]],
			If[edge==1 || edge=="y",
				Function[z,Evaluate@Hmat[z,Exp[I k]]],
				Print["ERR:Invalid edge "<>ToString[edge]];Abort[]
			]
	];
	(* Creates a list of {saddle point energy, {1/Sqrt[H''[zs]]*<x|E><<E|x> for all x pairs}} for each H[k] *)
	ExportJSON[
		Table[
			If[spcDebug, Print["Doing k = "<>ToString[k]]];
			With[{spf = SelectFirst[SPFlows1[HF[k],0,False,1],(#[[4]] != 0)&]}, (* Select the relevant saddle point *)
				If[spcDebug, Print["Got spf: "<>ToString[spf]]];
				If[spcDebug, Print["Passing to EigenF: "<>ToString[{TeXForm[Evaluate@HF[k][zeta][[1,1]]], spf[[2]], xpairs}]]];
				{
					spf[[2]] (* spf[[2]] is the saddle point energy *),
					spf[[5]] * EigenF[(Evaluate@HF[k][#][[1,1]])&, spf[[2]], xpairs] (* spf[[5]] is 1/Sqrt[H''[zs]] *)
				}
			],
			{k,ks}
		]
	]
)



(* ::Text:: *)
(*This is the main function that parses the command line arguments.*)


nowind = StartIndex + 1;

While[True,
If[Length[CommandLine]<nowind,Print["Specify function!"];Break[]];
Switch[CommandLine[[nowind]],
	"-dim",dim=ToExpression[CommandLine[[nowind+1]]];nowind=nowind+2,
	"-prec",prec=ToExpression[CommandLine[[nowind+1]]];nowind=nowind+2,
	"-edge",edge=CommandLine[[nowind+1]];nowind=nowind+2,
	"-v",v=ToExpression[CommandLine[[nowind+1]]];nowind=nowind+2,
	"-k",k=ToExpression[CommandLine[[nowind+1]]];nowind=nowind+2,
	"-debug",spcDebug=True;nowind=nowind+1,
	"-sps",Fsps[False];Break[],
	"-spflow",Fsps[True];Break[],
	"-effspf",FspfEff[ToExpression[CommandLine[[nowind+1]]]];Break[],
	"-lv",Flv[Flatten[ToExpression/@(CommandLine[[nowind+1;;]])]];Break[],
	"-hk",Fhk[Flatten[ToExpression/@(CommandLine[[nowind+1;;]])]];Break[],
	"-ef",Efun[ToExpression[CommandLine[[nowind+1]]],ToExpression[CommandLine[[nowind+2]]]];Break[],
	"-efk",EfunBatchK[ToExpression[CommandLine[[nowind+1]]],ToExpression[CommandLine[[nowind+2]]]];Break[],
	_,Print["No matching command '"<>CommandLine[[nowind]]<>"'!"];Break[]]
]
