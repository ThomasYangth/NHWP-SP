(* ::Package:: *)

BeginPackage["Eigenfun`"];

EigenfunctionL::usage = "[Ham,E,x1,x2] -> <x1|E><<E|x2>";
EigenfunctionR::usage = "[Ham,E,x1,x2] -> <L+1-x1|E><<E|L+1-x2>";

Begin["`Private`"];

(*Judge whether expr[var] is singular at var=0*)
Judge1[expr_,var_]:=Quiet[With[{tval=expr/.var->0},tval===ComplexInfinity||tval===Indeterminate]]
(*Find the highest negative exponent of var in Fun*)
NegDegree1[Fun_,var_]:=Module[{i,Nowf},i=0;Nowf=Fun;While[Judge1[Nowf,var],Nowf=Expand[Nowf*var];i+=1];i]
(*Returns L*<x1|E><E|x2> for x1, x2 near the left edge*)
(*Indexing starts at x=1*)
EigenfunctionL[Ham_,En_?NumericQ,x1_,x2_]:=
Module[{z,betas,m,bm,bmp,bL,bR,amp,aL,atmp,atR},
	m = NegDegree1[Ham[z],z];
	betas = Quiet[z/.Solve[Ham[z]==En,z], Solve::ratnz];
	betas = SortBy[betas,Abs];
	bm=betas[[m]];
	bmp=betas[[m+1]];
	bL=betas[[;;m-1]];
	bR=betas[[m+2;;]];
	(*Set am and \tilde am = 1*)
	amp = If[Length[bL]<=0,
	-1,
	-Product[(b/bm-1)/(b/bmp-1),{b,bL}]
	];
	atmp = If[Length[bR]<=0,
	-1,
	-Product[(bm-b)/(bmp-b),{b,bR}]
	];
	(*Solve for the rest of a and \tilde a*)
	aL = If[Length[bL]<=0,
	0,
	With[{xs = Range[0,-(Length[bL]-1),-1]},
	-Inverse[Transpose[VandermondeMatrix[(bL)^-1]]] . (bm^xs + amp*bmp^xs)]
	];
	atR = If[Length[bR]<=0,
	0,
	With[{xs = Range[0,(Length[bR]-1)]},
	-Inverse[Transpose[VandermondeMatrix[bR]]] . (bm^xs + atmp*bmp^xs)]
	];
	(*Calculate the final wave function*)
	N[(Plus@@(bL^(x1)*aL)+bm^x1+amp*bmp^x1)
	* (Plus@@(bR^(-x2)*atR)+bm^(-x2)+atmp*bmp^(-x2))
	/(1+amp*atmp)]
]
EigenfunctionR[Ham_,En_,x1_,x2_]:=
	EigenfunctionL[(Ham[1/#])&,En,x1,x2]

End[];
EndPackage[];





