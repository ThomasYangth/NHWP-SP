#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Text:: *)
(*This script helps the Python code handle differential equations to arbitrary precision. The input parameter gives the file name which contains the Hamiltonian and the initial vector, and the times at which we output should be evaluated.*)


SepArray[psi_]:=
	With[
		{psiA = Round[(Log10[Abs[psi]])/.{-Infinity->-9223372036854775808}]},
		{psiA, psi/(10^psiA)}/.{Indeterminate->0}
	]


inname = $ScriptCommandLine[[2]];
precision = ToExpression[$ScriptCommandLine[[3]]];
times = ToExpression[$ScriptCommandLine[[4]]];

Ham = BinaryReadList[inname<>"H.dat", "Complex64"];
psiA = BinaryReadList[inname<>"A.dat", "Integer64"];
psiP = BinaryReadList[inname<>"P.dat", "Complex64"];

Quiet[

psiamp = (10^psiA)/.{Underflow[]->0};
psi = SetPrecision[psiamp*psiP, precision];
Ham = SetPrecision[ArrayReshape[Ham,{Length[psi],Length[psi]}], precision];

psi = v /. NDSolve[
	{I v'[t]==Ham . v[t], v[0]==psi},
	v, {t,0,Last[times]},
	WorkingPrecision -> precision,
	MaxSteps -> Infinity
][[1]];

outA = {};
outP = {};
For[i=1, i<=Length[times], i=i+1,
	{psiA,psiP} = SepArray[psi[times[[i]]]];
	outA = Join[outA,psiA];
	outP = Join[outP,psiP];
];

BinaryWrite[inname<>"TA.dat", outA, "Integer64"];
BinaryWrite[inname<>"TP.dat", outP, "Complex64"];
,
{General::munfl,General::ovfl}
]

Print["Success"];
