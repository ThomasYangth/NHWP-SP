(* ::Package:: *)

(*Include LV1.wl*)
dir=If[$InputFileName!="",DirectoryName[$InputFileName],NotebookDirectory[]];
If[MemberQ[$Path,dir],0,AppendTo[$Path,dir]];
Get["LV1.wl"];

BeginPackage["Eigenfun`"];

EigenfunctionL::usage = "[Ham,E,x1,x2] -> <x1|E><<E|x2>";
EigenfunctionR::usage = "[Ham,E,x1,x2] -> <L+1-x1|E><<E|L+1-x2>";
FlowBetas::usage = "FlowBetas";

Begin["`Private`"];

(*Judge whether expr[var] is singular at var=0*)
Judge1[expr_,var_]:=Quiet[With[{tval=expr/.var->0},tval===ComplexInfinity||tval===Indeterminate]]
(*Find the highest negative exponent of var in Fun*)
NegDegree1[Fun_,var_]:=Module[{i,Nowf},i=0;Nowf=Fun;While[Judge1[Nowf,var],Nowf=Expand[Nowf*var];i+=1];i]

(* Use the bisection method to look for the point where fun changes sign in the given interval *)
searchSignChange[fun_,x1_,x2_,epsi_:0.01,iter_:10,givenlfv_:None,givenrfv_:None]:=
Module[{lfv,rfv},
	lfv =If[NumericQ[givenlfv],givenlfv, fun[x1]];
	rfv = If[NumericQ[givenrfv],givenrfv, fun[x2]];
	If[lfv*rfv > 0, {},
		With[{x3=(x1+x2)/2, mfv=fun[(x1+x2)/2]},
			If[x2-x1 < epsi*2^-(iter),
				{x3},
				searchSignChange[fun,x1,x3,epsi,iter,lfv,mfv]~Join~searchSignChange[fun,x3,x2,epsi,iter,mfv,rfv]
			]
		]
	]
]

(* Given the function that gives the GBZ gap and its derivative, look for all points on the GBZ *)
searchGbzTouching[gbzfun_,derfun_,x1_,x2_,epsi_:0.01,iter_:10]:=
Module[{xs,ys,signchanges},
	xs = Table[x1+i*epsi,{i,-1,Round[(x2-x1)/epsi]}];
	ys = derfun /@ xs;
	signchanges = Join @@ Table[searchSignChange[derfun,xs[[i]],xs[[i+1]],epsi,iter,ys[[i]],ys[[i+1]]], {i,1,Length[xs]-1}];
	Select[signchanges, (gbzfun[#] < 10*epsi*2^(-iter))&]
]

(***
Given an E, flow the solutions beta to until it touches the GBZ
And then return the betas at E, sorted by the modulus of their flowed values.
Would print an error if the flow doesn't touch the GBZ, i.e., if the saddle point is not relevant.
*)
FlowBetas[Ham_, En_?NumericQ, m_]:=
Module[{z, betas, degpos, pbctop, gbzgapfun, gbzgapfunder, gbztouchings, betaorders},
	betas = Quiet[z/.Solve[Ham[z]==En,z], Solve::ratnz];
	betas = SortBy[betas, Abs];
	(* Find which set of roots are degenerate *)
	degpos = With[{bdifs=Abs[betas][[2;;]]-Abs[betas][[;;Length[betas]-1]]},
		Ordering[bdifs,1][[1]]];
	pbctop = Quiet[NMaximize[Im[Ham[Exp[I*k]]],{k,0,2Pi}]][[1]] + 0.1;
	gbzgapfun[t_?NumericQ] := With[
		{bs = SortBy[Quiet[z/.Solve[Ham[z]==En+I*t,z], Solve::ratnz], Abs]},
		Abs[bs[[m+1]]]-Abs[bs[[m]]]
	];
	gbzgapfunder[t_?NumericQ] := With[
		{bs = SortBy[Quiet[z/.Solve[Ham[z]==En+I*t,z], Solve::ratnz], Abs]},
		With[{bm=bs[[m]],bm1=bs[[m+1]]},
		Re[I*Conjugate[bm1]/Ham'[bm1]]/Abs[bm1] - Re[I*Conjugate[bm]/Ham'[bm]]/Abs[bm]
	]
	];
	(* Find all energies at which we will touch the GBZ, excluding En itself *)
	gbztouchings = searchGbzTouching[gbzgapfun, gbzgapfunder, 10^-4, pbctop-Im[En]];
	If[Length[gbztouchings]==0 && degpos != m, Print[(
		"ERR: Saddle point not relevant: No gbz touchings, degpos is "<>ToString[degpos]<>", neg degree "<>ToString[m]<>"\n"
		<>"Energy "<>ToString[En]<>", Roots "<>ToString[betas]<>"\n"
		<>"For Hamiltonian "<>ToString[TeXForm[Ham[z]]])];
		Exit[]
	];
	(* Return a table that contains the orderings of the flowed betas *)
	betaorders = Table[
		Ordering[
			Table[
				(* Flow each element upwards to the GBZ toucing position *)
				Abs[Flow1[(#2-Ham[#1])&, betas[[i]], En, t, Sign[i-0.5-degpos], 0, 1][[1]]]
				(* Sign[i-0.5-degpos] ensures that the two degenerate roots flow in opposite directions*)
				(* For other roots, dir does not matter.*)
				(* This creates a table containing the Abs of flowed z's*)
			,{i, Length[betas]}]
		]
	,{t, gbztouchings}];
	If[degpos==m, {Range[1,Length[betas]]}~Join~betaorders, betaorders]
]

EigenfunctionL[Ham_,En_?NumericQ,xslst_,eps_:10^-5]:=
Module[{sps,spEs,e,m,z,betas,degpos,bos},
	(* First correct E to the actual saddle point energy *)
	sps = Quiet[z/.Solve[Ham'[z]==0,z], Solve::ratnz];
	spEs = Ham /@ sps;
	e = MinimalBy[spEs, Abs[#-En]&][[1]];
	m = NegDegree1[Ham[z],z];
	betas = Quiet[z/.Solve[Ham[z]==e,z], Solve::ratnz];
	betas = SortBy[betas,Abs];
	degpos = With[{bdifs=Abs[betas][[2;;]]-Abs[betas][[;;Length[betas]-1]]},
			Ordering[bdifs,1][[1]]];
	(* Find all betaorderings *)
	bos = FlowBetas[Ham, e, m];
	If[Length[bos]==0, Print["ERR: Saddle point not relevant."]; Exit[]];
	Table[
		With[{f=EigfunL[Ham,e,xt[[1]],xt[[2]],m,bos,degpos]},
			(* We average over exchange of the two degenerate roots to rule out Sqrt[E] terms *)
			(f[eps,True]+f[eps,False]-f[-eps,True]-f[-eps,False])/(4eps)
		]
		,{xt,xslst}
	]
]

(*Returns L*<x1|E><E|x2> for x1, x2 near the left edge*)
(*Indexing starts at x=1*)
EigfunL[Ham_,En_?NumericQ,x1_,x2_,m_,bos_,degpos_]:=
Module[{Eigfun},
	(* For each betaordering, calculate the wave function, and then add them together *)
	Eigfun[e_,exch_:False]:=
	Module[{bm,bmp,bL,bR,amp,aL,atmp,atR,thisbetas},
		thisbetas = SortBy[Quiet[z/.Solve[Ham[z]==En+e,z], Solve::ratnz], Abs];
		(* Insert exchange of the two roots degenerate at this saddle point *)
		If[exch, thisbetas[[degpos;;degpos+1]] = thisbetas[[degpos+1;;degpos;;-1]]];
		Plus @@ Table[
			bm=thisbetas[[ord[[m]]]];
			bmp=thisbetas[[ord[[m+1]]]];
			bL=thisbetas[[ord[[;;m-1]]]];
			bR=thisbetas[[ord[[m+2;;]]]];
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
			N[
				(Plus@@(bL^(x1)*aL)+bm^x1+amp*bmp^x1)
				*(Plus@@(bR^(-x2)*atR)+bm^(-x2)+atmp*bmp^(-x2))
				/(1+amp*atmp)
			]
		,{ord,bos}]
	];
	Eigfun
]

EigenfunctionR[Ham_,En_?NumericQ,xslst_]:=
	EigenfunctionL[(Ham[1/#])&,En,-1*xslst]

End[];
EndPackage[];

