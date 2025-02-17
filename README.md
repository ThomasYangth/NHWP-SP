# NHWP-SP

*Non-Hermitian Wave Packet - Saddle Point*

By Tian-Hua Yang [yangth@princeton.edu]

This is a Python & Mathematica package to numerically perform time evolution of wave packets governed by non-Hermitian lattice Hamiltonians, and theoretically calculate the saddle points that would govern the long-time behavior of such evolutions.

See our paper on arxiv: *T.-H. Yang and Chen Fang, Real time edge dynamics of non-Hermitian lattices*, arXiv:2503.xxxxx (to be posted soon). Please kindly cite it if you find this package helpful in your research.

## Dependencies

- numpy, scipy, matplotlib

- *(Optional)* Wolfram Mathematica in the command line. Needed if you want to incorporate saddle point calculations. Either install "wolframscript" or "math". Remember to set the `MMA_CL_NAME` and `MMA_CL_INDEX` arguments in `Config.json`.

- *(Optional)* cupy. Needed if you want to use GPU to accelerate time evolution computations. With cupy installed, enable GPU acceleration by setting `USE_GPU` to `true` in `Config.json`.

- *(Optional)* ffmpeg. Needed if you want to animate wave function profiles. After installing, set the `FFMPEG_PATH` argument in `Config.json` to the absolute path of the executable `ffmpeg` file.

## Structure

I run the codes by writing the function I want to run in a separate file (e.g. some file in the `examples` folder), import this file into `main.py`, and then running `main.py`. In `main.py` I do a `chdir` before running the script, to generate the plots in a directory separate from the codes.

## Examples

`examples/compare_sp.py` contains function that generate various time-evolution plots presented in the paper.

`examples/saddlepoints.nb` contains function that illustrate how to use the Mathematica scripts provided in `LV1.wl` and `LVn.wl` to calculate saddle points of a Hamiltonian.
