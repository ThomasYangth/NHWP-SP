from examples.compare_sp import *
from os import chdir

chdir("Datas/1D")

tryRun()
exit()

run1DModels()

chdir("../2D")
run2DModels()