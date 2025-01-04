from examples.compare_sp import *
from os import chdir
from sys import argv

chdir("Datas")

#runExoticModel()
run2D(argv[1], argv[2], argv[3].lower()=="true")