#import Sc0806.Runscript1114 as script
#import Sc0806.RunScript1023 as script
#import Sc0806.Runscript0307 as script
#import Sc0806.RunScript1122 as script
import RunScripts2023.RunScript0611 as script

from os import mkdir, chdir
from os.path import isdir

if __name__ == "__main__":

    sync = False
    data_dir = "NData0619" # This is the directory name under which output files will be saved

    if not sync:
        data_dir = "../"+data_dir
    
    if not isdir(data_dir):
        mkdir(data_dir)
 
    chdir(data_dir)
    script.run()
    