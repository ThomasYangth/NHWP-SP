import os
import json

# Read configurations from Config.json
config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "Config.json")
if not os.path.isfile(config_file):
    raise Exception("Configuration file not found at: " + config_file)
json_file = json.load(open(config_file, 'r'))

print("Initialized with:")
try:
    USE_GPU = json_file["USE_GPU"]
    print("USE_GPU:", USE_GPU)
except Exception as e:
    USE_GPU = False
    print("USE_GPU set to default value: False")

try:
    FFMPEG_PATH = json_file["FFMPEG_PATH"]
    print("FFMPEG_PATH:", FFMPEG_PATH)
except Exception as e:
    FFMPEG_PATH = "NO FFMPEG PROVIDED, CHECK Config.json"
    print("FFMPEG_PATH not found.")

try:
    MMA_CALLER = json_file["MMA_CALLER"]
    print("MMA_CALLER:", MMA_CALLER)
except Exception as e:
    MMA_CALLER = ["NO MATHEMATICA PROVIDED, CHECK Config.json"]
    print("MMA_CALLER not found.")

# Some plotting macro parameters
FONTSIZE = 8
SINGLE_FIGSIZE = (3.2, 2.4)
DOUBLE_FIGSIZE = (6, 2.4)
TRIPLE_FIGSIZE = (7, 2.4)
SQUARE_FIGSIZE = (3.5, 3.5)
SQUARE_SHOW_FIGSIZE = (8, 8)
LINEWIDTH = 1
THICK_LINEWIDTH = 1.5
FONTSET = "cm"
PLT_PARAMS = {"font.size":FONTSIZE, "lines.linewidth":LINEWIDTH, "mathtext.fontset":FONTSET}
