from imageManage import reduceDICOM, writeDICOM
from Pattern1D import Pattern1D
from Pattern2D import Pattern2D
from Logger import *
from utils import *

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import sys, os

def main():
    logger.info("---------- New Test ----------")
    img = np.array(pydicom.dcmread(f"Images/img411.dcm", force=True).pixel_array, np.uint16) / (1<<16)
    pat = np.array(pydicom.dcmread(f"Images/pat03.dcm", force=True).pixel_array, np.uint16) / (1<<16)
    
    solver = Pattern2D(pat)
    solver.load_detect(F=img, plot=True)
    plt.show()
    
    img = np.array(pydicom.dcmread(f"Images/img410.dcm", force=True).pixel_array, np.uint16) / (1<<16)
    pat = np.array(pydicom.dcmread(f"Images/pat01.dcm", force=True).pixel_array, np.uint16) / (1<<16)
    
    solver = Pattern2D(pat)
    solver.load_detect(F=img, plot=True)
    plt.show()

def main_with_path(imgPath, patPath):
    img = np.array(pydicom.dcmread(imgPath, force=True).pixel_array, np.uint16) / (1<<16)
    pat = np.array(pydicom.dcmread(patPath, force=True).pixel_array, np.uint16) / (1<<16)
    
    solver = Pattern2D(pat)
    solver.load_detect(F=img, plot=True)
    plt.show()

if __name__ == '__main__':
    try:
        main_with_path(sys.argv[1], sys.argv[2])
    except:
        try:
            main_with_path(sys.argv[1], f"Images/pat01.dcm")
        except:
            main()