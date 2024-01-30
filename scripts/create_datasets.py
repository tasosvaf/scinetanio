import os
import sys

#Adding path to libray
dirpath = os.path.dirname(__file__)
parent_dirpath, _ = os.path.split(dirpath)
sys.path.append(parent_dirpath)

from SCINet.utils.utils_ETTh import createETThDatasetGeneral, createGreekDatasetGeneral, getAnioInputOptions, getAnioOptions
  
def main():
    anio_options = getAnioOptions()
    anio_input_options = getAnioInputOptions()
    
    for anio in anio_options:
        for anio_input in anio_input_options:
            createGreekDatasetGeneral(anio, anio_input)
            createETThDatasetGeneral('ETTh1', anio, anio_input)
            createETThDatasetGeneral('ETTh2', anio, anio_input)

if __name__ == "__main__":
    main()