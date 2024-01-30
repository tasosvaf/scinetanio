import os
import sys

#Adding path to libray
dirpath = os.path.dirname(__file__)
parent_dirpath, _ = os.path.split(dirpath)
sys.path.append(parent_dirpath)

from SCINet.utils.utils_ETTh import deleteFileIfExist, getListOfFilesAsFilepaths, getRepoPath
    
def main():
    folderpath = os.path.join(getRepoPath(),"datasets","ETT-data")
    filepaths = getListOfFilesAsFilepaths(folderpath)
    datasets_filepaths = [filepath for filepath in filepaths if 'greek_energy' or 'ETTh' in filepath]
    
    for filepath in datasets_filepaths:
        deleteFileIfExist(filepath)

if __name__ == "__main__":
    main()