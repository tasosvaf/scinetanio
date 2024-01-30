import os
import sys

#Adding path to libray
dirpath = os.path.dirname(__file__)
parent_dirpath, _ = os.path.split(dirpath)
sys.path.append(parent_dirpath)

from SCINet.utils.utils_ETTh import deleteFilesInFolder, getRepoPath

def main():
    metrics_summary_path = os.path.join(getRepoPath(), "metrics_summary")
    plots_path = os.path.join(getRepoPath(), "plots")
    metrics_tables_path = os.path.join(getRepoPath(), "metrics_tables")
    plots_comparison_folderpath = os.path.join(getRepoPath(), "plots_comparison")

    deleteFilesInFolder(metrics_summary_path)
    deleteFilesInFolder(plots_path)
    deleteFilesInFolder(metrics_tables_path)
    deleteFilesInFolder(plots_comparison_folderpath)

if __name__ == "__main__":
    main()