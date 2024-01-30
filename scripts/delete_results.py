import os
import sys

#Adding path to libray
dirpath = os.path.dirname(__file__)
parent_dirpath, _ = os.path.split(dirpath)
sys.path.append(parent_dirpath)

from SCINet.utils.utils_ETTh import deleteFilesInFolder, getRepoPath

def main():
    event_path = os.path.join(getRepoPath(),"event","run_ETTh","SCINet")
    err_path = os.path.join(getRepoPath(), "err")
    log_path = os.path.join(getRepoPath(), "log")
    output_path = os.path.join(getRepoPath(), "output")
    predictions_path = os.path.join(getRepoPath(), "predictions")
    metrics_path = os.path.join(getRepoPath(), "metrics")
    group_outputs_path = os.path.join(getRepoPath(), "group_outputs")

    deleteFilesInFolder(event_path)
    deleteFilesInFolder(err_path)
    deleteFilesInFolder(log_path)
    deleteFilesInFolder(output_path)
    deleteFilesInFolder(predictions_path)
    deleteFilesInFolder(metrics_path)
    deleteFilesInFolder(group_outputs_path)

if __name__ == "__main__":
    main()