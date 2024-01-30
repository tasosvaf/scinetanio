import argparse
import os
import sys
import subprocess

#Adding path to libray
dirpath = os.path.dirname(__file__)
parent_dirpath, _ = os.path.split(dirpath)
sys.path.append(parent_dirpath)

from SCINet.utils.utils_ETTh import createFolderInRepoPath, getDatetimeAsString, getRepoPath

def runMultRunGroup(group_case, python_name, arguments):
    n_scripts = len(arguments)
    
    group_outputs_folderpath = createFolderInRepoPath("group_outputs")
    experiment_date = getDatetimeAsString()
    
    script = os.path.join(getRepoPath(),"scripts","mult_run.py")
    group_outputs = [ os.path.join(group_outputs_folderpath, f"group_output_{group_case}_{i}_{experiment_date}.txt") for i in range(n_scripts) ]
    
    processes = []

    for output, args in zip(group_outputs, arguments):
        with open(output, "w") as f:
            p = subprocess.Popen([python_name, script] + args, stdout=f, stderr=f)
            processes.append(p)
            
    print("len(processes): ", len(processes))
    for i, p in enumerate(processes):
        returncode = p.wait()
        print(f"process: {i} - returncode: {returncode}")

def runGroupCase(args):
    group_case = args.group_case
    
    if group_case == 1:
        # 16 runs per gpu - 4 gpus
        arguments = [
            ["--python_name", args.python_name, "--run_case", "11", "--gpu", "0"], 
            ["--python_name", args.python_name, "--run_case", "12", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "13", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "14", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "15", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "16", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "17", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "18", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "19", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "20", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "21", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "22", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "23", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "24", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "25", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "26", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "27", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "28", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "29", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "30", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "41", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "42", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "43", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "44", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "45", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "46", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "47", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "48", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "49", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "50", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "51", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "52", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "111", "--gpu", "0"], 
            ["--python_name", args.python_name, "--run_case", "112", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "113", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "114", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "115", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "116", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "117", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "118", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "119", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "120", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "121", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "122", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "123", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "124", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "125", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "126", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "127", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "128", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "129", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "130", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "141", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "142", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "143", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "144", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "145", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "146", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "147", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "148", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "149", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "150", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "151", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "152", "--gpu", "3"]]
        runMultRunGroup(group_case, args.python_name, arguments)
    elif group_case == 2:
        # 8 runs per gpu - 4 gpus
        arguments = [
            ["--python_name", args.python_name, "--run_case", "11", "111", "--gpu", "0"], 
            ["--python_name", args.python_name, "--run_case", "12", "112", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "13", "113", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "14", "114", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "15", "115", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "16", "116", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "17", "117", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "18", "118", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "19", "119", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "20", "120", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "21", "121", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "22", "122", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "23", "123", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "24", "124", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "25", "125", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "26", "126", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "27", "127", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "28", "128", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "29", "129", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "30", "130", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "41", "141", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "42", "142", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "43", "143", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "44", "144", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "45", "145", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "46", "146", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "47", "147", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "48", "148", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "49", "149", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "50", "150", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "51", "151", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "52", "152", "--gpu", "3"]]
        runMultRunGroup(group_case, args.python_name, arguments)
    elif group_case == 3:
        # 5 runs per gpu - 4 gpus
        arguments = [
            ["--python_name", args.python_name, "--run_case", "11", "12", "13", "--gpu", "0"], 
            ["--python_name", args.python_name, "--run_case", "14", "15", "16", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "17", "18", "19", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "20", "21", "22", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "23", "24", "25", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "26", "27", "28", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "29", "30", "41", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "42", "43", "44", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "45", "46", "47", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "48", "49", "50", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "51", "52", "111", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "112", "113", "114", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "115", "116", "117", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "118", "119", "120", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "121", "122", "123", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "124", "125", "126", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "127", "128", "129", "130", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "141", "142", "143", "144", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "145", "146", "147", "148", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "149", "150", "151", "152", "--gpu", "3"]]
        runMultRunGroup(group_case, args.python_name, arguments)
    elif group_case == 4:
        # 4 runs per gpu - 4 gpus
        arguments = [
            ["--python_name", args.python_name, "--run_case", "11", "12", "13", "14", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "15", "16", "17", "18", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "19", "20", "21", "22", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "23", "24", "25", "26", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "27", "28", "29", "30", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "41", "42", "43", "44", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "45", "46", "47", "48", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "49", "50", "51", "52", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "111", "112", "113", "114", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "115", "116", "117", "118", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "119", "120", "121", "122", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "123", "124", "125", "126", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "127", "128", "129", "130", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "141", "142", "143", "144", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "145", "146", "147", "148", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "149", "150", "151", "152", "--gpu", "3"]]
        runMultRunGroup(group_case, args.python_name, arguments)
    elif group_case == 5:
        # 2 runs per gpu - 4 gpus
        arguments = [
            ["--python_name", args.python_name, "--run_case", "11", "12", "13", "14", "15", "16", "17", "18", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "19", "20", "21", "22", "23", "24", "25", "26", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "27", "28", "29", "30", "41", "42", "43", "44", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "45", "46", "47", "48", "49", "50", "51", "52", "--gpu", "3"],
            ["--python_name", args.python_name, "--run_case", "111", "112", "113", "114", "115", "116", "117", "118", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "119", "120", "121", "122", "123", "124", "125", "126", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "127", "128", "129", "130", "141", "142", "143", "144", "--gpu", "2"],
            ["--python_name", args.python_name, "--run_case", "145", "146", "147", "148", "149", "150", "151", "152", "--gpu", "3"]]
        runMultRunGroup(group_case, args.python_name, arguments)
    elif group_case == 6:
        # 1 runs per gpu - 4 gpus
        arguments = [
            ["--python_name", args.python_name, "--run_case", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "27", "28", "29", "30", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "111", "112", "113", "114", "115", "116", "117", "118", "119", "120", "121", "122", "123", "124", "125", "126", "--gpu", "2"],
            ['--python_name', args.python_name, '--run_case', '127', '128', '129', '130', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '--gpu', '3']]
        runMultRunGroup(group_case, args.python_name, arguments)
    elif group_case == 7:
        # 1 runs per gpu - 3 gpus
        arguments = [
            ["--python_name", args.python_name, "--run_case", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "41", "42", "--gpu", "0"],
            ["--python_name", args.python_name, "--run_case", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "111", "112", "113", "114", "115", "116", "117", "118", "119", "120", "121", "--gpu", "1"],
            ["--python_name", args.python_name, "--run_case", "122", "123", "124", "125", "126", '127', '128', '129', '130', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', "--gpu", "2"]]
        runMultRunGroup(group_case, args.python_name, arguments)
    elif group_case == 8:
        # Test to see if multiple gpus work correctly
        arguments = [
            ["--python_name", args.python_name, "--run_case", "-12", "--gpu", "0"], 
            ["--python_name", args.python_name, "--run_case", "-13", "--gpu", "1"]]
        runMultRunGroup(group_case, args.python_name, arguments)
    elif group_case == 9:
        arguments = [
            ["--python_name", args.python_name, "--run_case", "-41", "-42", "-43"]]
        runMultRunGroup(group_case, args.python_name, arguments)

def main():
    parser = argparse.ArgumentParser(description='Group run script')

    parser.add_argument('--group_case', type=int, default=1)
    parser.add_argument('--python_name',type=str, default='python', choices=['python', 'python3'])

    args = parser.parse_args()

    print(f"group_case = {args.group_case}")
    
    runGroupCase(args)
    
if __name__ == "__main__":
    main()