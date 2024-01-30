import argparse
import os
import sys
from timeit import default_timer as timer
from datetime import timedelta

#Adding path to libray
dirpath = os.path.dirname(__file__)
parent_dirpath, _ = os.path.split(dirpath)
sys.path.append(parent_dirpath)

from SCINet.utils.utils_ETTh import deleteFileIfIsFalsePositiveErrorFile, isExperimentAlreadyRunned
from SCINet.utils.utils_ETTh import getExperimentName
from SCINet.utils.utils_ETTh import createFolderInRepoPath
from SCINet.utils.utils_ETTh import print_output_in_specific_file
from SCINet.utils.utils_ETTh import getRepoPath

# Create an object with arguments per different run
class RunArgs:
    def __init__(self, mult_run_args, index):
        self.run_case = mult_run_args.run_cases[index]
        self.python_name = mult_run_args.python_name
        self.log = mult_run_args.log
        self.run_same_experiment = mult_run_args.run_same_experiment

def getLogFilepath(log, log_type, experiment_name):
    if log == 'No':
        return None
    
    log_type_folder = createFolderInRepoPath(log_type)
    log_type_filename = '{}_{}.txt'.format(log_type, experiment_name)
    log_type_filepath = os.path.join(log_type_folder, log_type_filename)
    
    return log_type_filepath

def printImportantMessage(message, filepath):
   print_output_in_specific_file(f"----------------------------------------{message}------------------------------------", filepath)

def runCommandWithTimeMeasurement(command, filepath):
    start_time = timer()
    
    os.system(command)
    
    end_time = timer()
    print_output_in_specific_file(f"Run experiment in {timedelta(seconds = end_time - start_time)}", filepath)

def printBeforeRunCommand(command, run_case, log_filepath):
    print_output_in_specific_file(f"Run case: {run_case}", log_filepath)
    printImportantMessage("Start experiment", log_filepath)
    print_output_in_specific_file(f"Experiment Command: {command}", log_filepath)
    
def printAfterRunCommand(command, log_filepath, err_filepath):
    print_output_in_specific_file(f"Experiment Command: {command}", log_filepath)
    printImportantMessage("End experiment", log_filepath)
    deleteFileIfIsFalsePositiveErrorFile(err_filepath)   
       
def runExperiment(command, run_case, log_filepath, err_filepath, is_experiment_already_runned, run_same_experiment):
    if log_filepath != None:
        command= f"{command} >> {log_filepath} 2>> {err_filepath}"
    
    printBeforeRunCommand(command, run_case, log_filepath)
    if is_experiment_already_runned:
        print_output_in_specific_file("Experiment already runned", log_filepath)
        if run_same_experiment == 'No':
            print_output_in_specific_file("Execution of same experiment stopped", log_filepath)
            printAfterRunCommand(command, log_filepath, err_filepath)
            return
    runCommandWithTimeMeasurement(command, log_filepath)
    printAfterRunCommand(command, log_filepath, err_filepath)
    
def runFinancial(run_args, dataset_name = 'greek_scinet_dataset_load_load', window_size = 168, horizon = 24, future_unknown_days = 0, concat_len = 165, 
                 lr = 5e-3, epochs = 200, hidden_size = 8, dropout = 0.0, groups = 1, levels = 3, stacks = 2, decompose = "No"):
    
    financial_file = os.path.join(getRepoPath(),"SCINet","run_financial.py")
    
    command = f"{run_args.python_name} {financial_file}  --dataset_name {dataset_name} --window_size {window_size} --horizon {horizon} " \
                f"--future_unknown_days {future_unknown_days}  --concat_len {concat_len} --lr {lr} --epochs {epochs} " \
                f"--hidden-size {hidden_size} --dropout {dropout} --groups {groups} --levels {levels} --stacks {stacks} --decompose {decompose}"
                
    runExperiment(command, run_args.run_case, None, None, False, None)


def runETTh(run_args, data = 'greek_energy', features = 'M', future_unknown_days = 0, shift_data_y = "Yes", anio = "no_anio", anio_input = "Yes",
            seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 150, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No"):
    
    ETTh_file = os.path.join(getRepoPath(),"SCINet","run_ETTh.py")
    
    command = f"{run_args.python_name} {ETTh_file}  --data {data} --features {features} --future_unknown_days {future_unknown_days} " \
                f"--shift_data_y {shift_data_y}  --anio {anio} --anio_input {anio_input} " \
                f"--seq_len {seq_len} --label_len {label_len} --pred_len {pred_len} --train_epochs {train_epochs} " \
                f"--batch_size {batch_size} --lr {lr}  --hidden-size {hidden_size} " \
                f"--dropout {dropout} --groups {groups} --levels {levels} --stacks {stacks} --decompose {decompose}"
    
    experiment_name = getExperimentName(seq_len, label_len, pred_len, lr, levels, dropout, stacks, features, shift_data_y, decompose, data, train_epochs, 
                                        future_unknown_days, anio, anio_input)
    
    log_filepath = getLogFilepath(run_args.log, 'log', experiment_name)
    err_filepath = getLogFilepath(run_args.log, 'err', experiment_name)
    
    is_experiment_already_runned = isExperimentAlreadyRunned(experiment_name)
        
    runExperiment(command, run_args.run_case, log_filepath, err_filepath, is_experiment_already_runned, run_args.run_same_experiment)

def runETThInvestigation(run_args, data = 'greek_energy', features = 'M', future_unknown_days = 0, shift_data_y = "Yes", seq_len = 96, label_len = 48, pred_len = 48, 
                         train_epochs = 150, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No"):
    runETTh(run_args, data = data, features = features, future_unknown_days = future_unknown_days, shift_data_y = shift_data_y, anio = "no_anio", anio_input = "No", seq_len = seq_len, label_len = label_len, 
            pred_len = pred_len, train_epochs = train_epochs, batch_size = batch_size, lr = lr, hidden_size = hidden_size, dropout = dropout, groups = groups, levels = levels, stacks = stacks, decompose = decompose)
    runETTh(run_args, data = data, features = features, future_unknown_days = future_unknown_days, shift_data_y = shift_data_y, anio = "anio1", anio_input = "Yes", seq_len = seq_len, label_len = label_len, 
            pred_len = pred_len, train_epochs = train_epochs, batch_size = batch_size, lr = lr, hidden_size = hidden_size, dropout = dropout, groups = groups, levels = levels, stacks = stacks, decompose = decompose)
    runETTh(run_args, data = data, features = features, future_unknown_days = future_unknown_days, shift_data_y = shift_data_y, anio = "anio1", anio_input = "No", seq_len = seq_len, label_len = label_len, 
            pred_len = pred_len, train_epochs = train_epochs, batch_size = batch_size, lr = lr, hidden_size = hidden_size, dropout = dropout, groups = groups, levels = levels, stacks = stacks, decompose = decompose)
    runETTh(run_args, data = data, features = features, future_unknown_days = future_unknown_days, shift_data_y = shift_data_y, anio = "anio2", anio_input = "Yes", seq_len = seq_len, label_len = label_len, 
            pred_len = pred_len, train_epochs = train_epochs, batch_size = batch_size, lr = lr, hidden_size = hidden_size, dropout = dropout, groups = groups, levels = levels, stacks = stacks, decompose = decompose)
    runETTh(run_args, data = data, features = features, future_unknown_days = future_unknown_days, shift_data_y = shift_data_y, anio = "anio2", anio_input = "No", seq_len = seq_len, label_len = label_len, 
            pred_len = pred_len, train_epochs = train_epochs, batch_size = batch_size, lr = lr, hidden_size = hidden_size, dropout = dropout, groups = groups, levels = levels, stacks = stacks, decompose = decompose)
    runETTh(run_args, data = data, features = features, future_unknown_days = future_unknown_days, shift_data_y = shift_data_y, anio = "anio3", anio_input = "Yes", seq_len = seq_len, label_len = label_len, 
            pred_len = pred_len, train_epochs = train_epochs, batch_size = batch_size, lr = lr, hidden_size = hidden_size, dropout = dropout, groups = groups, levels = levels, stacks = stacks, decompose = decompose)
    runETTh(run_args, data = data, features = features, future_unknown_days = future_unknown_days, shift_data_y = shift_data_y, anio = "anio3", anio_input = "No", seq_len = seq_len, label_len = label_len, 
            pred_len = pred_len, train_epochs = train_epochs, batch_size = batch_size, lr = lr, hidden_size = hidden_size, dropout = dropout, groups = groups, levels = levels, stacks = stacks, decompose = decompose)
    runETTh(run_args, data = data, features = features, future_unknown_days = future_unknown_days, shift_data_y = shift_data_y, anio = "anio4", anio_input = "Yes", seq_len = seq_len, label_len = label_len, 
            pred_len = pred_len, train_epochs = train_epochs, batch_size = batch_size, lr = lr, hidden_size = hidden_size, dropout = dropout, groups = groups, levels = levels, stacks = stacks, decompose = decompose)
    runETTh(run_args, data = data, features = features, future_unknown_days = future_unknown_days, shift_data_y = shift_data_y, anio = "anio4", anio_input = "No", seq_len = seq_len, label_len = label_len, 
            pred_len = pred_len, train_epochs = train_epochs, batch_size = batch_size, lr = lr, hidden_size = hidden_size, dropout = dropout, groups = groups, levels = levels, stacks = stacks, decompose = decompose)
    runETTh(run_args, data = data, features = features, future_unknown_days = future_unknown_days, shift_data_y = shift_data_y, anio = "anio5", anio_input = "Yes", seq_len = seq_len, label_len = label_len, 
            pred_len = pred_len, train_epochs = train_epochs, batch_size = batch_size, lr = lr, hidden_size = hidden_size, dropout = dropout, groups = groups, levels = levels, stacks = stacks, decompose = decompose)
    runETTh(run_args, data = data, features = features, future_unknown_days = future_unknown_days, shift_data_y = shift_data_y, anio = "anio5", anio_input = "No", seq_len = seq_len, label_len = label_len, 
            pred_len = pred_len, train_epochs = train_epochs, batch_size = batch_size, lr = lr, hidden_size = hidden_size, dropout = dropout, groups = groups, levels = levels, stacks = stacks, decompose = decompose)

def runETThInvestigationForBestAnio(run_args, data = 'greek_energy', features = 'M', future_unknown_days = 0, shift_data_y = "Yes", seq_len = 96, label_len = 48, pred_len = 48, 
                                    train_epochs = 150, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No"):
    runETTh(run_args, data = data, features = features, future_unknown_days = future_unknown_days, shift_data_y = shift_data_y, anio = "no_anio", anio_input = "No", seq_len = seq_len, label_len = label_len, 
            pred_len = pred_len, train_epochs = train_epochs, batch_size = batch_size, lr = lr, hidden_size = hidden_size, dropout = dropout, groups = groups, levels = levels, stacks = stacks, decompose = decompose)
    runETTh(run_args, data = data, features = features, future_unknown_days = future_unknown_days, shift_data_y = shift_data_y, anio = "anio1", anio_input = "Yes", seq_len = seq_len, label_len = label_len, 
            pred_len = pred_len, train_epochs = train_epochs, batch_size = batch_size, lr = lr, hidden_size = hidden_size, dropout = dropout, groups = groups, levels = levels, stacks = stacks, decompose = decompose)
    runETTh(run_args, data = data, features = features, future_unknown_days = future_unknown_days, shift_data_y = shift_data_y, anio = "anio1", anio_input = "No", seq_len = seq_len, label_len = label_len, 
            pred_len = pred_len, train_epochs = train_epochs, batch_size = batch_size, lr = lr, hidden_size = hidden_size, dropout = dropout, groups = groups, levels = levels, stacks = stacks, decompose = decompose)
    runETTh(run_args, data = data, features = features, future_unknown_days = future_unknown_days, shift_data_y = shift_data_y, anio = "anio4", anio_input = "Yes", seq_len = seq_len, label_len = label_len, 
            pred_len = pred_len, train_epochs = train_epochs, batch_size = batch_size, lr = lr, hidden_size = hidden_size, dropout = dropout, groups = groups, levels = levels, stacks = stacks, decompose = decompose)
    runETTh(run_args, data = data, features = features, future_unknown_days = future_unknown_days, shift_data_y = shift_data_y, anio = "anio4", anio_input = "No", seq_len = seq_len, label_len = label_len, 
            pred_len = pred_len, train_epochs = train_epochs, batch_size = batch_size, lr = lr, hidden_size = hidden_size, dropout = dropout, groups = groups, levels = levels, stacks = stacks, decompose = decompose)   

def runTestCase(run_args):
    
    run_case = run_args.run_case
    
    if run_case == -1:
        # Single test experiment - greek - no_anio
        runETTh(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes", anio = "no_anio", anio_input = "No",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 2, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -2:
        # Single test experiment - greek - anio1 - no_anio_input
        runETTh(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes", anio = "anio1", anio_input = "No",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 2, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -3:
        # Single test experiment - greek - anio1 - anio_input
        runETTh(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes", anio = "anio1", anio_input = "Yes",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 2, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -4:
        # Single test experiment - greek - anio2 - no_anio_input
        runETTh(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes", anio = "anio2", anio_input = "No",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 2, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -5:
        # Single test experiment - greek - anio2 - anio_input
        runETTh(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes", anio = "anio2", anio_input = "Yes",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 2, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -6:
        # Single test experiment - greek - anio3 - no_anio_input
        runETTh(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes", anio = "anio3", anio_input = "No",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 2, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -7:
        # Single test experiment - greek - anio3 - anio_input
        runETTh(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes", anio = "anio3", anio_input = "Yes",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 2, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -8:
        # Single test experiment - greek - anio4 - no_anio_input
        runETTh(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes", anio = "anio4", anio_input = "No",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 2, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -9:
        # Single test experiment - greek - anio4 - anio_input
        runETTh(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes", anio = "anio4", anio_input = "Yes",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 2, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -10:
        # Single test experiment - greek - anio5 - no_anio_input
        runETTh(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes", anio = "anio5", anio_input = "No",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 2, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -11:
        # Single test experiment - greek - anio5 - anio_input
        runETTh(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes", anio = "anio5", anio_input = "Yes",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 2, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -12:
        # Single test experiment - ETTh1 - multivariate - no_anio
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 48, label_len = 24, pred_len = 24, train_epochs = 2, batch_size = 8, lr = 3e-3, hidden_size = 4, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -13:
        # Single test experiment - ETTh1 - multivariate - anio1 - no_anio_input
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "anio1", anio_input = "No",
                seq_len = 48, label_len = 24, pred_len = 24, train_epochs = 2, batch_size = 8, lr = 3e-3, hidden_size = 4, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -14:
        # Single test experiment - ETTh1 - multivariate - anio1 - anio_input
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "anio1", anio_input = "Yes",
                seq_len = 48, label_len = 24, pred_len = 24, train_epochs = 2, batch_size = 8, lr = 3e-3, hidden_size = 4, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -15:
        # Single test experiment - ETTh1 - multivariate - anio2 - no_anio_input
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "anio2", anio_input = "No",
                seq_len = 48, label_len = 24, pred_len = 24, train_epochs = 2, batch_size = 8, lr = 3e-3, hidden_size = 4, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -16:
        # Single test experiment - ETTh1 - multivariate - anio2 - anio_input
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "anio2", anio_input = "Yes",
                seq_len = 48, label_len = 24, pred_len = 24, train_epochs = 2, batch_size = 8, lr = 3e-3, hidden_size = 4, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -17:
        # Single test experiment - ETTh1 - multivariate - anio3 - no_anio_input
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "anio3", anio_input = "No",
                seq_len = 48, label_len = 24, pred_len = 24, train_epochs = 2, batch_size = 8, lr = 3e-3, hidden_size = 4, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -18:
        # Single test experiment - ETTh1 - multivariate - anio3 - anio_input
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "anio3", anio_input = "Yes",
                seq_len = 48, label_len = 24, pred_len = 24, train_epochs = 2, batch_size = 8, lr = 3e-3, hidden_size = 4, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -19:
        # Single test experiment - ETTh1 - multivariate - anio4 - no_anio_input
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "anio4", anio_input = "No",
                seq_len = 48, label_len = 24, pred_len = 24, train_epochs = 2, batch_size = 8, lr = 3e-3, hidden_size = 4, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -20:
        # Single test experiment - ETTh1 - multivariate - anio4 - anio_input
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "anio4", anio_input = "Yes",
                seq_len = 48, label_len = 24, pred_len = 24, train_epochs = 2, batch_size = 8, lr = 3e-3, hidden_size = 4, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -21:
        # Single test experiment - ETTh1 - multivariate - anio5 - no_anio_input
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "anio5", anio_input = "No",
                seq_len = 48, label_len = 24, pred_len = 24, train_epochs = 2, batch_size = 8, lr = 3e-3, hidden_size = 4, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -22:
        # Single test experiment - ETTh1 - multivariate - anio5 - anio_input
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "anio5", anio_input = "Yes",
                seq_len = 48, label_len = 24, pred_len = 24, train_epochs = 2, batch_size = 8, lr = 3e-3, hidden_size = 4, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")    
    elif run_case == -23:
        # Single test experiment - ETTh2 - multivariate - no_anio
        runETTh(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 2, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -24:
        # Single test experiment - ETTh2 - multivariate - anio1 - no_anio_input
        runETTh(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "anio1", anio_input = "No",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 2, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -25:
        # Single test experiment - ETTh2 - multivariate - anio1 - anio_input
        runETTh(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "anio1", anio_input = "Yes", seq_len = 96, 
                label_len = 48, pred_len = 48, train_epochs = 2, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -26:
        # Single test experiment - ETTh2 - multivariate - anio2 - no_anio_input
        runETTh(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "anio2", anio_input = "No",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 2, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -27:
        # Single test experiment - ETTh2 - multivariate - anio2 - anio_input
        runETTh(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "anio2", anio_input = "Yes",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 2, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -28:
        # Single test experiment - ETTh2 - multivariate - anio3 - no_anio_input
        runETTh(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "anio3", anio_input = "No",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 2, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -29:
        # Single test experiment - ETTh2 - multivariate - anio3 - anio_input
        runETTh(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "anio3", anio_input = "Yes",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 2, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -30:
        # Single test experiment - ETTh2 - multivariate - anio4 - no_anio_input
        runETTh(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "anio4", anio_input = "No",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 2, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -31:
        # Single test experiment - ETTh2 - multivariate - anio4 - anio_input
        runETTh(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "anio4", anio_input = "Yes",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 2, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -32:
        # Single test experiment - ETTh2 - multivariate - anio5 - no_anio_input
        runETTh(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "anio5", anio_input = "No",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 2, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -33:
        # Single test experiment - ETTh2 - multivariate - anio5 - anio_input
        runETTh(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "anio5", anio_input = "Yes",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 2, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -40:
        runETTh(run_args)    
    elif run_case == -41:
        # Group test - greek dataset
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 2, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == -42:
        # Group test - ETTh1 - multivariate
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 48, label_len = 24, pred_len = 24, train_epochs = 2, batch_size = 8, lr = 3e-3, hidden_size = 4, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 2, batch_size = 16, lr = 0.009, hidden_size = 4, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "No")
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 336, label_len = 168, pred_len = 168, train_epochs = 2, batch_size = 32, lr = 5e-4, hidden_size = 4, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 336, label_len = 336, pred_len = 336, train_epochs = 2, batch_size = 512, lr = 1e-4, hidden_size = 1, dropout = 0.5, groups =1, levels = 4, stacks = 1, decompose = "No")
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 736, label_len = 720, pred_len = 720, train_epochs = 2, batch_size = 256, lr = 5e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 5, stacks = 1, decompose = "No")
    elif run_case == -43:
        # Group test - ETTh1 - univariate
        runETTh(run_args, data = 'ETTh1', features='S', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 64, label_len = 24, pred_len = 24, train_epochs = 2, batch_size = 64, lr = 0.007, hidden_size = 8, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "No")
        runETTh(run_args, data = 'ETTh1', features='S', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 720, label_len = 48, pred_len = 48, train_epochs = 2, batch_size = 8, lr = 0.0001, hidden_size = 4, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "No")
        runETTh(run_args, data = 'ETTh1', features='S', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 720, label_len = 168, pred_len = 168, train_epochs = 2, batch_size = 8, lr = 5e-5, hidden_size = 4, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "No")
        runETTh(run_args, data = 'ETTh1', features='S', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 720, label_len = 336, pred_len = 336, train_epochs = 2, batch_size = 128, lr = 1e-3, hidden_size = 1, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "No")
        runETTh(run_args, data = 'ETTh1', features='S', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 736, label_len = 720, pred_len = 720, train_epochs = 2, batch_size = 32, lr = 1e-4, hidden_size = 4, dropout = 0.5, groups = 1, levels = 5, stacks = 1, decompose = "No")
    elif run_case == -44:
        runETTh(run_args)
    elif run_case == 1:
        #ETTh1 - multivariate - no_anio
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 48, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 8, lr = 3e-3, hidden_size = 4, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 16, lr = 0.009, hidden_size = 4, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "No")
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 336, label_len = 168, pred_len = 168, train_epochs = 300, batch_size = 32, lr = 5e-4, hidden_size = 4, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 336, label_len = 336, pred_len = 336, train_epochs = 300, batch_size = 512, lr = 1e-4, hidden_size = 1, dropout = 0.5, groups =1, levels = 4, stacks = 1, decompose = "No")
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 736, label_len = 720, pred_len = 720, train_epochs = 300, batch_size = 256, lr = 5e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 5, stacks = 1, decompose = "No")
    elif run_case == 2:
        #ETTh1 - univariate - no_anio
        runETTh(run_args, data = 'ETTh1', features='S', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 64, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 64, lr = 0.007, hidden_size = 8, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "No")
        runETTh(run_args, data = 'ETTh1', features='S', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 720, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 8, lr = 0.0001, hidden_size = 4, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "No")
        runETTh(run_args, data = 'ETTh1', features='S', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 720, label_len = 168, pred_len = 168, train_epochs = 300, batch_size = 8, lr = 5e-5, hidden_size = 4, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "No")
        runETTh(run_args, data = 'ETTh1', features='S', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 720, label_len = 336, pred_len = 336, train_epochs = 300, batch_size = 128, lr = 1e-3, hidden_size = 1, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "No")
        runETTh(run_args, data = 'ETTh1', features='S', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 736, label_len = 720, pred_len = 720, train_epochs = 300, batch_size = 32, lr = 1e-4, hidden_size = 4, dropout = 0.5, groups = 1, levels = 5, stacks = 1, decompose = "No")
    elif run_case == 3:
        #ETTh2 - multivariate - no_anio
        runETTh(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 48, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 16, lr = 0.007, hidden_size = 8, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "No")
        runETTh(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 4, lr = 0.007, hidden_size = 4, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "No")
        runETTh(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 336, label_len = 168, pred_len = 168, train_epochs = 300, batch_size = 16, lr = 5e-5, hidden_size = 0.5, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "No")
        runETTh(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 336, label_len = 336, pred_len = 336, train_epochs = 300, batch_size = 128, lr = 5e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "No")
        runETTh(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 736, label_len = 720, pred_len = 720, train_epochs = 300, batch_size = 128, lr = 1e-5, hidden_size = 4, dropout = 0.5, groups = 1, levels = 5, stacks = 1, decompose = "No")
    elif run_case == 4:
        #ETTh2 - univariate - no_anio
        runETTh(run_args, data = 'ETTh2', features='S', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 48, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 16, lr = 0.001, hidden_size = 4, dropout = 0.0, groups = 1, levels = 3, stacks = 1, decompose = "No")
        runETTh(run_args, data = 'ETTh2', features='S', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 32, lr = 0.001, hidden_size = 4, dropout = 0.5, groups = 1, levels = 4, stacks = 2, decompose = "No")
        runETTh(run_args, data = 'ETTh2', features='S', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 336, label_len = 168, pred_len = 168, train_epochs = 300, batch_size = 8, lr = 1e-4, hidden_size = 4, dropout = 0.0, groups = 1, levels = 3, stacks = 1, decompose = "No")
        runETTh(run_args, data = 'ETTh2', features='S', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 336, label_len = 336, pred_len = 336, train_epochs = 300, batch_size = 512, lr = 5e-4, hidden_size = 8, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
        runETTh(run_args, data = 'ETTh2', features='S', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 720, label_len = 720, pred_len = 720, train_epochs = 300, batch_size = 128, lr = 1e-5, hidden_size = 8, dropout = 0.6, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 11:
        #ETTh1 - multivariate - 48 - 24 - 24 - simple case anio investigation
        runETThInvestigation(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 48, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 8, lr = 3e-3, hidden_size = 4, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 12:
        #ETTh1 - multivariate - 96 - 48 - 48 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 16, lr = 0.009, hidden_size = 4, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 13:
        #ETTh1 - multivariate - 336 - 168 - 168 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 336, label_len = 168, pred_len = 168, train_epochs = 300, batch_size = 32, lr = 5e-4, hidden_size = 4, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 14:
        #ETTh1 - multivariate - 336 - 336 - 336 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 336, label_len = 336, pred_len = 336, train_epochs = 300, batch_size = 256, lr = 1e-4, hidden_size = 1, dropout = 0.5, groups =1, levels = 4, stacks = 1, decompose = "No")
    elif run_case == 15:
        #ETTh1 - multivariate - 720 - 720 - 720 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 736, label_len = 720, pred_len = 720, train_epochs = 300, batch_size = 256, lr = 5e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 5, stacks = 1, decompose = "No")
    elif run_case == 16:
        #ETTh1 - univariate - 64 - 24 - 24 - simple case anio investigation
        runETThInvestigation(run_args, data = 'ETTh1', features='S', future_unknown_days = 0, shift_data_y = "No", 
                             seq_len = 64, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 64, lr = 0.007, hidden_size = 8, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 17:
        #ETTh1 - univariate - 720 - 48 - 48 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'ETTh1', features='S', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 720, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 8, lr = 0.0001, hidden_size = 4, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "No")
    elif run_case == 18:
        #ETTh1 - univariate - 720 - 168 - 168 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'ETTh1', features='S', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 720, label_len = 168, pred_len = 168, train_epochs = 300, batch_size = 8, lr = 5e-5, hidden_size = 4, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "No")
    elif run_case == 19:
        #ETTh1 - univariate - 720 - 336 - 336 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'ETTh1', features='S', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 720, label_len = 336, pred_len = 336, train_epochs = 300, batch_size = 128, lr = 1e-3, hidden_size = 1, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "No")
    elif run_case == 20:
        #ETTh1 - univariate - 736 - 720 - 720 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'ETTh1', features='S', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 736, label_len = 720, pred_len = 720, train_epochs = 300, batch_size = 32, lr = 1e-4, hidden_size = 4, dropout = 0.5, groups = 1, levels = 5, stacks = 1, decompose = "No")
    elif run_case == 21:
        #ETTh2 - multivariate - 48 - 24 - 24 - simple case anio investigation
        runETThInvestigation(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 48, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 16, lr = 0.007, hidden_size = 8, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 22:
        #ETTh2 - multivariate - 96 - 48 - 48 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 4, lr = 0.007, hidden_size = 4, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "No")
    elif run_case == 23:
        #ETTh2 - multivariate - 336 - 168 - 168 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 336, label_len = 168, pred_len = 168, train_epochs = 300, batch_size = 16, lr = 5e-5, hidden_size = 0.5, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "No")
    elif run_case == 24:
        #ETTh2 - multivariate - 336 - 336 - 336 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 336, label_len = 336, pred_len = 336, train_epochs = 300, batch_size = 128, lr = 5e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "No")
    elif run_case == 25:
        #ETTh2 - multivariate - 736 - 720 - 720 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 736, label_len = 720, pred_len = 720, train_epochs = 300, batch_size = 128, lr = 1e-5, hidden_size = 4, dropout = 0.5, groups = 1, levels = 5, stacks = 1, decompose = "No")
    elif run_case == 26:
        #ETTh2 - univariate - 48 - 24 - 24 - simple case anio investigation
        runETThInvestigation(run_args, data = 'ETTh2', features='S', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 48, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 16, lr = 0.001, hidden_size = 4, dropout = 0.0, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 27:
        #ETTh2 - univariate - 96 - 48 - 48 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'ETTh2', features='S', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 32, lr = 0.001, hidden_size = 4, dropout = 0.5, groups = 1, levels = 4, stacks = 2, decompose = "No")
    elif run_case == 28:
        #ETTh2 - univariate - 336 - 168 - 168 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'ETTh2', features='S', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 336, label_len = 168, pred_len = 168, train_epochs = 300, batch_size = 8, lr = 1e-4, hidden_size = 4, dropout = 0.0, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 29:
        #ETTh2 - univariate - 336 - 336 - 336 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'ETTh2', features='S', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 336, label_len = 336, pred_len = 336, train_epochs = 300, batch_size = 512, lr = 5e-4, hidden_size = 8, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 30:
        #ETTh2 - univariate - 720 - 720 - 720 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'ETTh2', features='S', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 720, label_len = 720, pred_len = 720, train_epochs = 300, batch_size = 128, lr = 1e-5, hidden_size = 8, dropout = 0.6, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 41:
        #Greek_energy - 96 - 24 - 24 - simple case anio investigation
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 96, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 42:
        #Greek_energy - 192 - 24 - 24 - simple case anio investigation
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 192, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 43:
        #Greek_energy - 336 - 24 - 24 - simple case anio investigation
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 336, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 44:
        #Greek_energy - 96 - 48 - 48 - simple case anio investigation
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 45:
        #Greek_energy - 192 - 48 - 48 - simple case anio investigation
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 192, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 46:
        #Greek_energy - 336 - 48 - 48 - simple case anio investigation
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 336, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 47:
        #Greek_energy - multivariate - 336 - 168 - 168 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 336, label_len = 168, pred_len = 168, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "No")
    elif run_case == 48:
        #Greek_energy - multivariate - 336 - 336 - 336 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 336, label_len = 336, pred_len = 336, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "No")
    elif run_case == 49:
        #Greek_energy - multivariate - 736 - 720 - 720 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 736, label_len = 720, pred_len = 720, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 5, stacks = 1, decompose = "No")
    elif run_case == 50:
        #Greek_energy - 96 - 24 - 24 - 0.25 - simple case anio investigation
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 96, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 51:
        #Greek_energy - 192 - 24 - 24 - 0.25 - simple case anio investigation
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 192, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 52:
        #Greek_energy - 336 - 24 - 24 - 0.25 - simple case anio investigation
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 336, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 53:
        #Greek_energy - 96 - 48 - 48 - 0.25 - simple case anio investigation
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes", 
                             seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 54:
        #Greek_energy - 192 - 48 - 48 - 0.25 - simple case anio investigation
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 192, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 55:
        #Greek_energy - 336 - 48 - 48 - 0.25 - simple case anio investigation
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 336, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 56:
        #Greek_energy - multivariate - 336 - 168 - 168 - 0.25 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 336, label_len = 168, pred_len = 168, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 4, stacks = 1, decompose = "No")
    elif run_case == 57:
        #Greek_energy - multivariate - 336 - 336 - 336 - 0.25 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 336, label_len = 336, pred_len = 336, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 4, stacks = 1, decompose = "No")
    elif run_case == 58:
        #Greek_energy - multivariate - 736 - 720 - 720 - 0.25 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 736, label_len = 720, pred_len = 720, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 5, stacks = 1, decompose = "No")
    elif run_case == 101:
        #ETTh1 - multivariate - no_anio - decompose
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 48, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 8, lr = 3e-3, hidden_size = 4, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 16, lr = 0.009, hidden_size = 4, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 336, label_len = 168, pred_len = 168, train_epochs = 300, batch_size = 32, lr = 5e-4, hidden_size = 4, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 336, label_len = 336, pred_len = 336, train_epochs = 300, batch_size = 512, lr = 1e-4, hidden_size = 1, dropout = 0.5, groups =1, levels = 4, stacks = 1, decompose = "Yes")
        runETTh(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 736, label_len = 720, pred_len = 720, train_epochs = 300, batch_size = 256, lr = 5e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 5, stacks = 1, decompose = "Yes")
    elif run_case == 102:
        #ETTh1 - univariate - no_anio - decompose
        runETTh(run_args, data = 'ETTh1', features='S', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 64, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 64, lr = 0.007, hidden_size = 8, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
        runETTh(run_args, data = 'ETTh1', features='S', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 720, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 8, lr = 0.0001, hidden_size = 4, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "Yes")
        runETTh(run_args, data = 'ETTh1', features='S', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 720, label_len = 168, pred_len = 168, train_epochs = 300, batch_size = 8, lr = 5e-5, hidden_size = 4, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "Yes")
        runETTh(run_args, data = 'ETTh1', features='S', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 720, label_len = 336, pred_len = 336, train_epochs = 300, batch_size = 128, lr = 1e-3, hidden_size = 1, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "Yes")
        runETTh(run_args, data = 'ETTh1', features='S', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 736, label_len = 720, pred_len = 720, train_epochs = 300, batch_size = 32, lr = 1e-4, hidden_size = 4, dropout = 0.5, groups = 1, levels = 5, stacks = 1, decompose = "Yes")
    elif run_case == 103:
        #ETTh2 - multivariate - no_anio - decompose
        runETTh(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 48, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 16, lr = 0.007, hidden_size = 8, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
        runETTh(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 4, lr = 0.007, hidden_size = 4, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "Yes")
        runETTh(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 336, label_len = 168, pred_len = 168, train_epochs = 300, batch_size = 16, lr = 5e-5, hidden_size = 0.5, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "Yes")
        runETTh(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 336, label_len = 336, pred_len = 336, train_epochs = 300, batch_size = 128, lr = 5e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "Yes")
        runETTh(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 736, label_len = 720, pred_len = 720, train_epochs = 300, batch_size = 128, lr = 1e-5, hidden_size = 4, dropout = 0.5, groups = 1, levels = 5, stacks = 1, decompose = "Yes")
    elif run_case == 104:
        #ETTh2 - univariate - no_anio - decompose
        runETTh(run_args, data = 'ETTh2', features='S', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 48, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 16, lr = 0.001, hidden_size = 4, dropout = 0.0, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
        runETTh(run_args, data = 'ETTh2', features='S', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 32, lr = 0.001, hidden_size = 4, dropout = 0.5, groups = 1, levels = 4, stacks = 2, decompose = "Yes")
        runETTh(run_args, data = 'ETTh2', features='S', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 336, label_len = 168, pred_len = 168, train_epochs = 300, batch_size = 8, lr = 1e-4, hidden_size = 4, dropout = 0.0, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
        runETTh(run_args, data = 'ETTh2', features='S', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 336, label_len = 336, pred_len = 336, train_epochs = 300, batch_size = 512, lr = 5e-4, hidden_size = 8, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
        runETTh(run_args, data = 'ETTh2', features='S', future_unknown_days = 0, shift_data_y = "No", anio = "no_anio", anio_input = "No",
                seq_len = 720, label_len = 720, pred_len = 720, train_epochs = 300, batch_size = 128, lr = 1e-5, hidden_size = 8, dropout = 0.6, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
    elif run_case == 111:
        #ETTh1 - multivariate - 48 - 24 - 24 - simple case anio investigation - decompose
        runETThInvestigation(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 48, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 8, lr = 3e-3, hidden_size = 4, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
    elif run_case == 112:
        #ETTh1 - multivariate - 96 - 48 - 48 - simple case anio investigation - decompose   
        runETThInvestigation(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 16, lr = 0.009, hidden_size = 4, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
    elif run_case == 113:
        #ETTh1 - multivariate - 336 - 168 - 168 - simple case anio investigation - decompose    
        runETThInvestigation(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 336, label_len = 168, pred_len = 168, train_epochs = 300, batch_size = 32, lr = 5e-4, hidden_size = 4, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
    elif run_case == 114:
        #ETTh1 - multivariate - 336 - 336 - 336 - simple case anio investigation - decompose    
        runETThInvestigation(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 336, label_len = 336, pred_len = 336, train_epochs = 300, batch_size = 256, lr = 1e-4, hidden_size = 1, dropout = 0.5, groups =1, levels = 4, stacks = 1, decompose = "Yes")
    elif run_case == 115:
        #ETTh1 - multivariate - 720 - 720 - 720 - simple case anio investigation - decompose    
        runETThInvestigation(run_args, data = 'ETTh1', features='M', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 736, label_len = 720, pred_len = 720, train_epochs = 300, batch_size = 256, lr = 5e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 5, stacks = 1, decompose = "Yes")
    elif run_case == 116:
        #ETTh1 - univariate - 64 - 24 - 24 - simple case anio investigation - decompose
        runETThInvestigation(run_args, data = 'ETTh1', features='S', future_unknown_days = 0, shift_data_y = "No", 
                             seq_len = 64, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 64, lr = 0.007, hidden_size = 8, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
    elif run_case == 117:
        #ETTh1 - univariate - 720 - 48 - 48 - simple case anio investigation - decompose    
        runETThInvestigation(run_args, data = 'ETTh1', features='S', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 720, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 8, lr = 0.0001, hidden_size = 4, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "Yes")
    elif run_case == 118:
        #ETTh1 - univariate - 720 - 168 - 168 - simple case anio investigation - decompose    
        runETThInvestigation(run_args, data = 'ETTh1', features='S', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 720, label_len = 168, pred_len = 168, train_epochs = 300, batch_size = 8, lr = 5e-5, hidden_size = 4, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "Yes")
    elif run_case == 119:
        #ETTh1 - univariate - 720 - 336 - 336 - simple case anio investigation - decompose    
        runETThInvestigation(run_args, data = 'ETTh1', features='S', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 720, label_len = 336, pred_len = 336, train_epochs = 300, batch_size = 128, lr = 1e-3, hidden_size = 1, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "Yes")
    elif run_case == 120:
        #ETTh1 - univariate - 736 - 720 - 720 - simple case anio investigation - decompose    
        runETThInvestigation(run_args, data = 'ETTh1', features='S', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 736, label_len = 720, pred_len = 720, train_epochs = 300, batch_size = 32, lr = 1e-4, hidden_size = 4, dropout = 0.5, groups = 1, levels = 5, stacks = 1, decompose = "Yes")
    elif run_case == 121:
        #ETTh2 - multivariate - 48 - 24 - 24 - simple case anio investigation - decompose
        runETThInvestigation(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 48, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 16, lr = 0.007, hidden_size = 8, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
    elif run_case == 122:
        #ETTh2 - multivariate - 96 - 48 - 48 - simple case anio investigation - decompose    
        runETThInvestigation(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 4, lr = 0.007, hidden_size = 4, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "Yes")
    elif run_case == 123:
        #ETTh2 - multivariate - 336 - 168 - 168 - simple case anio investigation - decompose    
        runETThInvestigation(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 336, label_len = 168, pred_len = 168, train_epochs = 300, batch_size = 16, lr = 5e-5, hidden_size = 0.5, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "Yes")
    elif run_case == 124:
        #ETTh2 - multivariate - 336 - 336 - 336 - simple case anio investigation - decompose    
        runETThInvestigation(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 336, label_len = 336, pred_len = 336, train_epochs = 300, batch_size = 128, lr = 5e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "Yes")
    elif run_case == 125:
        #ETTh2 - multivariate - 736 - 720 - 720 - simple case anio investigation - decompose    
        runETThInvestigation(run_args, data = 'ETTh2', features='M', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 736, label_len = 720, pred_len = 720, train_epochs = 300, batch_size = 64, lr = 1e-5, hidden_size = 4, dropout = 0.5, groups = 1, levels = 5, stacks = 1, decompose = "Yes")
    elif run_case == 126:
        #ETTh2 - univariate - 48 - 24 - 24 - simple case anio investigation - decompose
        runETThInvestigation(run_args, data = 'ETTh2', features='S', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 48, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 16, lr = 0.001, hidden_size = 4, dropout = 0.0, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
    elif run_case == 127:
        #ETTh2 - univariate - 96 - 48 - 48 - simple case anio investigation - decompose    
        runETThInvestigation(run_args, data = 'ETTh2', features='S', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 32, lr = 0.001, hidden_size = 4, dropout = 0.5, groups = 1, levels = 4, stacks = 2, decompose = "Yes")
    elif run_case == 128:
        #ETTh2 - univariate - 336 - 168 - 168 - simple case anio investigation - decompose    
        runETThInvestigation(run_args, data = 'ETTh2', features='S', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 336, label_len = 168, pred_len = 168, train_epochs = 300, batch_size = 8, lr = 1e-4, hidden_size = 4, dropout = 0.0, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
    elif run_case == 129:
        #ETTh2 - univariate - 336 - 336 - 336 - simple case anio investigation - decompose    
        runETThInvestigation(run_args, data = 'ETTh2', features='S', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 336, label_len = 336, pred_len = 336, train_epochs = 300, batch_size = 512, lr = 5e-4, hidden_size = 8, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
    elif run_case == 130:
        #ETTh2 - univariate - 720 - 720 - 720 - simple case anio investigation - decompose    
        runETThInvestigation(run_args, data = 'ETTh2', features='S', future_unknown_days = 0, shift_data_y = "No",
                             seq_len = 720, label_len = 720, pred_len = 720, train_epochs = 300, batch_size = 128, lr = 1e-5, hidden_size = 8, dropout = 0.6, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
    elif run_case == 141:
        #Greek_energy - 96 - 24 - 24 - simple case anio investigation - decompose
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 96, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
    elif run_case == 142:
        #Greek_energy - 192 - 24 - 24 - simple case anio investigation - decompose
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 192, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
    elif run_case == 143:
        #Greek_energy - 336 - 24 - 24 - simple case anio investigation - decompose
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 336, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
    elif run_case == 144:
        #Greek_energy - 96 - 48 - 48 - simple case anio investigation - decompose
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
    elif run_case == 145:
        #Greek_energy - 192 - 48 - 48 - simple case anio investigation - decompose
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 192, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
    elif run_case == 146:
        #Greek_energy - 336 - 48 - 48 - simple case anio investigation - decompose
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 336, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
    elif run_case == 147:
        #Greek_energy - multivariate - 336 - 168 - 168 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 336, label_len = 168, pred_len = 168, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "Yes")
    elif run_case == 148:
        #Greek_energy - multivariate - 336 - 336 - 336 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 336, label_len = 336, pred_len = 336, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 4, stacks = 1, decompose = "Yes")
    elif run_case == 149:
        #Greek_energy - multivariate - 736 - 720 - 720 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 736, label_len = 720, pred_len = 720, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 5, stacks = 1, decompose = "Yes")
    elif run_case == 150:
        #Greek_energy - 96 - 24 - 24 - 0.25 - simple case anio investigation - decompose
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 96, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
    elif run_case == 151:
        #Greek_energy - 192 - 24 - 24 - 0.25 - simple case anio investigation - decompose
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 192, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
    elif run_case == 152:
        #Greek_energy - 336 - 24 - 24 - 0.25 - simple case anio investigation - decompose
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 336, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
    elif run_case == 153:
        #Greek_energy - 96 - 48 - 48 - 0.25 - simple case anio investigation - decompose
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes", 
                             seq_len = 96, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
    elif run_case == 154:
        #Greek_energy - 192 - 48 - 48 - 0.25 - simple case anio investigation - decompose
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 192, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
    elif run_case == 155:
        #Greek_energy - 336 - 48 - 48 - 0.25 - simple case anio investigation - decompose
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 336, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "Yes")
    elif run_case == 156:
        #Greek_energy - multivariate - 336 - 168 - 168 - 0.25 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 336, label_len = 168, pred_len = 168, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 4, stacks = 1, decompose = "Yes")
    elif run_case == 157:
        #Greek_energy - multivariate - 336 - 336 - 336 - 0.25 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 336, label_len = 336, pred_len = 336, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 4, stacks = 1, decompose = "Yes")
    elif run_case == 158:
        #Greek_energy - multivariate - 736 - 720 - 720 - 0.25 - simple case anio investigation    
        runETThInvestigation(run_args, data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes",
                             seq_len = 736, label_len = 720, pred_len = 720, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 5, stacks = 1, decompose = "Yes")
    elif run_case == 201:
        #Greek_energy - univariate - 336 - 24 - 24 - 0.25 - best case anio investigation
        runETThInvestigationForBestAnio(run_args, data = 'greek_energy', features='S', future_unknown_days = 0, shift_data_y = "Yes",
                                        seq_len = 336, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 202:
        #Greek_energy - univariate - 336 - 48 - 48 - 0.25 - best case anio investigation
        runETThInvestigationForBestAnio(run_args, data = 'greek_energy', features='S', future_unknown_days = 0, shift_data_y = "Yes",
                                        seq_len = 336, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 203:
        #Greek_energy - univariate - 336 - 336 - 336 - 0.25 - best case anio investigation    
        runETThInvestigationForBestAnio(run_args, data = 'greek_energy', features='S', future_unknown_days = 0, shift_data_y = "Yes",
                                        seq_len = 336, label_len = 336, pred_len = 336, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 4, stacks = 1, decompose = "No")
    elif run_case == 204:
        #Greek_energy - univariate - 736 - 720 - 720 - 0.25 - best case anio investigation    
        runETThInvestigationForBestAnio(run_args, data = 'greek_energy', features='S', future_unknown_days = 0, shift_data_y = "Yes",
                                        seq_len = 736, label_len = 720, pred_len = 720, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 5, stacks = 1, decompose = "No")
    elif run_case == 211:
        #Greek_energy - 1 unknown day - univariate - 336 - 24 - 24 - 0.25 - best case anio investigation
        runETThInvestigationForBestAnio(run_args, data = 'greek_energy', features='M', future_unknown_days = 1, shift_data_y = "Yes",
                                        seq_len = 336, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 212:
        #Greek_energy - 1 unknown day - univariate - 336 - 48 - 48 - 0.25 - best case anio investigation
        runETThInvestigationForBestAnio(run_args, data = 'greek_energy', features='M', future_unknown_days = 1, shift_data_y = "Yes",
                                        seq_len = 336, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 213:
        #Greek_energy - 1 unknown day - univariate - 336 - 336 - 336 - 0.25 - best case anio investigation    
        runETThInvestigationForBestAnio(run_args, data = 'greek_energy', features='M', future_unknown_days = 1, shift_data_y = "Yes",
                                        seq_len = 336, label_len = 336, pred_len = 336, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 4, stacks = 1, decompose = "No")
    elif run_case == 214:
        #Greek_energy - 1 unknown day - univariate - 736 - 720 - 720 - 0.25 - best case anio investigation    
        runETThInvestigationForBestAnio(run_args, data = 'greek_energy', features='M', future_unknown_days = 1, shift_data_y = "Yes",
                                        seq_len = 736, label_len = 720, pred_len = 720, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 5, stacks = 1, decompose = "No")
    elif run_case == 221:
        #Greek_energy - 4 unknown day - univariate - 336 - 24 - 24 - 0.25 - best case anio investigation
        runETThInvestigationForBestAnio(run_args, data = 'greek_energy', features='M', future_unknown_days = 4, shift_data_y = "Yes",
                                        seq_len = 336, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 222:
        #Greek_energy - 4 unknown day - univariate - 336 - 48 - 48 - 0.25 - best case anio investigation
        runETThInvestigationForBestAnio(run_args, data = 'greek_energy', features='M', future_unknown_days = 4, shift_data_y = "Yes",
                                        seq_len = 336, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 223:
        #Greek_energy - 4 unknown day - univariate - 336 - 336 - 336 - 0.25 - best case anio investigation    
        runETThInvestigationForBestAnio(run_args, data = 'greek_energy', features='M', future_unknown_days = 4, shift_data_y = "Yes",
                                        seq_len = 336, label_len = 336, pred_len = 336, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 4, stacks = 1, decompose = "No")
    elif run_case == 224:
        #Greek_energy - 4 unknown day - univariate - 736 - 720 - 720 - 0.25 - best case anio investigation    
        runETThInvestigationForBestAnio(run_args, data = 'greek_energy', features='M', future_unknown_days = 4, shift_data_y = "Yes",
                                        seq_len = 736, label_len = 720, pred_len = 720, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 5, stacks = 1, decompose = "No")
    elif run_case == 231:
        #Greek_energy - 10 unknown day - univariate - 336 - 24 - 24 - 0.25 - best case anio investigation
        runETThInvestigationForBestAnio(run_args, data = 'greek_energy', features='M', future_unknown_days = 10, shift_data_y = "Yes",
                                        seq_len = 336, label_len = 24, pred_len = 24, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 232:
        #Greek_energy - 10 unknown day - univariate - 336 - 48 - 48 - 0.25 - best case anio investigation
        runETThInvestigationForBestAnio(run_args, data = 'greek_energy', features='M', future_unknown_days = 10, shift_data_y = "Yes",
                                        seq_len = 336, label_len = 48, pred_len = 48, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 3, stacks = 1, decompose = "No")
    elif run_case == 233:
        #Greek_energy - 10 unknown day - univariate - 336 - 336 - 336 - 0.25 - best case anio investigation    
        runETThInvestigationForBestAnio(run_args, data = 'greek_energy', features='M', future_unknown_days = 10, shift_data_y = "Yes",
                                        seq_len = 336, label_len = 336, pred_len = 336, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 4, stacks = 1, decompose = "No")
    elif run_case == 234:
        #Greek_energy - 10 unknown day - univariate - 736 - 720 - 720 - 0.25 - best case anio investigation    
        runETThInvestigationForBestAnio(run_args, data = 'greek_energy', features='M', future_unknown_days = 10, shift_data_y = "Yes",
                                        seq_len = 736, label_len = 720, pred_len = 720, train_epochs = 300, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.25, groups = 1, levels = 5, stacks = 1, decompose = "No")
    elif run_case == 1000:
        print("Between ETTh and financial")
    elif run_case == 1001:
        runFinancial(run_args, dataset_name = 'greek_scinet_dataset_load_load', window_size = 168, horizon = 24, future_unknown_days = 0, concat_len = 165, 
                 lr = 5e-3, epochs = 2, hidden_size = 8, dropout = 0.0, groups = 1, levels = 3, stacks = 2, decompose = "No")
    elif run_case == 1002:
        runFinancial(run_args, dataset_name = 'greek_scinet_dataset_load_athens', window_size = 168, horizon = 24, future_unknown_days = 0, concat_len = 165, 
                 lr = 5e-3, epochs = 2, hidden_size = 8, dropout = 0.0, groups = 1, levels = 3, stacks = 2, decompose = "No")
    elif run_case == 1003:
        runFinancial(run_args, dataset_name = 'greek_scinet_dataset_load_athens_thess', window_size = 168, horizon = 24, future_unknown_days = 0, concat_len = 165, 
                 lr = 5e-3, epochs = 2, hidden_size = 8, dropout = 0.0, groups = 1, levels = 3, stacks = 2, decompose = "No")
    elif run_case == 1004:
        runFinancial(run_args, dataset_name = 'greek_scinet_dataset_load_load', window_size = 168, horizon = 24, future_unknown_days = 0, concat_len = 165, 
                 lr = 5e-3, epochs = 2, hidden_size = 8, dropout = 0.0, groups = 1, levels = 3, stacks = 2, decompose = "No")
        runFinancial(run_args, dataset_name = 'greek_scinet_dataset_load_athens', window_size = 168, horizon = 24, future_unknown_days = 0, concat_len = 165, 
                 lr = 5e-3, epochs = 2, hidden_size = 8, dropout = 0.0, groups = 1, levels = 3, stacks = 2, decompose = "No")
        runFinancial(run_args, dataset_name = 'greek_scinet_dataset_load_athens_thess', window_size = 168, horizon = 24, future_unknown_days = 0, concat_len = 165, 
                 lr = 5e-3, epochs = 2, hidden_size = 8, dropout = 0.0, groups = 1, levels = 3, stacks = 2, decompose = "No")
    elif run_case == 1005:
        runFinancial(run_args, dataset_name = 'greek_scinet_dataset_load_load', window_size = 168, horizon = 24, future_unknown_days = 0, concat_len = 165, 
                 lr = 5e-3, epochs = 200, hidden_size = 8, dropout = 0.0, groups = 1, levels = 3, stacks = 2, decompose = "No")
        runFinancial(run_args, dataset_name = 'greek_scinet_dataset_load_athens', window_size = 168, horizon = 24, future_unknown_days = 0, concat_len = 165, 
                 lr = 5e-3, epochs = 200, hidden_size = 8, dropout = 0.0, groups = 1, levels = 3, stacks = 2, decompose = "No")
        runFinancial(run_args, dataset_name = 'greek_scinet_dataset_load_athens_thess', window_size = 168, horizon = 24, future_unknown_days = 0, concat_len = 165, 
                 lr = 5e-3, epochs = 200, hidden_size = 8, dropout = 0.0, groups = 1, levels = 3, stacks = 2, decompose = "No") 

def printGPUInformation():
    try:
        gpu_devices = os.environ["CUDA_VISIBLE_DEVICES"]
    except:
        gpu_devices = ""
    print(f"CUDA_VISIBLE_DEVICES = {gpu_devices}")
    
def setGPU(gpu):
    if gpu != '':
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    

def runTestCaseWithTimeMeasurement(run_args):
    printImportantMessage(f"Start run_case {run_args.run_case}", None)
    printGPUInformation()
    start_time = timer()
    
    runTestCase(run_args)
    
    end_time = timer()
    print(f"Run run_case {run_args.run_case} in {timedelta(seconds = end_time - start_time)}")
    printImportantMessage(f"End run_case {run_args.run_case}", None)

def main():
    parser = argparse.ArgumentParser(description='Multiple run script')

    parser.add_argument('--run_cases', nargs='+', type=int, default=[-2])
    parser.add_argument('--python_name',type=str, default='python', choices=['python', 'python3'])
    parser.add_argument('--log',type=str, default='Yes', choices=['Yes', 'No'])
    parser.add_argument('--gpu',type=str, default='', choices=['', '0', '1', '2', '3'])
    parser.add_argument('--run_same_experiment',type=str, default='No', choices=['Yes', 'No'])

    mult_run_args = parser.parse_args()

    setGPU(mult_run_args.gpu)

    print(f"run_cases = {mult_run_args.run_cases}")

    for i in range(0, len(mult_run_args.run_cases)):
        run_args = RunArgs(mult_run_args, i)
        runTestCaseWithTimeMeasurement(run_args)
    
if __name__ == "__main__":
    main()