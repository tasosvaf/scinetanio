from datetime import datetime
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

####################################Folders##################################################################

def getParentDirPath(dirpath):
    parent_dirpath, _ = os.path.split(dirpath)
    return parent_dirpath

def getRepoPath():
    dirpath = os.path.dirname(__file__)
    parent_dirpath = getParentDirPath(dirpath)
    parent2_dirpath = getParentDirPath(parent_dirpath)
    
    return parent2_dirpath
    
def createFolderInRepoPath(folder_name):
    folder_path = os.path.join(getRepoPath(), folder_name)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    return folder_path

def createFolderInSciNetFolder(folder_name):
    folder_path = os.path.join(getRepoPath(), "SCINet", folder_name)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    return folder_path

def getListOfFiles(filepath):
    files = [f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))]

    return files

def getListOfFilesAsFilepaths(filepath):
    files = getListOfFiles(filepath)

    filepaths = [os.path.join(filepath, file) for file in files]

    return filepaths

def isFileEmpty(filepath):
    return os.path.isfile(filepath) and os.path.getsize(filepath) == 0

def isFileWithOnlyInformationAboutTensorflowLibrary(filepath):
    false_error_string = ": I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1"
    with open(filepath, 'r') as file:
            text = file.read().rstrip()
    return os.path.isfile(filepath) and os.path.getsize(filepath) < 150 and false_error_string in text

def isFalsePositiveErrorFile(filepath):
    if (filepath == None) or (not os.path.isfile(filepath)):
        return False
    return isFileEmpty(filepath) or isFileWithOnlyInformationAboutTensorflowLibrary(filepath)

def deleteFileIfIsFalsePositiveErrorFile(filepath):
    if isFalsePositiveErrorFile(filepath):
        os.remove(filepath)
        print("False positive error file was deleted")

def deleteFileIfExist(filepath):
    if os.path.isfile(filepath):
        os.remove(filepath)  

def deleteFilesInFolder(folderpath):
    if not os.path.exists(folderpath):
        return
    
    filepaths = getListOfFilesAsFilepaths(folderpath)

    for filepath in filepaths:
        os.remove(filepath)

########################################Logging################################################################
    
def print_output_in_specific_file(message, filename):
    if filename != None:
        with open(filename, 'a') as file:
            file.write(message + '\n')
    print(message)
    
#######################################Anio general#############################################################
    
anio_dict = {'no_anio': 0, 'anio1': 1, 'anio2': 2, 'anio3': 3, 'anio4': 4, 'anio5': 5}
anio_input_options = ['Yes', 'No']

def getAnioCode(anio):
    anio_code = anio_dict[anio]
    return anio_code

def getAnioOptions():
    return list(anio_dict.keys())

def getAnioInputOptions():
    return anio_input_options

def getGreekEnergyCSVName(anio, anio_input):
    if anio != 'no_anio':
        csv_name = 'greek_energy_an{}{}.csv'.format(getAnioCode(anio), anio_input)
    else:
        csv_name = 'greek_energy.csv'
        
    return csv_name

def getGreekEnergyFeatures(anio, anio_input):
    if (anio == 'no_anio'):
        features = ['Athens_temp', 'Thessaloniki_temp']
    elif (anio == 'anio1'):
        if anio_input  == "Yes": 
            features = ['Athens_temp', 'Thessaloniki_temp',
                        'Athens_temp_yesterday','Athens_temp_week','Athens_temp_month',
                        'Thessaloniki_temp_yesterday','Thessaloniki_temp_week','Thessaloniki_temp_month',
                        'anio_load_yesterday','anio_load_week', 'anio_load_month',
                        'load_yesterday','load_week','load_month']
        else:
            features = ['Athens_temp', 'Thessaloniki_temp']
    elif (anio in ['anio2', 'anio3']):
        if anio_input  == "Yes": 
            features = ['Athens_temp', 'Thessaloniki_temp',
                        'Athens_temp_yesterday','Athens_temp_week','Athens_temp_month',
                        'Thessaloniki_temp_yesterday','Thessaloniki_temp_week','Thessaloniki_temp_month',
                        'anio_load_yesterday','anio_load_week', 'anio_load_month']
        else:
            features = ['Athens_temp', 'Thessaloniki_temp']
    elif (anio == 'anio4'):
        if anio_input  == "Yes": 
            features = ['Athens_temp', 'Thessaloniki_temp',
                        'Athens_temp_yesterday',
                        'Thessaloniki_temp_yesterday',
                        'anio_load_yesterday']
        else:
            features = ['Athens_temp', 'Thessaloniki_temp']
    elif (anio == 'anio5'):
        if anio_input  == "Yes": 
            features = ['Athens_temp', 'Thessaloniki_temp',
                        'Athens_temp_yesterday','Athens_temp_week',
                        'Thessaloniki_temp_yesterday','Thessaloniki_temp_week',
                        'anio_load_yesterday','anio_load_week']
        else:
            features = ['Athens_temp', 'Thessaloniki_temp']
    else:
        raise Exception("anio value was invalid")
    
    return features

def getGreekEnergyTarget(anio):
    if anio == "no_anio":
        target = 'TOTAL_CONS'
    else:
        target = 'new_target'
        
    return target

def getETThDatasetCSVName(dataset_name, anio, anio_input):
    if anio != 'no_anio':
        csv_name = f'{dataset_name}_an{getAnioCode(anio)}{anio_input}.csv'
    else:
        csv_name = f'{dataset_name}.csv'
        
    return csv_name

def getETThDatasetFeatures(anio, anio_input):
    if (anio == 'no_anio'):
        features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
    elif (anio == 'anio1'):
        if anio_input  == "Yes": 
            features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL',
                        'HUFL_yesterday','HUFL_week','HUFL_month',
                        'HULL_yesterday','HULL_week','HULL_month',
                        'MUFL_yesterday','MUFL_week','MUFL_month',
                        'MULL_yesterday','MULL_week','MULL_month',
                        'LUFL_yesterday','LUFL_week','LUFL_month',
                        'LULL_yesterday','LULL_week','LULL_month',
                        'anio_OT_yesterday','anio_OT_week', 'anio_OT_month',
                        'OT_yesterday','OT_week','OT_month']
        else:
            features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
    elif (anio in ['anio2', 'anio3']):
        if anio_input  == "Yes": 
            features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL',
                        'HUFL_yesterday','HUFL_week','HUFL_month',
                        'HULL_yesterday','HULL_week','HULL_month',
                        'MUFL_yesterday','MUFL_week','MUFL_month',
                        'MULL_yesterday','MULL_week','MULL_month',
                        'LUFL_yesterday','LUFL_week','LUFL_month',
                        'LULL_yesterday','LULL_week','LULL_month',
                        'anio_OT_yesterday','anio_OT_week', 'anio_OT_month']
        else:
            features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
    elif (anio == 'anio4'):
        if anio_input  == "Yes": 
            features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL',
                        'HUFL_yesterday',
                        'HULL_yesterday',
                        'MUFL_yesterday',
                        'MULL_yesterday',
                        'LUFL_yesterday',
                        'LULL_yesterday',
                        'anio_OT_yesterday']
        else:
            features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
    elif (anio == 'anio5'):
        if anio_input  == "Yes": 
            features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL',
                        'HUFL_yesterday','HUFL_week',
                        'HULL_yesterday','HULL_week',
                        'MUFL_yesterday','MUFL_week',
                        'MULL_yesterday','MULL_week',
                        'LUFL_yesterday','LUFL_week',
                        'LULL_yesterday','LULL_week',
                        'anio_OT_yesterday','anio_OT_week']
        else:
            features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
    else:
        raise Exception("anio value was invalid")
    
    return features

def getETThDatasetTarget(anio):
    if anio == "no_anio":
        target = 'OT'
    else:
        target = 'new_target'
        
    return target

#######################################Experiment naming#############################################################

def getDatetimeAsString():
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return date

def getExperimentName(seq_len, label_len, pred_len, lr, levels, dropout, stacks, 
                      features, shift_data_y, decompose, data, train_epochs, 
                      future_unknown_days, anio, anio_input):
    experiment_date = getDatetimeAsString()
    experiment_name = 'seq{}_lab{}_pr{}_lr{}_lev{}_dp{}_st{}_ft{}_shif{}_dec{}_dat{}_ep{}_ukn{}_an{}{}_{}'.format(seq_len, label_len, pred_len, lr, levels, 
                                                                                                                  dropout, stacks, features, shift_data_y, decompose,
                                                                                                                  data[:5], train_epochs, future_unknown_days, 
                                                                                                                  getAnioCode(anio), anio_input, experiment_date)
    return experiment_name

def getExperimentNameFromArgs(args):
    return getExperimentName(args.seq_len, args.label_len, args.pred_len, args.lr, args.levels, args.dropout, args.stacks, 
                             args.features, args.shift_data_y, args.decompose, args.data, args.train_epochs, 
                             args.future_unknown_days, args.anio, args.anio_input)

def createSettingName(model, data, features, seq_len, label_len, pred_len, lr, 
                      batch_size, hidden_size, stacks, levels, dropout, 
                      inverse, decompose, anio, anio_input, index):
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_bs{}_hid{}_s{}_l{}_dp{}_inv{}_dec{}_an{}{}itr{}'.format(model, data, features, seq_len, label_len, pred_len, lr, 
                                                                                                      batch_size, hidden_size, stacks, levels, dropout, 
                                                                                                      inverse, decompose, getAnioCode(anio), anio_input, index)
    return setting
    
def createSettingNameFromArgs(args, index = 0):
    setting = createSettingName(args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len, args.lr, 
                                args.batch_size, args.hidden_size, args.stacks, args.levels, args.dropout, 
                                args.inverse, args.decompose, args.anio, args.anio_input, index)
    return setting
    
    
#######################################Get latest experiment results#############################################################

def getMostRecentFinishedExperimentFilepathInFolder(experiment_name, folder_name):
    folder_path = createFolderInRepoPath(folder_name)
    filepaths = getListOfFilesAsFilepaths(folder_path)
    
    pure_experiment_name = removeTrailingTime(experiment_name)
    
    valid_filepaths = []
    for filepath in filepaths:
        if pure_experiment_name in filepath and os.path.getsize(filepath) > 100:
            valid_filepaths.append(filepath)
    
    if len(valid_filepaths) == 0:
        return None
    
    valid_filepaths.sort()
    
    return valid_filepaths[-1]

def isExperimentAlreadyRunned(experiment_name):
    output_filepath = getMostRecentFinishedExperimentFilepathInFolder(experiment_name, "output")
    
    if output_filepath != None:
        return True
    else:
        return False
    
def getLatestMetricsFilepathFromETThExperiment(experiment_name):
    metrics_filepath = getMostRecentFinishedExperimentFilepathInFolder(experiment_name, "metrics")
    
    return metrics_filepath
    
def getLatestMetricsFilepathFromETThRun(data = 'greek_energy', features='M', future_unknown_days = 0, shift_data_y = "Yes", anio = "no_anio", anio_input = "Yes", seq_len = 96, 
            label_len = 48, pred_len = 48, train_epochs = 150, batch_size = 32, lr = 1e-5, hidden_size = 1, dropout = 0.5, groups = 1, levels = 3, stacks = 1, decompose = False):

    experiment_name = getExperimentName(seq_len, label_len, pred_len, lr, levels, dropout, stacks, features, shift_data_y, decompose, data, train_epochs, 
                                        future_unknown_days, anio, anio_input)
    
    metrics_filepath = getLatestMetricsFilepathFromETThExperiment(experiment_name)
    
    return metrics_filepath

def getLatestMetricsFilesFromFiles(files):
    experiment_names = [keepExperimentNameOnly(file) for file in files]
    unique_experiment_names = list(set(experiment_names))
    
    latest_files = []
    for experiment_name in unique_experiment_names:
        metrics_filespath = getLatestMetricsFilepathFromETThExperiment(experiment_name)
        metrics_file = os.path.basename(metrics_filespath)
        assert(metrics_file in files)
        latest_files.append(metrics_file)
    
    return latest_files

def getOnlyFilesThatBelongToSpecificExperiments(files, given_experiment_names):
    if given_experiment_names == None or len(given_experiment_names) == 0:
        return files
    
    result_files = []
    for file in files:
        experiment_name = keepExperimentNameOnly(file)
        if experiment_name in given_experiment_names:
            result_files.append(file)
            
    return result_files

def getOnlyFilesThatApplyToAndFilters(files, and_filters):
    if and_filters == None or len(and_filters) == 0:
        return files
    
    result_files = []
    for file in files:
        if all(and_filter in file for and_filter in and_filters):
            result_files.append(file)
            
    return result_files

def getOnlyFilesThatApplyToOrFilters(files, or_filters):
    if or_filters == None or len(or_filters) == 0:
        return files
    
    result_files = []
    for file in files:
        if any(and_filter in file for and_filter in or_filters):
            result_files.append(file)
            
    return result_files

def getLatestMetricsFilepathsThatBelongToGivenFilters(given_experiment_names = [], and_filters = [], or_filters = []):
    metrics_folderpath = createFolderInRepoPath("metrics")
    files = getListOfFiles(metrics_folderpath)
    files = getLatestMetricsFilesFromFiles(files)
    files = getOnlyFilesThatBelongToSpecificExperiments(files, given_experiment_names)
    files = getOnlyFilesThatApplyToAndFilters(files, and_filters)
    files = getOnlyFilesThatApplyToOrFilters(files, or_filters)
    
    filepaths = [os.path.join(metrics_folderpath, file) for file in files]
    return filepaths

#######################################Test cases#############################################################

expected_metrics = {
    "seq48_lab24_pr24_lr0.003_lev3_dp0.5_st1_ftM_shifNo_decNo_datETTh1_ep2_ukn0_an0No": {
        "valid": {
            "mape": 0.143620,
            "c1": 0.607750,
            "c2": 0.116651,
            "c3": 0.275597
        },
        "test": {
            "mape": 0.526607,
            "c1": 0.254918,
            "c2": 0.091363,
            "c3": 0.653718
        }
    },
    "seq96_lab48_pr48_lr0.009_lev3_dp0.25_st1_ftM_shifNo_decNo_datETTh1_ep2_ukn0_an0No": {
        "valid": {
            "mape": 0.201245,
            "c1": 0.453027,
            "c2": 0.139869,
            "c3": 0.407102
        },
        "test": {
            "mape": 0.769455,
            "c1": 0.172664,
            "c2": 0.066476,
            "c3": 0.760859
        }
    },
    "seq336_lab168_pr168_lr0.0005_lev3_dp0.5_st1_ftM_shifNo_decNo_datETTh1_ep2_ukn0_an0No": {
        "valid": {
            "mape": 0.212610,
            "c1": 0.482516,
            "c2": 0.135999,
            "c3": 0.381485
        },
        "test": {
            "mape": 0.914522,
            "c1": 0.196494,
            "c2": 0.072043,
            "c3": 0.731463
        }
    },
    "seq336_lab336_pr336_lr0.0001_lev4_dp0.5_st1_ftM_shifNo_decNo_datETTh1_ep2_ukn0_an0No": {
        "valid": {
            "mape": 0.358955,
            "c1": 0.209229,
            "c2": 0.087203,
            "c3": 0.703568
        },
        "test": {
            "mape": 2.030646,
            "c1": 0.095919,
            "c2": 0.029108,
            "c3": 0.874973
        }
    },
    "seq736_lab720_pr720_lr5e-05_lev5_dp0.5_st1_ftM_shifNo_decNo_datETTh1_ep2_ukn0_an0No": {
        "valid": {
            "mape": 0.530404,
            "c1": 0.182912,
            "c2": 0.092626,
            "c3": 0.724462
        },
        "test": {
            "mape": 2.095933,
            "c1": 0.077083,
            "c2": 0.023611,
            "c3": 0.899306
        }
    },
    "seq64_lab24_pr24_lr0.007_lev3_dp0.25_st1_ftS_shifNo_decNo_datETTh1_ep2_ukn0_an0No": {
        "valid": {
            "mape": 0.138709,
            "c1": 0.577254,
            "c2": 0.126667,
            "c3": 0.296080
        },
        "test": {
            "mape": 0.486648,
            "c1": 0.277195,
            "c2": 0.100551,
            "c3": 0.622254
        }
    },
    "seq720_lab48_pr48_lr0.0001_lev4_dp0.5_st1_ftS_shifNo_decNo_datETTh1_ep2_ukn0_an0No": {
        "valid": {
            "mape": 0.190321,
            "c1": 0.470458,
            "c2": 0.158866,
            "c3": 0.370677
        },
        "test": {
            "mape": 0.689724,
            "c1": 0.198615,
            "c2": 0.077646,
            "c3": 0.723739
        }
    },
    "seq720_lab168_pr168_lr5e-05_lev4_dp0.5_st1_ftS_shifNo_decNo_datETTh1_ep2_ukn0_an0No": {
        "valid": {
            "mape": 0.228035,
            "c1": 0.347945,
            "c2": 0.156838,
            "c3": 0.495217
        },
        "test": {
            "mape": 0.852692,
            "c1": 0.188883,
            "c2": 0.073718,
            "c3": 0.737399
        }
    },
    "seq720_lab336_pr336_lr0.001_lev4_dp0.5_st1_ftS_shifNo_decNo_datETTh1_ep2_ukn0_an0No": {
        "valid": {
            "mape": 0.230073,
            "c1": 0.293179,
            "c2": 0.164505,
            "c3": 0.542316
        },
        "test": {
            "mape": 1.048889,
            "c1": 0.155320,
            "c2": 0.059901,
            "c3": 0.784778
        }
    },
    "seq736_lab720_pr720_lr0.0001_lev5_dp0.5_st1_ftS_shifNo_decNo_datETTh1_ep2_ukn0_an0No": {
        "valid": {
            "mape": 0.246213,
            "c1": 0.290392,
            "c2": 0.114108,
            "c3": 0.595500
        },
        "test": {
            "mape": 1.203680,
            "c1": 0.106944,
            "c2": 0.035417,
            "c3": 0.857639
        }
    },
    "seq96_lab48_pr48_lr1e-05_lev3_dp0.5_st1_ftM_shifYes_decNo_datgreek_ep2_ukn0_an0No": {
        "valid": {
            "mape": 0.089771,
            "c1": 0.640664,
            "c2": 0.183992,
            "c3": 0.175345
        },
        "test": {
            "mape": 0.076963,
            "c1": 0.717707,
            "c2": 0.160836,
            "c3": 0.121458
        }
    },
    "seq96_lab48_pr48_lr1e-05_lev3_dp0.5_st1_ftM_shifYes_decNo_datgreek_ep2_ukn0_an1Yes": {
        "valid": {
            "mape": 0.071900,
            "c1": 0.759505,
            "c2": 0.118597,
            "c3": 0.121898
        },
        "test": {
            "mape": 0.053638,
            "c1": 0.848174,
            "c2": 0.099574,
            "c3": 0.052252
        }
    },
    "seq96_lab48_pr48_lr1e-05_lev3_dp0.5_st1_ftM_shifYes_decNo_datgreek_ep2_ukn0_an1No": {
        "valid": {
            "mape": 0.063428,
            "c1": 0.804918,
            "c2": 0.097968,
            "c3": 0.097115
        },
        "test": {
            "mape": 0.054124,
            "c1": 0.844469,
            "c2": 0.103080,
            "c3": 0.052451
        }
    },
    "seq96_lab48_pr48_lr1e-05_lev3_dp0.5_st1_ftM_shifYes_decNo_datgreek_ep2_ukn0_an2Yes": {
        "valid": {
            "mape": 0.060291,
            "c1": 0.816000,
            "c2": 0.104675,
            "c3": 0.079325
        },
        "test": {
            "mape": 0.052805,
            "c1": 0.851625,
            "c2": 0.099613,
            "c3": 0.048763
        }
    },
    "seq96_lab48_pr48_lr1e-05_lev3_dp0.5_st1_ftM_shifYes_decNo_datgreek_ep2_ukn0_an2No": {
        "valid": {
            "mape": 0.063456,
            "c1": 0.805178,
            "c2": 0.097704,
            "c3": 0.097118
        },
        "test": {
            "mape": 0.054165,
            "c1": 0.844354,
            "c2": 0.102972,
            "c3": 0.052675
        }
    },
    "seq96_lab48_pr48_lr1e-05_lev3_dp0.5_st1_ftM_shifYes_decNo_datgreek_ep2_ukn0_an3Yes": {
        "valid": {
            "mape": 0.061054,
            "c1": 0.814528,
            "c2": 0.102355,
            "c3": 0.083117
        },
        "test": {
            "mape": 0.053187,
            "c1": 0.851422,
            "c2": 0.099874,
            "c3": 0.048704
        }
    },  
    "seq96_lab48_pr48_lr1e-05_lev3_dp0.5_st1_ftM_shifYes_decNo_datgreek_ep2_ukn0_an3No": {
        "valid": {
            "mape": 0.063428,
            "c1": 0.804918,
            "c2": 0.097968,
            "c3": 0.097115
        },
        "test": {
            "mape": 0.054124,
            "c1": 0.844469,
            "c2": 0.103080,
            "c3": 0.052451
        }
    },  
}

def removeFirstWord(filename):
    index = filename.find("_")
    experiment_and_time = filename[index+1:]
    
    return experiment_and_time
    
def removeTrailingTime(filename):
    index = filename.find("_202")
    experiment_name = filename[:index]
    
    return experiment_name

def keepExperimentNameOnly(filename):
    experiment_and_time = removeFirstWord(filename)
    experiment_name = removeTrailingTime(experiment_and_time)
    
    return experiment_name

def changeDetailToReadableDetail(index, detail):
    if index == 0:
        readable_detail = detail.replace("seq", "history ")
        return readable_detail
    elif index == 1:
        readable_detail = detail.replace("lab", "label ")
        return readable_detail
    elif index == 2:
        readable_detail = detail.replace("pr", "prediction ")
        return readable_detail
    elif index == 3:
        readable_detail = detail.replace("lr", "learning rate ")
        return readable_detail
    elif index == 4:
        readable_detail = detail.replace("lev", "levels ")
        return readable_detail
    elif index == 5:
        readable_detail = detail.replace("dp", "dropout ")
        return readable_detail
    elif index == 6:
        readable_detail = detail.replace("st", "stacks ")
        return readable_detail
    elif index == 7:
        if detail == "ftM":
            readable_detail = "multivariate"
        elif detail == "ftS":
            readable_detail = "univariate"
        else:
            raise Exception("feature detail was invalid")
        return readable_detail
    elif index == 8:
        if detail == "shifNo":
            readable_detail = "no shift data"
        elif detail == "shifYes":
            readable_detail = "shift data"
        else:
            raise Exception("shift data detail was invalid")
        return readable_detail
    elif index == 9:
        readable_detail = detail.replace("dec", "decompose ")
        if detail == "decNo":
            readable_detail = "no decompose"
        elif detail == "decYes":
            readable_detail = "decompose"
        else:
            raise Exception("decompose detail was invalid")
        return readable_detail
    elif index == 10:
        readable_detail = detail.replace("dat", "dataset ")
        return readable_detail
    elif index == 11:
        readable_detail = detail.replace("ep", "epochs ")
        return readable_detail
    elif index == 12:
        readable_detail = detail.replace("ukn", "future unknown days ")
        return readable_detail
    elif index == 13:
        if detail.startswith("an0"):
            readable_detail = "no anio"
        else:
            anio_code = detail[2]
            assert(anio_code in ['1', '2', '3', '4', '5'])
            if "Yes" in detail:
                readable_detail = f"anio {anio_code} with input"
            elif "No" in detail:
                readable_detail = f"anio {anio_code} without input"
            else:
                raise Exception("anio input detail was invalid")
        return readable_detail
    else:
        raise Exception("index was invalid")
    
def changeDetailToSortableDetail(index, detail):
    if index == 0:
        sortable_string = detail.replace("seq", "")
        sortable_number = int(sortable_string)
        return sortable_number
    elif index == 1:
        sortable_string = detail.replace("lab", "")
        sortable_number = int(sortable_string)
        return sortable_number
    elif index == 2:
        sortable_string = detail.replace("pr", "")
        sortable_number = int(sortable_string)
        return sortable_number
    elif index == 3: #lr
        return detail
    elif index == 4:
        sortable_string = detail.replace("lev", "")
        sortable_number = int(sortable_string)
        return sortable_number
    elif index == 5: #dp
        return detail
    elif index == 6:
        sortable_string = detail.replace("st", "")
        sortable_number = int(sortable_string)
        return sortable_number
    elif index == 7: #ft
        return detail
    elif index == 8: #shift
        return detail
    elif index == 9: #dec
        return detail
    elif index == 10: #dat
        return detail
    elif index == 11:
        sortable_string = detail.replace("ep", "")
        sortable_number = int(sortable_string)
        return sortable_number
    elif index == 12:
        sortable_string = detail.replace("ukn", "")
        sortable_number = int(sortable_string)
        return sortable_number
    elif index == 13: # an
        return detail
    else:
        raise Exception("index was invalid")
    
all_readable_components = ['se', 'la', 'pr', 'lr', 'le', 'dp', 'st', 'ft', 'sh', 'de', 'da', 'ep', 'uk', 'an']

def checkReadableComponents(readable_components):
    for readable_component in readable_components:
        assert(readable_component in all_readable_components)
    
def getReadableExperimentName(experiment_name, readable_components):
    details = experiment_name.split("_")
    
    readable_details  = []
    for index, detail in enumerate(details):
        starting_detail = detail[:2]
        if starting_detail in readable_components:
            readable_detail = changeDetailToReadableDetail(index, detail)
            readable_details.append(readable_detail)
    
    assert(len(readable_details) > 0)
    
    readable_experiment_name = " - ".join(readable_details)
    return readable_experiment_name

def getSortableTupleForReadableExperimentName(experiment_name, readable_components):
    details = experiment_name.split("_")
    
    sortable_details  = []
    for index, detail in enumerate(details):
        starting_detail = detail[:2]
        if starting_detail in readable_components:
            sortable_detail = changeDetailToSortableDetail(index, detail)
            sortable_details.append(sortable_detail)
    
    assert(len(sortable_details) > 0)
    
    sortable_tuple = tuple(sortable_details)
    return sortable_tuple

def metric_filepath_sort(filepath, readable_components):
    file = os.path.basename(filepath)
    experiment_name = keepExperimentNameOnly(file)
    sortable_tuple = getSortableTupleForReadableExperimentName(experiment_name, readable_components)
    
    return sortable_tuple

def getExpectedMetrics(experiment, case):
    experiment_expected_metrics = expected_metrics[experiment][case]
    expected_mape = experiment_expected_metrics['mape']
    expected_c1 = experiment_expected_metrics['c1']
    expected_c2 = experiment_expected_metrics['c2']
    expected_c3 = experiment_expected_metrics['c3']
    
    return expected_mape, expected_c1, expected_c2, expected_c3

def checkIfSameAsExpected(expected_metric, metric, filename, metric_type):
    if (abs(expected_metric - metric) > 1e-5):
        raise Exception(f"Error in filename: {filename} with (expected_{metric_type}, {metric_type}, abs(diff)) = ({expected_metric}, {metric}, {abs(expected_metric - metric)})")

def testMetricsForETThAlgorithm(filepath, case, mape, c1, c2, c3):
    filename = os.path.basename(filepath)
    if 'ep2' not in filename:
        return

    experiment_name = keepExperimentNameOnly(filename)
    if experiment_name not in expected_metrics:
        return
    
    expected_mape, expected_c1, expected_c2, expected_c3 = getExpectedMetrics(experiment_name, case)
    
    checkIfSameAsExpected(expected_mape, mape, filename, 'mape')
    checkIfSameAsExpected(expected_c1, c1, filename, 'c1')
    checkIfSameAsExpected(expected_c2, c2, filename, 'c2')
    checkIfSameAsExpected(expected_c3, c3, filename, 'c3')
    
    print(f"Test completed successfully of {experiment_name}")
    
 #######################################Metrics- selection process#############################################################

def SelectRowsInFirstDimension(array, start, end):
    if array.ndim == 1:
        return array[start:end]
    elif array.ndim == 2:
        return array[start:end,:]
    elif array.ndim == 3:
        return array[start:end,:,:]

def GetBorders(length, bucket):
    borders = []
    for i in range(0,bucket):
        borders.append((i*length)//bucket)
    borders.append(length)
    return borders

def CreateListOfArrayParts(arr, borders):
    parts = []
    for i in range(0, len(borders)-1):
       parts.append(SelectRowsInFirstDimension(arr, borders[i], borders[i+1]))

    return parts

def CreateListOfSubSelectionsFromParts(parts):
    sub_selections = []
    for i in range(0, len(parts)):
        parts_collection = []
        for j in range(0, len(parts)):
            if j != i:
                parts_collection.append(parts[j])
        sub_selection = np.concatenate(parts_collection)
        sub_selections.append(sub_selection)
        
    return sub_selections

def CreateListOfSubSelections(arr, bucket):
    borders = GetBorders(arr.shape[0], bucket)
    parts = CreateListOfArrayParts(arr, borders)
    sub_selections = CreateListOfSubSelectionsFromParts(parts)
    
    return sub_selections

def GetMeanAndStdFromMetricsArr(metrics_list, bucket):
    metrics_arr = np.array(metrics_list)
    sub_selections = CreateListOfSubSelections(metrics_arr, bucket)
    metrics = []
    for sub_selection in sub_selections:
        metrics.append(np.mean(sub_selection))
    
    mean = np.mean(metrics)
    std = np.std(metrics) 
    return mean, std

#######################################Metrics#############################################################

def addPreFixAndSuffixToDictKeys(dict, prefix, suffix):
    return {prefix + k + suffix: v for k, v in dict.items()}

def changeAnyValueInDictToBeAList(dict):
    return {k: [v] for k, v in dict.items()}


class Metric:
    def __init__(self, metric_name, metric = 0, 
                 metric_mean_5 = 0, metric_std_5 = 0, 
                 metric_mean_10 = 0, metric_std_10 = 0,
                 metric_mean_20 = 0, metric_std_20 = 0):
        self.metric_name = metric_name
        self.metric = metric
        self.metric_mean_5 = metric_mean_5
        self.metric_std_5 = metric_std_5
        self.metric_mean_10 = metric_mean_10
        self.metric_std_10 = metric_std_10
        self.metric_mean_20 = metric_mean_20
        self.metric_std_20 = metric_std_20
    
    def print(self):
        print("metric_name: ", self.metric_name)
        print("metric: ", self.metric)
        print("metric_mean_5: ", self.metric_mean_5)
        print("metric_std_5: ", self.metric_std_5)
        print("metric_mean_10: ", self.metric_mean_10)
        print("metric_std_10: ", self.metric_std_10)
        print("metric_mean_20: ", self.metric_mean_20)
        print("metric_std_20: ", self.metric_std_20)
    
    def to_dict_using_metric_name(self):
        dict = {
            f'{self.metric_name}': self.metric,
            f'{self.metric_name}_mean_5': self.metric_mean_5,
            f'{self.metric_name}_std_5': self.metric_std_5,
            f'{self.metric_name}_mean_10': self.metric_mean_10,
            f'{self.metric_name}_std_10': self.metric_std_10,
            f'{self.metric_name}_mean_20': self.metric_mean_20,
            f'{self.metric_name}_std_20': self.metric_std_20
        }
        return dict
        
    def to_dict(self, key_prefix, key_suffix):
        return addPreFixAndSuffixToDictKeys(self.to_dict_using_metric_name(), key_prefix, key_suffix)


class Metrics:
    def __init__(self, 
                 RSE = Metric(metric_name = 'RSE'), 
                 CORR = Metric(metric_name = 'CORR'),
                 Corr = Metric(metric_name = 'Corr'),
                 MAE = Metric(metric_name = 'MAE'),
                 MSE = Metric(metric_name = 'MSE'),
                 RMSE = Metric(metric_name = 'RMSE'),
                 MAPE = Metric(metric_name = 'MAPE'),
                 MSPE = Metric(metric_name = 'MSPE')):
        self.RSE = RSE
        self.CORR = CORR
        self.Corr = Corr
        self.MAE = MAE
        self.MSE = MSE
        self.RMSE = RMSE
        self.MAPE = MAPE
        self.MSPE = MSPE
    
    def print(self):
        print("RSE: ")
        self.RSE.print()
        print("CORR: ")
        self.CORR.print()
        print("Corr: ")
        self.Corr.print()
        print("MAE: ")
        self.MAE.print()
        print("MSE: ")
        self.MSE.print()
        print("RMSE: ")
        self.RMSE.print()
        print("MAPE: ")
        self.MAPE.print()
        print("MSPE: ")
        self.MSPE.print()
    
    def to_dict(self, key_prefix, key_suffix):
        RSE_dict = self.RSE.to_dict(key_prefix, key_suffix)
        CORR_dict = self.CORR.to_dict(key_prefix ,key_suffix)
        Corr_dict = self.Corr.to_dict(key_prefix ,key_suffix)
        MAE_dict = self.MAE.to_dict(key_prefix ,key_suffix)
        MSE_dict = self.MSE.to_dict(key_prefix ,key_suffix)
        RMSE_dict = self.RMSE.to_dict(key_prefix ,key_suffix)
        MAPE_dict = self.MAPE.to_dict(key_prefix ,key_suffix)
        MSPE_dict = self.MSPE.to_dict(key_prefix ,key_suffix)
        
        dict = {**RSE_dict, **CORR_dict, **Corr_dict, **MAE_dict, **MSE_dict, **RMSE_dict, **MAPE_dict, **MSPE_dict}
        return dict

class MyMetrics:
    def __init__(self,
                 my_mape = Metric(metric_name = 'my_mape'), 
                 my_c1 = Metric(metric_name = 'my_c1'),
                 my_c2 = Metric(metric_name = 'my_c2'),
                 my_c3 = Metric(metric_name = 'my_c3')):
        self.my_mape = my_mape
        self.my_c1 = my_c1
        self.my_c2 = my_c2
        self.my_c3 = my_c3
        
    def print(self):
        print("my_mape: ")
        self.my_mape.print()
        print("my_c1: ")
        self.my_c1.print()
        print("my_c2: ")
        self.my_c2.print()
        print("my_c3: ")
        self.my_c3.print()
        
    def to_dict(self, key_prefix, key_suffix):
        my_mape_dict = self.my_mape.to_dict(key_prefix, key_suffix)
        my_c1_dict = self.my_c1.to_dict(key_prefix ,key_suffix)
        my_c2_dict = self.my_c2.to_dict(key_prefix ,key_suffix)
        my_c3_dict = self.my_c3.to_dict(key_prefix ,key_suffix)
        
        dict = {**my_mape_dict, **my_c1_dict, **my_c2_dict, **my_c3_dict}
        return dict        
        

class AlgorithmMetrics:
    def __init__(self, metrics = Metrics(), mid_metrics = Metrics()):
        self.metrics = metrics
        self.mid_metrics = mid_metrics
        
    def print(self):
        print("metrics: ")
        self.metrics.print()
        print("mid_metrics: ")
        self.mid_metrics.print()
        
    def to_dict(self, key_prefix, key_suffix):
        metrics_dict = self.metrics.to_dict(key_prefix, key_suffix)
        mid_metrics_dict = self.mid_metrics.to_dict(key_prefix + 'mid_', key_suffix)
        
        dict = {**metrics_dict, **mid_metrics_dict}
        return dict

class MyAlgorithmMetrics:
    def __init__(self, my_metrics = MyMetrics(), mid_my_metrics = MyMetrics()):
        self.my_metrics = my_metrics
        self.mid_my_metrics = mid_my_metrics
    
    def print(self):
        print("my_metrics: ")
        self.my_metrics.print()
        print("mid_my_metrics: ")
        self.mid_my_metrics.print()
    
    def to_dict(self, key_prefix, key_suffix):
        my_metrics_dict = self.my_metrics.to_dict(key_prefix, key_suffix)
        mid_my_metrics_dict = self.mid_my_metrics.to_dict(key_prefix + 'mid_', key_suffix)
        
        dict = {**my_metrics_dict, **mid_my_metrics_dict}
        return dict      
   
class EpochMetrics:
    def __init__(self,
               valid_algorithm_metrics_full = AlgorithmMetrics(),
               valid_algorithm_metrics = AlgorithmMetrics(), 
               valid_algorithm_metrics_red = AlgorithmMetrics(),
               test_algorithm_metrics_full = AlgorithmMetrics(),
               test_algorithm_metrics = AlgorithmMetrics(),
               test_algorithm_metrics_red = AlgorithmMetrics(),
               valid_my_algorithm_metrics = MyAlgorithmMetrics(),
               test_my_algorithm_metrics = MyAlgorithmMetrics()):
        self.valid_algorithm_metrics_full = valid_algorithm_metrics_full
        self.valid_algorithm_metrics = valid_algorithm_metrics
        self.valid_algorithm_metrics_red = valid_algorithm_metrics_red
        self.test_algorithm_metrics_full = test_algorithm_metrics_full
        self.test_algorithm_metrics = test_algorithm_metrics
        self.test_algorithm_metrics_red = test_algorithm_metrics_red
        self.valid_my_algorithm_metrics = valid_my_algorithm_metrics
        self.test_my_algorithm_metrics = test_my_algorithm_metrics
    
    def print(self):
        print("valid_algorithm_metrics_full: ")
        self.valid_algorithm_metrics_full.print()
        print("valid_algorithm_metrics: ")
        self.valid_algorithm_metrics.print()
        print("valid_algorithm_metrics_red: ")
        self.valid_algorithm_metrics_red.print()
        print("test_algorithm_metrics_full: ")
        self.test_algorithm_metrics_full.print()
        print("test_algorithm_metrics: ")
        self.test_algorithm_metrics.print()
        print("test_algorithm_metrics_red: ")
        self.test_algorithm_metrics_red.print()
        print("valid_my_algorithm_metrics: ")
        self.valid_my_algorithm_metrics.print()
        print("test_my_algorithm_metrics: ")
        self.test_my_algorithm_metrics.print()
        
    def to_dict(self):
        valid_algorithm_metrics_full_dict = self.valid_algorithm_metrics_full.to_dict('valid_', '_full')
        valid_algorithm_metrics_dict = self.valid_algorithm_metrics.to_dict('valid_', '')
        valid_algorithm_metrics_red_dict = self.valid_algorithm_metrics_red.to_dict('valid_', '_red')
        test_algorithm_metrics_full_dict = self.test_algorithm_metrics_full.to_dict('test_', '_full')
        test_algorithm_metrics_dict = self.test_algorithm_metrics.to_dict('test_', '')
        test_algorithm_metrics_red_dict = self.test_algorithm_metrics_red.to_dict('test_', '_red')
        valid_my_algorithm_metrics_dict = self.valid_my_algorithm_metrics.to_dict('valid_', '')
        test_my_algorithm_metrics_dict = self.test_my_algorithm_metrics.to_dict('test_', '')
        
        dict = {**valid_algorithm_metrics_full_dict, **valid_algorithm_metrics_dict, **valid_algorithm_metrics_red_dict,
                **test_algorithm_metrics_full_dict, **test_algorithm_metrics_dict, **test_algorithm_metrics_red_dict, 
                **valid_my_algorithm_metrics_dict, **test_my_algorithm_metrics_dict}
        
        return dict
    
    def getDictForDataFrame(self):
        dict_with_values = self.to_dict()
        dict_with_lists = changeAnyValueInDictToBeAList(dict_with_values)
        return dict_with_lists

def getDatasetName(filename):
    index = filename.find("dat")
    dataset_name = filename[index+3:index+8]
    
    return dataset_name
    
def anioOfDataset(filename):
    index = filename.find("an")
        
    anio_number = filename[index+2]
    if anio_number == '0':
        return 'no_anio'
    else :
        return 'anio' + anio_number
    
def anioInputOfDataset(filename):
    index = filename.find("an")
        
    anio_input_first_char = filename[index+3]
    if anio_input_first_char == 'Y':
        return 'Yes'
    else :
        return 'No'
    
def getDatesToCalculateMetrics(dataset_name, case):
    if dataset_name.startswith('ETTh'):
        if case == 'valid':
            dates = pd.date_range('2017-07-01','2017-10-31 23:00:00',freq='h')
        elif case == 'test':
            dates = pd.date_range('2017-11-01','2018-02-28 23:00:00',freq='h')
    else:
        if case == 'valid':
            dates = pd.date_range('2017-01-01','2017-12-31 23:00:00',freq='h')
        elif case == 'test':
            dates = pd.date_range('2018-01-01','2018-12-31 23:00:00',freq='h')
    
    return dates

def getMonthIndicesToCalculateMetrics(dataset_name, case):
    dates = getDatesToCalculateMetrics(dataset_name, case)
    indices = find_month_indices(dates)
    
    return indices
    

def calculateTargetsAndDenormalizedValues(dataset_name, anio, anio_input):
    targets = []
    denormalized_values = []
    
    if dataset_name.startswith('ETTh'):
        targets, denormalized_values = createETThDatasetGeneral(dataset_name, anio, anio_input)
    elif dataset_name.startswith('greek'):
        targets, denormalized_values = createGreekDatasetGeneral(anio, anio_input)
    
    return targets, denormalized_values

def getEpochsFromDataFrame(data):
    epochs_list = [int(re.search(r'\d+', x).group()) for x in data.columns if re.search(r'\d+', x)]
    epochs = max(epochs_list) + 1
    
    return epochs

def findIndexInListOfDoublesCore(list, target_value, tolerance):
    try:
        index = list.index(target_value)
    except ValueError:
        index = next((i for i, x in enumerate(list) if abs(x - target_value) < tolerance), None)
        
    return index
def findIndexInListOfDoubles(list, target_value):
    index = findIndexInListOfDoublesCore(list, target_value, 1e-15)
    
    if index != None:
        return index
    
    index = findIndexInListOfDoublesCore(list, target_value, 5e-14)
    
    if index != None:
        return index
    
    index = findIndexInListOfDoublesCore(list, target_value, 1e-14)
    
    return index 
    
def find_month_indices(dates):
    indices = [0]
    current_month = -1
    for i, date in enumerate(dates):        
        if i == 0:
            current_month = date.month
        if date.month != current_month:
            indices.append(i)
            current_month = date.month
    indices.append(len(dates))
    return indices

def denormalizePredictionsAndTrueValues(predictions, true_values, targets, denormalized_values_arr, anio):
    if anio != 'no_anio':
        length = len(true_values)
        first = true_values[0]
        index = findIndexInListOfDoubles(targets, first)
        denormalized_values = denormalized_values_arr[index : index + length]
        predictions = [ (pr + 1) * de for pr,de in zip(predictions, denormalized_values)]
        true_values = [ (tr + 1) * de for tr,de in zip(true_values, denormalized_values)]
        
    return predictions, true_values

def calculateMyMetrics(predictions, true_values, indices):    
    mape_arr = [abs((y-x)/y) if abs(y) > 1e-3 else 0 for x,y in zip(predictions,true_values)]
    
    C1perc_arr = []
    C2perc_arr = []
    C3perc_arr = []
    for i in range(0, len(indices) - 1):
        start = indices[i]
        end = indices[i + 1]

        C1 = len([j for j in mape_arr[start:end] if 0 <= j < 0.1])
        C2 = len([j for j in mape_arr[start:end] if 0.1 <= j <= 0.15])
        C3 = len([j for j in mape_arr[start:end] if j > 0.15])
        Ctotal = C1 + C2 + C3
        
        if Ctotal == 0:
            continue

        C1perc_arr.append(C1/Ctotal)
        C2perc_arr.append(C2/Ctotal)
        C3perc_arr.append(C3/Ctotal)
    
    my_mape = np.mean(mape_arr)
    C1perc = np.mean(C1perc_arr)
    C2perc = np.mean(C2perc_arr)
    C3perc = np.mean(C3perc_arr)
    
    my_mape_mean_5, my_mape_std_5 = GetMeanAndStdFromMetricsArr(mape_arr, 5)
    my_mape_mean_10, my_mape_std_10 = GetMeanAndStdFromMetricsArr(mape_arr, 10)
    my_mape_mean_20, my_mape_std_20 = GetMeanAndStdFromMetricsArr(mape_arr, 20)
    C1perc_mean_5, C1perc_std_5 = GetMeanAndStdFromMetricsArr(C1perc_arr, 5)
    C1perc_mean_10, C1perc_std_10 = GetMeanAndStdFromMetricsArr(C1perc_arr, 10)
    C1perc_mean_20, C1perc_std_20 = GetMeanAndStdFromMetricsArr(C1perc_arr, 20)
    C2perc_mean_5, C2perc_std_5 = GetMeanAndStdFromMetricsArr(C2perc_arr, 5)
    C2perc_mean_10, C2perc_std_10 = GetMeanAndStdFromMetricsArr(C2perc_arr, 10)
    C2perc_mean_20, C2perc_std_20 = GetMeanAndStdFromMetricsArr(C2perc_arr, 20)
    C3perc_mean_5, C3perc_std_5 = GetMeanAndStdFromMetricsArr(C3perc_arr, 5)
    C3perc_mean_10, C3perc_std_10 = GetMeanAndStdFromMetricsArr(C3perc_arr, 10)
    C3perc_mean_20, C3perc_std_20 = GetMeanAndStdFromMetricsArr(C3perc_arr, 20)
            
    my_mape_metric = Metric(metric_name = 'my_mape', metric = my_mape, metric_mean_5 = my_mape_mean_5, metric_std_5 = my_mape_std_5, metric_mean_10 = my_mape_mean_10, metric_std_10 = my_mape_std_10, metric_mean_20 = my_mape_mean_20, metric_std_20 = my_mape_std_20)
    my_c1_metric = Metric(metric_name = 'my_c1', metric = C1perc, metric_mean_5 = C1perc_mean_5, metric_std_5 = C1perc_std_5, metric_mean_10 = C1perc_mean_10, metric_std_10 = C1perc_std_10, metric_mean_20 = C1perc_mean_20, metric_std_20 = C1perc_std_20)
    my_c2_metric = Metric(metric_name = 'my_c2', metric = C2perc, metric_mean_5 = C2perc_mean_5, metric_std_5 = C2perc_std_5, metric_mean_10 = C2perc_mean_10, metric_std_10 = C2perc_std_10, metric_mean_20 = C2perc_mean_20, metric_std_20 = C2perc_std_20)
    my_c3_metric = Metric(metric_name = 'my_c3', metric = C3perc, metric_mean_5 = C3perc_mean_5, metric_std_5 = C3perc_std_5, metric_mean_10 = C3perc_mean_10, metric_std_10 = C3perc_std_10, metric_mean_20 = C3perc_mean_20, metric_std_20 = C3perc_std_20)
    
    myMetrics = MyMetrics(my_mape = my_mape_metric, my_c1 = my_c1_metric, my_c2 = my_c2_metric, my_c3 = my_c3_metric)
    return myMetrics

def calculateMyMetricsFromDenormalizedValues(predictions, true_values, targets, denormalized_values_arr, anio, indices):
    predictions, true_values = denormalizePredictionsAndTrueValues(predictions, true_values, targets, denormalized_values_arr, anio)    
    my_metrics = calculateMyMetrics(predictions, true_values, indices)
    
    return my_metrics

def calculateMyMetricsFromDenormalizedValuesUsingFilePathInfo(predictions, true_values, filepath, case):
    file = os.path.basename(filepath)
    dataset_name = getDatasetName(file)
    anio = anioOfDataset(file)
    anio_input = anioInputOfDataset(file)
    targets, denormalized_values_arr = calculateTargetsAndDenormalizedValues(dataset_name, anio, anio_input)
    
    indices = getMonthIndicesToCalculateMetrics(dataset_name, case)
    
    my_metrics = calculateMyMetricsFromDenormalizedValues(predictions, true_values, targets, denormalized_values_arr, anio, indices)
    return my_metrics

def getMetricsTitle(metric_name, metrics_titles_dict):
    if metric_name in metrics_titles_dict:
        return metrics_titles_dict[metric_name]
    else:
        return metric_name

def createMetricsPerEpochPlot(metrics_dict, metrics_titles_dict, filepath, case, mean_plots, case_in_title, metric_case):
    file = os.path.basename(filepath)
    experiment_name = keepExperimentNameOnly(file)    
    experiment_date = getDatetimeAsString()
    pdf_name = f"plot_metric_case_{metric_case}_{experiment_name}_{experiment_date}.pdf"
    plots_folderpath = createFolderInRepoPath("plots")
    pdf_filepath = os.path.join(plots_folderpath, pdf_name)
    
    # Check if the file exists
    if not os.path.isfile(pdf_filepath):
        # Create the file if it doesn't exist
        with open(pdf_filepath, 'wb') as f:
            f.write(b'')
    
    with PdfPages(pdf_filepath) as pdf:
        for k,v in metrics_dict.items():
            title = getMetricsTitle(k,metrics_titles_dict)
            if case_in_title == True:
                title += 'for ' + case
            if mean_plots == True and 'mean' in k:
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                colors_iter = iter(colors)
                std_metric_name = k.replace('mean', 'std')
                upper_list = [x + y for x, y in zip(v, metrics_dict[std_metric_name])]
                lower_list = [x - y for x, y in zip(v, metrics_dict[std_metric_name])]
                color = next(colors_iter)
                plt.plot(upper_list, color = color, label = 'Upper')
                plt.plot(lower_list, color = color, label = 'Lower ')
            else:
                plt.plot(v)
            plt.title(title)
            plt.xlabel("epochs")
            plt.ylabel("metric")
            if mean_plots == True and 'mean' in k:
                plt.legend(bbox_to_anchor=(1.05, 1))
                pdf.savefig(bbox_inches='tight')            
            pdf.savefig()
            plt.clf()
            #plt.show()
            #print("Name for case: ", k + ' for ' + case)
            #print("Metric: ", v)
            #print("Last value of metric: ", v[-1])
            
def createMetricsPerEpochComparisonPlot(file_metrics_dict, metrics_titles_dict, case, readable_components, mean_plots, case_in_title, metric_case):
    experiment_date = getDatetimeAsString()
    pdf_name = f"plot_comparison_metric_case_{metric_case}_{experiment_date}.pdf"
    plots_comparison_folderpath = createFolderInRepoPath("plots_comparison")
    pdf_filepath = os.path.join(plots_comparison_folderpath, pdf_name)
    
    # Check if the file exists
    if not os.path.isfile(pdf_filepath):
        # Create the file if it doesn't exist
        with open(pdf_filepath, 'wb') as f:
            f.write(b'')
    
    first_metrics_dict = next(iter(file_metrics_dict.values()))
    metrics_names = list(first_metrics_dict.keys())
    
    with PdfPages(pdf_filepath) as pdf:
        for metric_name in metrics_names:
            title = getMetricsTitle(metric_name,metrics_titles_dict)
            if case_in_title == True:
                title += 'for ' + case
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            colors_iter = iter(colors)
            for k,v in file_metrics_dict.items():
                experiment_name = keepExperimentNameOnly(k)
                readable_experiment_name = getReadableExperimentName(experiment_name, readable_components)
                if mean_plots == True and 'mean' in metric_name:
                    std_metric_name = metric_name.replace('mean', 'std')
                    upper_list = [x + y for x, y in zip(v[metric_name], v[std_metric_name])]
                    lower_list = [x - y for x, y in zip(v[metric_name], v[std_metric_name])]
                    color = next(colors_iter)
                    plt.plot(upper_list, color = color, label = 'Upper ' + readable_experiment_name)
                    plt.plot(lower_list, color = color, label = 'Lower ' + readable_experiment_name)
                else:
                    plt.plot(v[metric_name], label = readable_experiment_name)
                    # plt.plot(v1, color='red', label='line 1') # plot the first line in red
                    # plt.plot(v2, color='blue', label='line 2') # plot the second line in blue
            plt.title(title)
            plt.xlabel("epochs")
            plt.ylabel("metric")
            plt.legend(bbox_to_anchor=(1.05, 1))
            pdf.savefig(bbox_inches='tight')
            plt.clf()

metrics_names_arr = [
    'RSE_full', 'CORR_full', 'Corr_full', 'MAE_full', 'MSE_full', 'RMSE_full', 'MAPE_full', 'MSPE_full',
    'RSE', 'CORR', 'Corr', 'MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE',
    'RSE_red', 'CORR_red', 'Corr_red', 'MAE_red', 'MSE_red', 'RMSE_red', 'MAPE_red', 'MSPE_red',
    'my_mape', 'my_c1', 'my_c2', 'my_c3'
    ]

def getAllMetricsNames():
    return metrics_names_arr

def keepUniqueValuesInListWithOlder(list):
    new_list = []
    for x in list:
        if x not in new_list:
            new_list.append(x)

    return new_list

def getMetricsNames(given_metric_names = [], financial_metrics = False, ETTh_metrics = False,
                    other_metrics = False, my_mape_metric = False, my_C_metrics = False):
    metrics_names = []
    if financial_metrics == True:
        metrics_names += ['RSE_full', 'CORR_full', 'RSE', 'CORR', 'RSE_red', 'CORR_red']  
    if ETTh_metrics == True:
        metrics_names += ['MAE_full', 'MSE_full', 'RMSE_full', 'MAPE_full', 'MAE', 'MSE', 'RMSE', 'MAPE', 'MAE_red', 'MSE_red', 'RMSE_red', 'MAPE_red']
    if other_metrics == True:
        metrics_names += ['Corr_full', 'MSPE_full', 'Corr', 'MSPE', 'Corr_red', 'MSPE_red']
    if my_mape_metric:
        metrics_names += ['my_mape']
    if my_C_metrics:
        metrics_names += ['my_c1', 'my_c2', 'my_c3']
    if len(given_metric_names) > 0:
        metrics_names += given_metric_names
        
    if len(metrics_names) == 0:
        metrics_names += ['MAPE_full']
        
    metrics_names = keepUniqueValuesInListWithOlder(metrics_names)
        
    return metrics_names

def outputMetricsForETThAlgorithm(filepath, case, print_func = print, metrics_names = ['MAPE_full'],  metrics_titles_dict = {}, plot_graphs = False, mean_plots = False, case_in_title = False, metric_case = '-1'):
    data = pd.read_csv(filepath)
    
    metrics_dict = {}
    
    for metric_name in metrics_names:
        metrics_dict[metric_name] = data[str(case) + '_' + metric_name].values.tolist()
        if 'mean' in metric_name:
            std_metric = metric_name.replace('mean', 'std')
            metrics_dict[std_metric] = data[str(case) + '_' + std_metric].values.tolist()
    
    epochs = len(metrics_dict[metrics_names[0]])
    print_func("Epochs: {}".format(epochs))
    
    for metric_name in metrics_names:
        print_func("{}: {:.4f}".format(metric_name, metrics_dict[metric_name][-1]))
    
    if plot_graphs:
        createMetricsPerEpochPlot(metrics_dict, metrics_titles_dict, filepath, case, mean_plots, case_in_title, metric_case)
        
    if "my_mape" in metrics_names and "my_c1" in metrics_names and "my_c2" in metrics_names and "my_c3" in metrics_names:
        my_mape = metrics_dict['my_mape'][-1]
        my_c1 = metrics_dict['my_c1'][-1]
        my_c2 = metrics_dict['my_c2'][-1]
        my_c3 = metrics_dict['my_c3'][-1]
        testMetricsForETThAlgorithm(filepath, case, my_mape, my_c1, my_c2, my_c3)
    
    return metrics_dict
    
def outputMetricsForETThAlgorithmGeneral(filepath, case, print_func = print, 
                                         print_financial_metrics = True, print_ETTh_metrics = True, print_other_metrics = True, 
                                         print_my_mape_metric = True, print_my_C_metrics = True, metrics_titles_dict = {}, plot_graphs = False, mean_plots = False, case_in_title = False, metric_case = '-1'):
    metrics_names = getMetricsNames(given_metric_names = [], financial_metrics = print_financial_metrics, ETTh_metrics = print_ETTh_metrics,
                    other_metrics = print_other_metrics, my_mape_metric = print_my_mape_metric, my_C_metrics = print_my_C_metrics)
    
    metrics_dict = outputMetricsForETThAlgorithm(filepath, case, print_func = print_func, metrics_names = metrics_names,  metrics_titles_dict = metrics_titles_dict, plot_graphs = plot_graphs, mean_plots = mean_plots, case_in_title = case_in_title, metric_case = metric_case)
    
    return metrics_dict

#######################################Metrics calculations#############################################################
def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean()

def Corr(pred, true):
    sig_p = np.std(pred, axis=0)
    sig_g = np.std(true, axis=0)
    m_p = pred.mean(0)
    m_g = true.mean(0)
    ind = (sig_g != 0)
    corr = ((pred - m_p) * (true - m_g)).mean(0) / (sig_p * sig_g)
    corr = (corr[ind]).mean()
    return corr

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mae = MAE(pred, true)
    #print(mae)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    #corr1 = CORR(pred, true)
    corr = 0 #Corr(pred, true)
    return mae,mse,rmse,mape,mspe,corr

def calculate_metrics_core(pred, true):
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    corr1 = Corr(pred, true)
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return rse, corr, corr1, mae, mse, rmse, mape, mspe

def calculate_mean_and_std_of_metrics(pred, true, bucket):
    pred_collections = CreateListOfSubSelections(pred, bucket)
    true_collections = CreateListOfSubSelections(true, bucket)
    
    rse_arr = []
    corr_arr = []
    corr1_arr = []
    mae_arr = []
    mse_arr = []
    rmse_arr = []
    mape_arr = []
    mspe_arr = []
    
    assert(len(pred_collections) == len(true_collections))
    for i in range(len(pred_collections)):
        assert(pred_collections[i].shape == true_collections[i].shape)
        rse, corr, corr1, mae, mse, rmse, mape, mspe = calculate_metrics_core(pred_collections[i], true_collections[i])
        rse_arr.append(rse)
        corr_arr.append(corr)
        corr1_arr.append(corr1)
        mae_arr.append(mae)
        mse_arr.append(mse)
        rmse_arr.append(rmse)
        mape_arr.append(mape)
        mspe_arr.append(mspe)
    
    rse_mean = np.mean(rse_arr)
    rse_std = np.std(rse_arr)
    corr_mean = np.mean(corr_arr)
    corr_std = np.std(corr_arr)
    corr1_mean = np.mean(corr1_arr)
    corr1_std = np.std(corr1_arr)
    mae_mean = np.mean(mae_arr)
    mae_std = np.std(mae_arr)
    mse_mean = np.mean(mse_arr)
    mse_std = np.std(mse_arr)
    rmse_mean = np.mean(rmse_arr)
    rmse_std = np.std(rmse_arr)
    mape_mean = np.mean(mape_arr)
    mape_std = np.std(mape_arr)
    mspe_mean = np.mean(mspe_arr)
    mspe_std = np.std(mspe_arr)
    
    return rse_mean, rse_std, corr_mean, corr_std, corr1_mean, corr1_std, mae_mean, mae_std, mse_mean, mse_std, rmse_mean, rmse_std, mape_mean, mape_std, mspe_mean, mspe_std
    
def calculate_metrics(pred, true):
    rse, corr, corr1, mae, mse, rmse, mape, mspe = calculate_metrics_core(pred, true)
    rse_mean_5, rse_std_5, corr_mean_5, corr_std_5, corr1_mean_5, corr1_std_5, mae_mean_5, mae_std_5, mse_mean_5, mse_std_5, rmse_mean_5, rmse_std_5, mape_mean_5, mape_std_5, mspe_mean_5, mspe_std_5 = calculate_mean_and_std_of_metrics(pred, true, 5)
    rse_mean_10, rse_std_10, corr_mean_10, corr_std_10, corr1_mean_10, corr1_std_10, mae_mean_10, mae_std_10, mse_mean_10, mse_std_10, rmse_mean_10, rmse_std_10, mape_mean_10, mape_std_10, mspe_mean_10, mspe_std_10 = calculate_mean_and_std_of_metrics(pred, true, 10)
    rse_mean_20, rse_std_20, corr_mean_20, corr_std_20, corr1_mean_20, corr1_std_20, mae_mean_20, mae_std_20, mse_mean_20, mse_std_20, rmse_mean_20, rmse_std_20, mape_mean_20, mape_std_20, mspe_mean_20, mspe_std_20 = calculate_mean_and_std_of_metrics(pred, true, 20)
    
    RSE_metric = Metric(metric_name = 'RSE', metric = rse, metric_mean_5 = rse_mean_5, metric_std_5 = rse_std_5, metric_mean_10 = rse_mean_10, metric_std_10 = rse_std_10, metric_mean_20 = rse_mean_20, metric_std_20 = rse_std_20)
    CORR_metric = Metric(metric_name = 'CORR', metric = corr, metric_mean_5 = corr_mean_5, metric_std_5 = corr_std_5, metric_mean_10 = corr_mean_10, metric_std_10 = corr_std_10, metric_mean_20 = corr_mean_20, metric_std_20 = corr_std_20)
    Corr_metric = Metric(metric_name = 'Corr', metric = corr1, metric_mean_5 = corr1_mean_5, metric_std_5 = corr1_std_5, metric_mean_10 = corr1_mean_10, metric_std_10 = corr1_std_10, metric_mean_20 = corr1_mean_20, metric_std_20 = corr1_std_20)
    MAE_metric = Metric(metric_name = 'MAE', metric = mae, metric_mean_5 = mae_mean_5, metric_std_5 = mae_std_5, metric_mean_10 = mae_mean_10, metric_std_10 = mae_std_10, metric_mean_20 = mae_mean_20, metric_std_20 = mae_std_20)
    MSE_metric = Metric(metric_name = 'MSE', metric = mse, metric_mean_5 = mse_mean_5, metric_std_5 = mse_std_5, metric_mean_10 = mse_mean_10, metric_std_10 = mse_std_10, metric_mean_20 = mse_mean_20, metric_std_20 = mse_std_20)
    RMSE_metric = Metric(metric_name = 'RMSE', metric = rmse, metric_mean_5 = rmse_mean_5, metric_std_5 = rmse_std_5, metric_mean_10 = rmse_mean_10, metric_std_10 = rmse_std_10, metric_mean_20 = rmse_mean_20, metric_std_20 = rmse_std_20)
    MAPE_metric = Metric(metric_name = 'MAPE', metric = mape, metric_mean_5 = mape_mean_5, metric_std_5 = mape_std_5, metric_mean_10 = mape_mean_10, metric_std_10 = mape_std_10, metric_mean_20 = mape_mean_20, metric_std_20 = mape_std_20)
    MSPE_metric = Metric(metric_name = 'MSPE', metric = mspe, metric_mean_5 = mspe_mean_5, metric_std_5 = mspe_std_5, metric_mean_10 = mspe_mean_10, metric_std_10 = mspe_std_10, metric_mean_20 = mspe_mean_20, metric_std_20 = mspe_std_20)
    
    metrics = Metrics(RSE_metric, CORR_metric, Corr_metric, MAE_metric, MSE_metric, RMSE_metric, MAPE_metric, MSPE_metric)
    return metrics

def denormalizePredictionsAndTrueValues2DArrays(predictions_arr, true_values_arr, targets, denormalized_values_arr, anio):
    assert(predictions_arr.shape == true_values_arr.shape)
    for i in range(0, predictions_arr.shape[0]):
        predictions_arr[i,:] , true_values_arr[i,:] = denormalizePredictionsAndTrueValues(predictions_arr[i,:], true_values_arr[i,:], targets, denormalized_values_arr, anio)
    
    return predictions_arr, true_values_arr

def calculateAlgorithmMetrics(pred_scales_full, true_scales_full, mid_scales_full, correct_predictions, correct_true, correct_mid_predictions, filepath):
    file = os.path.basename(filepath)
    dataset_name = getDatasetName(file)
    anio = anioOfDataset(file)
    anio_input = anioInputOfDataset(file)
    targets, denormalized_values_arr = calculateTargetsAndDenormalizedValues(dataset_name, anio, anio_input)
    
    metrics_full = calculate_metrics(pred_scales_full, true_scales_full)
    pred_scales = pred_scales_full[:,:,-1].copy()
    true_scales = true_scales_full[:,:,-1].copy()
    pred_scales, true_scales = denormalizePredictionsAndTrueValues2DArrays(pred_scales, true_scales, targets, denormalized_values_arr, anio)
    metrics = calculate_metrics(pred_scales, true_scales)
    if mid_scales_full is not None:
        mid_metrics_full = calculate_metrics(mid_scales_full, true_scales_full)
        mid_scales = mid_scales_full[:,:,-1].copy()
        true_scales = true_scales_full[:,:,-1].copy()
        mid_scales, true_scales = denormalizePredictionsAndTrueValues2DArrays(mid_scales, true_scales, targets, denormalized_values_arr, anio)
        mid_metrics = calculate_metrics(mid_scales, true_scales)
    else:
        mid_metrics_full = Metrics()
        mid_metrics = Metrics()
    algorithm_metrics_full = AlgorithmMetrics(metrics = metrics_full, mid_metrics = mid_metrics_full)
    algorithm_metrics = AlgorithmMetrics(metrics = metrics, mid_metrics = mid_metrics)
    
    correct_predictions_copy = correct_predictions.copy()
    correct_true_copy = correct_true.copy()
    correct_predictions_copy , correct_true_copy = denormalizePredictionsAndTrueValues(correct_predictions_copy, correct_true_copy, targets, denormalized_values_arr, anio)
    metrics_red = calculate_metrics(np.array(correct_predictions_copy), np.array(correct_true_copy))
    if correct_mid_predictions is not None:
        correct_mid_predictions_copy = correct_mid_predictions.copy()
        correct_true_copy = correct_true.copy()
        correct_mid_predictions_copy , correct_true_copy = denormalizePredictionsAndTrueValues(correct_mid_predictions_copy, correct_true_copy, targets, denormalized_values_arr, anio)
        mid_metrics_red = calculate_metrics(np.array(correct_mid_predictions_copy), np.array(correct_true_copy))
    else:
        mid_metrics_red = Metrics()
    algorithm_metrics_red = AlgorithmMetrics(metrics = metrics_red, mid_metrics = mid_metrics_red)
    
    return algorithm_metrics_full, algorithm_metrics, algorithm_metrics_red

#############################################Greek dataset - ETTh algorithm###################################################################

def getOriginalGreekCsvFile():
    csv_name = os.path.join(getRepoPath(),"scripts","Multi_cities_Deh_Dataset.csv")
    return csv_name

# For anio1, anio2, anio3
def createGreekDatasetWithGeneralAnioColumnsForAnio123(anio):
    dataset_csv = os.path.join(getRepoPath(),"datasets","ETT-data",f"greek_energy_{anio}.csv")
    if (os.path.exists(dataset_csv)):
        return dataset_csv

    csv_name = getOriginalGreekCsvFile()
    data = pd.read_csv(csv_name, nrows= 72336)
    dates = pd.date_range(start='10-01-2010', end = '12-31-2018 23:00:00', freq='H')
    total_cons = data['TOTAL_CONS'].values.tolist()
    athens_temp = data['Athens_temp'].values.tolist()
    thess_temp = data['Thessaloniki_temp'].values.tolist()
    df = pd.DataFrame({'date': dates, 'Athens_temp': athens_temp, 'Thessaloniki_temp': thess_temp, 'target':total_cons})

    if anio == 'anio1':
        maximum_past_hours = 28 * 24
    elif anio == 'anio2':
        maximum_past_hours = 35 * 24
    elif anio == 'anio3':
        maximum_past_hours = 28 * 24
    else:
        raise Exception("anio value was invalid")

    Thessaloniki_temp_yesterday = []
    Thessaloniki_temp_week = []
    Thessaloniki_temp_month = []
    Athens_temp_yesterday = []
    Athens_temp_week = []
    Athens_temp_month = []
    anio_load_yesterday = []
    anio_load_week = []
    anio_load_month = []
    load_yesterday = []
    load_week = []
    load_month = []
    new_target = []
    denormalized_values = []


    for i in range(maximum_past_hours,len(df)):
        Thessaloniki_temp_yesterday.append(df.loc[i - 24, 'Thessaloniki_temp'])
        Thessaloniki_temp_week.append(df.loc[i - (24 * 7), 'Thessaloniki_temp'])
        Thessaloniki_temp_month.append(df.loc[i - (24 * 28), 'Thessaloniki_temp'])
        Athens_temp_yesterday.append(df.loc[i - 24, 'Athens_temp'])
        Athens_temp_week.append(df.loc[i - (24 * 7), 'Athens_temp'])
        Athens_temp_month.append(df.loc[i - (24 * 28), 'Athens_temp'])
        anio_load_yesterday.append(df.loc[i - 24, 'target'] / df.loc[i - (24 * 7), 'target'] - 1)
        anio_load_week.append(df.loc[i - (24 * 7), 'target'] / df.loc[i - (24 * 14), 'target'] - 1)
        load_yesterday.append(df.loc[i - 24, 'target'])
        load_week.append(df.loc[i - (24 * 7), 'target'])
        load_month.append(df.loc[i - (24 * 28), 'target'])
        if anio == 'anio2':
            anio_load_month.append(df.loc[i - (24 * 28), 'target'] / df.loc[i - (24 * 35), 'target'] - 1)
        else:
            anio_load_month.append(df.loc[i - (24 * 28), 'target'] / df.loc[i - (24 * 7), 'target'] - 1)        
        new_target.append(df.loc[i, 'target'] / df.loc[i - (24 * 7) , 'target'] - 1)
        denormalized_values.append(df.loc[i - (24 * 7) , 'target'])


    df = df[maximum_past_hours:]
    df['new_target'] = new_target
    df['denormalized_values'] = denormalized_values
    df['Thessaloniki_temp_yesterday'] = Thessaloniki_temp_yesterday
    df['Thessaloniki_temp_week'] = Thessaloniki_temp_week
    df['Thessaloniki_temp_month'] = Thessaloniki_temp_month
    df['Athens_temp_yesterday'] = Athens_temp_yesterday
    df['Athens_temp_week'] = Athens_temp_week
    df['Athens_temp_month'] = Athens_temp_month
    df['anio_load_yesterday'] = anio_load_yesterday
    df['anio_load_week'] = anio_load_week
    df['anio_load_month'] = anio_load_month
    if anio == 'anio1':
        df['load_yesterday'] = load_yesterday
        df['load_week'] = load_week
        df['load_month'] = load_month

    df.to_csv(dataset_csv, index = False, header = True)
    
    return dataset_csv

# For anio4
def createGreekDatasetWithGeneralAnioColumnsForAnio4(anio):
    dataset_csv = os.path.join(getRepoPath(),"datasets","ETT-data",f"greek_energy_{anio}.csv")
    if (os.path.exists(dataset_csv)):
        return dataset_csv

    csv_name = getOriginalGreekCsvFile()
    data = pd.read_csv(csv_name, nrows= 72336)
    dates = pd.date_range(start='10-01-2010', end = '12-31-2018 23:00:00', freq='H')
    total_cons = data['TOTAL_CONS'].values.tolist()
    athens_temp = data['Athens_temp'].values.tolist()
    thess_temp = data['Thessaloniki_temp'].values.tolist()
    df = pd.DataFrame({'date': dates, 'Athens_temp': athens_temp, 'Thessaloniki_temp': thess_temp, 'target':total_cons})

    if anio == 'anio4':
        maximum_past_hours = 2 * 24
    else:
        raise Exception("anio value was invalid")

    Thessaloniki_temp_yesterday = []
    Athens_temp_yesterday = []
    anio_load_yesterday = []
    new_target = []
    denormalized_values = []

    for i in range(maximum_past_hours,len(df)):
        Thessaloniki_temp_yesterday.append(df.loc[i - 24, 'Thessaloniki_temp'])
        Athens_temp_yesterday.append(df.loc[i - 24, 'Athens_temp'])
        anio_load_yesterday.append(df.loc[i - 24, 'target'] / df.loc[i - (24 * 2), 'target'] - 1)     
        new_target.append(df.loc[i, 'target'] / df.loc[i - (24 * 2) , 'target'] - 1)
        denormalized_values.append(df.loc[i - (24 * 2) , 'target'])

    df = df[maximum_past_hours:]
    df['new_target'] = new_target
    df['denormalized_values'] = denormalized_values
    df['Thessaloniki_temp_yesterday'] = Thessaloniki_temp_yesterday
    df['Athens_temp_yesterday'] = Athens_temp_yesterday
    df['anio_load_yesterday'] = anio_load_yesterday 

    df.to_csv(dataset_csv, index = False, header = True)
    
    return dataset_csv

# For anio5
def createGreekDatasetWithGeneralAnioColumnsForAnio5(anio):
    dataset_csv = os.path.join(getRepoPath(),"datasets","ETT-data",f"greek_energy_{anio}.csv")
    if (os.path.exists(dataset_csv)):
        return dataset_csv

    csv_name = getOriginalGreekCsvFile()
    data = pd.read_csv(csv_name, nrows= 72336)
    dates = pd.date_range(start='10-01-2010', end = '12-31-2018 23:00:00', freq='H')
    total_cons = data['TOTAL_CONS'].values.tolist()
    athens_temp = data['Athens_temp'].values.tolist()
    thess_temp = data['Thessaloniki_temp'].values.tolist()
    df = pd.DataFrame({'date': dates, 'Athens_temp': athens_temp, 'Thessaloniki_temp': thess_temp, 'target':total_cons})

    if anio == 'anio5':
        maximum_past_hours = 14 * 24
    else:
        raise Exception("anio value was invalid")

    Thessaloniki_temp_yesterday = []
    Thessaloniki_temp_week = []
    Athens_temp_yesterday = []
    Athens_temp_week = []
    anio_load_yesterday = []
    anio_load_week = []
    new_target = []
    denormalized_values = []


    for i in range(maximum_past_hours,len(df)):
        Thessaloniki_temp_yesterday.append(df.loc[i - 24, 'Thessaloniki_temp'])
        Thessaloniki_temp_week.append(df.loc[i - (24 * 7), 'Thessaloniki_temp'])
        Athens_temp_yesterday.append(df.loc[i - 24, 'Athens_temp'])
        Athens_temp_week.append(df.loc[i - (24 * 7), 'Athens_temp'])
        anio_load_yesterday.append(df.loc[i - 24, 'target'] / df.loc[i - (24 * 7), 'target'] - 1)
        anio_load_week.append(df.loc[i - (24 * 7), 'target'] / df.loc[i - (24 * 14), 'target'] - 1)       
        new_target.append(df.loc[i, 'target'] / df.loc[i - (24 * 7) , 'target'] - 1)
        denormalized_values.append(df.loc[i - (24 * 7) , 'target'])

    df = df[maximum_past_hours:]
    df['new_target'] = new_target
    df['denormalized_values'] = denormalized_values
    df['Thessaloniki_temp_yesterday'] = Thessaloniki_temp_yesterday
    df['Thessaloniki_temp_week'] = Thessaloniki_temp_week
    df['Athens_temp_yesterday'] = Athens_temp_yesterday
    df['Athens_temp_week'] = Athens_temp_week
    df['anio_load_yesterday'] = anio_load_yesterday
    df['anio_load_week'] = anio_load_week

    df.to_csv(dataset_csv, index = False, header = True)
    
    return dataset_csv

def createGreekDatasetWithGeneralAnioColumns(anio):
    if anio in ['anio1', 'anio2', 'anio3']:
        dataset_csv = createGreekDatasetWithGeneralAnioColumnsForAnio123(anio)
    elif anio == 'anio4':
        dataset_csv = createGreekDatasetWithGeneralAnioColumnsForAnio4(anio)
    elif anio == 'anio5':
        dataset_csv = createGreekDatasetWithGeneralAnioColumnsForAnio5(anio)
    else:
        raise Exception("anio value was invalid")
        
    return dataset_csv 

def createGreekEnergyDataset(anio = 'no_anio', anio_input = 'Yes'):
    features = getGreekEnergyFeatures(anio, anio_input)
    target = getGreekEnergyTarget(anio)
    filename = getGreekEnergyCSVName(anio, anio_input)
    
    targets = []
    denormalized_values = []
    
    dataset_csv = os.path.join(getRepoPath(),"datasets","ETT-data",filename)
    if os.path.exists(dataset_csv):
        return targets, denormalized_values
    
    csv_name = getOriginalGreekCsvFile()
    data = pd.read_csv(csv_name, nrows= 72336)
    dates = pd.date_range(start='10-01-2010', end = '12-31-2018 23:00:00', freq='H')

    dict = {}

    dict['date'] = dates
    for feature in features:
        dict[feature] = data[feature].values.tolist()

    dict['target'] = data[target].values.tolist()
    df = pd.DataFrame(dict)
    df.to_csv(dataset_csv, index = False, header = True)
    
    return targets, denormalized_values

    
def createGreekDatasetAnio(anio, anio_input):
    features = getGreekEnergyFeatures(anio, anio_input)
    target = getGreekEnergyTarget(anio)
    output_filename = getGreekEnergyCSVName(anio, anio_input)
    
    csv_name = createGreekDatasetWithGeneralAnioColumns(anio)
    data = pd.read_csv(csv_name)
    dict = {}
    denormalized_values = data['denormalized_values'].values.tolist()
    targets = data[target].values.tolist()
    
    dataset_csv = os.path.join(getRepoPath(),"datasets","ETT-data",output_filename)
    if os.path.exists(dataset_csv):
        return targets, denormalized_values 
    
    dict['date'] = data['date'].values.tolist()
    for feature in features:
        dict[feature] = data[feature].values.tolist()

    dict['target'] = data[target].values.tolist()
    df = pd.DataFrame(dict)
    df.to_csv(dataset_csv, index = False, header = True)
    
    return targets, denormalized_values

def createGreekDatasetGeneral(anio, anio_input):
    if anio == 'no_anio':
        targets, denormalized_values = createGreekEnergyDataset(anio, anio_input)
    else:
        targets, denormalized_values = createGreekDatasetAnio(anio, anio_input)

    return targets, denormalized_values


#############################################ETTh datasets - ETTh algorithm###################################################################

def getOriginalETThDatasetCsvFile(dataset_name):
    csv_name = os.path.join(getRepoPath(),"datasets","ETT-data","ETT",f"{dataset_name}.csv")
    return csv_name

# For anio1, anio2, anio3
def createETThDatasetWithGeneralAnioColumnsForAnio123(dataset_name, anio):
    dataset_csv = os.path.join(getRepoPath(),"datasets","ETT-data",f"{dataset_name}_{anio}.csv")
    if (os.path.exists(dataset_csv)):
        return dataset_csv

    csv_name = getOriginalETThDatasetCsvFile(dataset_name)
    data = pd.read_csv(csv_name)
    dates = pd.date_range(start='2016-07-01', end = '2018-06-26 19:00:00', freq='H')
    OT = data['OT'].values.tolist()
    HUFL = data['HUFL'].values.tolist()
    HULL = data['HULL'].values.tolist()
    MUFL = data['MUFL'].values.tolist()
    MULL = data['MULL'].values.tolist()
    LUFL = data['LUFL'].values.tolist()
    LULL = data['LULL'].values.tolist()
    OT = [0.1 if abs(x) < 0.1 else x for x in OT]
    df = pd.DataFrame({'date': dates, 'HUFL': HUFL, 'HULL': HULL, 'MUFL': MUFL, 'MULL': MULL, 'LUFL': LUFL, 'LULL': LULL, 'target':OT})

    if anio == 'anio1':
        maximum_past_hours = 28 * 24
    elif anio == 'anio2':
        maximum_past_hours = 35 * 24
    elif anio == 'anio3':
        maximum_past_hours = 28 * 24
    else:
        raise Exception("anio value was invalid")

    HUFL_yesterday = []
    HUFL_week = []
    HUFL_month = []
    HULL_yesterday = []
    HULL_week = []
    HULL_month = []
    MUFL_yesterday = []
    MUFL_week = []
    MUFL_month = []
    MULL_yesterday = []
    MULL_week = []
    MULL_month = []
    LUFL_yesterday = []
    LUFL_week = []
    LUFL_month = []
    LULL_yesterday = []
    LULL_week = []
    LULL_month = []
    anio_OT_yesterday = []
    anio_OT_week = []
    anio_OT_month = []
    OT_yesterday = []
    OT_week = []
    OT_month = []
    new_target = []
    denormalized_values = []

    for i in range(maximum_past_hours,len(df)):
        HUFL_yesterday.append(df.loc[i - 24, 'HUFL'])
        HUFL_week.append(df.loc[i - (24 * 7), 'HUFL'])
        HUFL_month.append(df.loc[i - (24 * 28), 'HUFL'])
        HULL_yesterday.append(df.loc[i - 24, 'HULL'])
        HULL_week.append(df.loc[i - (24 * 7), 'HULL'])
        HULL_month.append(df.loc[i - (24 * 28), 'HULL'])
        MUFL_yesterday.append(df.loc[i - 24, 'MUFL'])
        MUFL_week.append(df.loc[i - (24 * 7), 'MUFL'])
        MUFL_month.append(df.loc[i - (24 * 28), 'MUFL'])
        MULL_yesterday.append(df.loc[i - 24, 'MULL'])
        MULL_week.append(df.loc[i - (24 * 7), 'MULL'])
        MULL_month.append(df.loc[i - (24 * 28), 'MULL'])
        LUFL_yesterday.append(df.loc[i - 24, 'LUFL'])
        LUFL_week.append(df.loc[i - (24 * 7), 'LUFL'])
        LUFL_month.append(df.loc[i - (24 * 28), 'LUFL'])
        LULL_yesterday.append(df.loc[i - 24, 'LULL'])
        LULL_week.append(df.loc[i - (24 * 7), 'LULL'])
        LULL_month.append(df.loc[i - (24 * 28), 'LULL'])
        anio_OT_yesterday.append(df.loc[i - 24, 'target'] / df.loc[i - (24 * 7), 'target'] - 1)
        anio_OT_week.append(df.loc[i - (24 * 7), 'target'] / df.loc[i - (24 * 14), 'target'] - 1)
        OT_yesterday.append(df.loc[i - 24, 'target'])
        OT_week.append(df.loc[i - (24 * 7), 'target'])
        OT_month.append(df.loc[i - (24 * 28), 'target'])
        if anio == 'anio2':
            anio_OT_month.append(df.loc[i - (24 * 28), 'target'] / df.loc[i - (24 * 35), 'target'] - 1)
        else:
            anio_OT_month.append(df.loc[i - (24 * 28), 'target'] / df.loc[i - (24 * 7), 'target'] - 1)
        new_target.append(df.loc[i, 'target'] / df.loc[i - (24 * 7) , 'target'] - 1)
        denormalized_values.append(df.loc[i - (24 * 7) , 'target'])
        

    df = df[maximum_past_hours:]
    df['new_target'] = new_target
    df['denormalized_values'] = denormalized_values
    df['HUFL_yesterday'] = HUFL_yesterday
    df['HUFL_week'] = HUFL_week
    df['HUFL_month'] = HUFL_month
    df['HULL_yesterday'] = HULL_yesterday
    df['HULL_week'] = HULL_week
    df['HULL_month'] = HULL_month
    df['MUFL_yesterday'] = MUFL_yesterday
    df['MUFL_week'] = MUFL_week
    df['MUFL_month'] = MUFL_month
    df['MULL_yesterday'] = MULL_yesterday
    df['MULL_week'] = MULL_week
    df['MULL_month'] = MULL_month
    df['LUFL_yesterday'] = LUFL_yesterday
    df['LUFL_week'] = LUFL_week
    df['LUFL_month'] = LUFL_month
    df['LULL_yesterday'] = LULL_yesterday
    df['LULL_week'] = LULL_week
    df['LULL_month'] = LULL_month  
    df['anio_OT_yesterday'] = anio_OT_yesterday
    df['anio_OT_week'] = anio_OT_week
    df['anio_OT_month'] = anio_OT_month
    if anio == 'anio1':
        df['OT_yesterday'] = OT_yesterday
        df['OT_week'] = OT_week
        df['OT_month'] = OT_month

    df.to_csv(dataset_csv, index = False, header = True)
    
    return dataset_csv

# For anio4
def createETThDatasetWithGeneralAnioColumnsForAnio4(dataset_name, anio):
    dataset_csv = os.path.join(getRepoPath(),"datasets","ETT-data",f"{dataset_name}_{anio}.csv")
    if (os.path.exists(dataset_csv)):
        return dataset_csv

    csv_name = getOriginalETThDatasetCsvFile(dataset_name)
    data = pd.read_csv(csv_name)
    dates = pd.date_range(start='2016-07-01', end = '2018-06-26 19:00:00', freq='H')
    OT = data['OT'].values.tolist()
    HUFL = data['HUFL'].values.tolist()
    HULL = data['HULL'].values.tolist()
    MUFL = data['MUFL'].values.tolist()
    MULL = data['MULL'].values.tolist()
    LUFL = data['LUFL'].values.tolist()
    LULL = data['LULL'].values.tolist()
    OT = [0.1 if abs(x) < 0.1 else x for x in OT]
    df = pd.DataFrame({'date': dates, 'HUFL': HUFL, 'HULL': HULL, 'MUFL': MUFL, 'MULL': MULL, 'LUFL': LUFL, 'LULL': LULL, 'target':OT})

    if anio == 'anio4':
        maximum_past_hours = 2 * 24
    else:
        raise Exception("anio value was invalid")

    HUFL_yesterday = []
    HULL_yesterday = []
    MUFL_yesterday = []
    MULL_yesterday = []
    LUFL_yesterday = []
    LULL_yesterday = []
    anio_OT_yesterday = []
    new_target = []
    denormalized_values = []

    for i in range(maximum_past_hours,len(df)):
        HUFL_yesterday.append(df.loc[i - 24, 'HUFL'])
        HULL_yesterday.append(df.loc[i - 24, 'HULL'])
        MUFL_yesterday.append(df.loc[i - 24, 'MUFL'])
        MULL_yesterday.append(df.loc[i - 24, 'MULL'])
        LUFL_yesterday.append(df.loc[i - 24, 'LUFL'])
        LULL_yesterday.append(df.loc[i - 24, 'LULL'])
        anio_OT_yesterday.append(df.loc[i - 24, 'target'] / df.loc[i - (24 * 2), 'target'] - 1)
        new_target.append(df.loc[i, 'target'] / df.loc[i - (24 * 2) , 'target'] - 1)
        denormalized_values.append(df.loc[i - (24 * 2) , 'target'])

    df = df[maximum_past_hours:]
    df['new_target'] = new_target
    df['denormalized_values'] = denormalized_values
    df['HUFL_yesterday'] = HUFL_yesterday
    df['HULL_yesterday'] = HULL_yesterday
    df['MUFL_yesterday'] = MUFL_yesterday
    df['MULL_yesterday'] = MULL_yesterday
    df['LUFL_yesterday'] = LUFL_yesterday
    df['LULL_yesterday'] = LULL_yesterday
    df['anio_OT_yesterday'] = anio_OT_yesterday

    df.to_csv(dataset_csv, index = False, header = True)
    
    return dataset_csv

# For anio5
def createETThDatasetWithGeneralAnioColumnsForAnio5(dataset_name, anio):
    dataset_csv = os.path.join(getRepoPath(),"datasets","ETT-data",f"{dataset_name}_{anio}.csv")
    if (os.path.exists(dataset_csv)):
        return dataset_csv

    csv_name = getOriginalETThDatasetCsvFile(dataset_name)
    data = pd.read_csv(csv_name)
    dates = pd.date_range(start='2016-07-01', end = '2018-06-26 19:00:00', freq='H')
    OT = data['OT'].values.tolist()
    HUFL = data['HUFL'].values.tolist()
    HULL = data['HULL'].values.tolist()
    MUFL = data['MUFL'].values.tolist()
    MULL = data['MULL'].values.tolist()
    LUFL = data['LUFL'].values.tolist()
    LULL = data['LULL'].values.tolist()
    OT = [0.1 if abs(x) < 0.1 else x for x in OT]
    df = pd.DataFrame({'date': dates, 'HUFL': HUFL, 'HULL': HULL, 'MUFL': MUFL, 'MULL': MULL, 'LUFL': LUFL, 'LULL': LULL, 'target':OT})

    if anio == 'anio5':
        maximum_past_hours = 14 * 24
    else:
        raise Exception("anio value was invalid")

    HUFL_yesterday = []
    HUFL_week = []
    HULL_yesterday = []
    HULL_week = []
    MUFL_yesterday = []
    MUFL_week = []
    MULL_yesterday = []
    MULL_week = []
    LUFL_yesterday = []
    LUFL_week = []
    LULL_yesterday = []
    LULL_week = []
    anio_OT_yesterday = []
    anio_OT_week = []
    new_target = []
    denormalized_values = []

    for i in range(maximum_past_hours,len(df)):
        HUFL_yesterday.append(df.loc[i - 24, 'HUFL'])
        HUFL_week.append(df.loc[i - (24 * 7), 'HUFL'])
        HULL_yesterday.append(df.loc[i - 24, 'HULL'])
        HULL_week.append(df.loc[i - (24 * 7), 'HULL'])
        MUFL_yesterday.append(df.loc[i - 24, 'MUFL'])
        MUFL_week.append(df.loc[i - (24 * 7), 'MUFL'])
        MULL_yesterday.append(df.loc[i - 24, 'MULL'])
        MULL_week.append(df.loc[i - (24 * 7), 'MULL'])
        LUFL_yesterday.append(df.loc[i - 24, 'LUFL'])
        LUFL_week.append(df.loc[i - (24 * 7), 'LUFL'])
        LULL_yesterday.append(df.loc[i - 24, 'LULL'])
        LULL_week.append(df.loc[i - (24 * 7), 'LULL'])
        anio_OT_yesterday.append(df.loc[i - 24, 'target'] / df.loc[i - (24 * 7), 'target'] - 1)
        anio_OT_week.append(df.loc[i - (24 * 7), 'target'] / df.loc[i - (24 * 14), 'target'] - 1)
        new_target.append(df.loc[i, 'target'] / df.loc[i - (24 * 7) , 'target'] - 1)
        denormalized_values.append(df.loc[i - (24 * 7) , 'target'])
        
    df = df[maximum_past_hours:]
    df['new_target'] = new_target
    df['denormalized_values'] = denormalized_values
    df['HUFL_yesterday'] = HUFL_yesterday
    df['HUFL_week'] = HUFL_week
    df['HULL_yesterday'] = HULL_yesterday
    df['HULL_week'] = HULL_week
    df['MUFL_yesterday'] = MUFL_yesterday
    df['MUFL_week'] = MUFL_week
    df['MULL_yesterday'] = MULL_yesterday
    df['MULL_week'] = MULL_week
    df['LUFL_yesterday'] = LUFL_yesterday
    df['LUFL_week'] = LUFL_week
    df['LULL_yesterday'] = LULL_yesterday
    df['LULL_week'] = LULL_week
    df['anio_OT_yesterday'] = anio_OT_yesterday
    df['anio_OT_week'] = anio_OT_week

    df.to_csv(dataset_csv, index = False, header = True)
    
    return dataset_csv

def createETThDatasetWithGeneralAnioColumns(dataset_name, anio):
    if anio in ['anio1', 'anio2', 'anio3']:
        dataset_csv = createETThDatasetWithGeneralAnioColumnsForAnio123(dataset_name, anio)
    elif anio == 'anio4':
        dataset_csv = createETThDatasetWithGeneralAnioColumnsForAnio4(dataset_name, anio)
    elif anio == 'anio5':
        dataset_csv = createETThDatasetWithGeneralAnioColumnsForAnio5(dataset_name, anio)
    else:
        raise Exception("anio value was invalid")
        
    return dataset_csv

def createETThDataset(dataset_name = 'ETTh1', anio = 'no_anio', anio_input = 'Yes'):
    features = getETThDatasetFeatures(anio, anio_input)
    target = getETThDatasetTarget(anio)
    filename = getETThDatasetCSVName(dataset_name, anio, anio_input)
    
    targets = []
    denormalized_values = []
    
    dataset_csv = os.path.join(getRepoPath(),"datasets","ETT-data",filename)
    if os.path.exists(dataset_csv):
        return targets, denormalized_values
    
    csv_name = getOriginalETThDatasetCsvFile(dataset_name)
    data = pd.read_csv(csv_name)
    dates = pd.date_range(start='2016-07-01', end = '2018-06-26 19:00:00', freq='H')

    dict = {}

    dict['date'] = dates
    for feature in features:
        dict[feature] = data[feature].values.tolist()

    dict['target'] = data[target].values.tolist()
    df = pd.DataFrame(dict)
    df.to_csv(dataset_csv, index = False, header = True)
    
    return targets, denormalized_values

def createETThDatasetAnio(dataset_name, anio, anio_input):
    features = getETThDatasetFeatures(anio, anio_input)
    target = getETThDatasetTarget(anio)
    output_filename = getETThDatasetCSVName(dataset_name, anio, anio_input)
    
    csv_name = createETThDatasetWithGeneralAnioColumns(dataset_name, anio)
    data = pd.read_csv(csv_name)
    dict = {}
    denormalized_values = data['denormalized_values'].values.tolist()
    targets = data[target].values.tolist()
    
    dataset_csv = os.path.join(getRepoPath(),"datasets","ETT-data",output_filename)
    if os.path.exists(dataset_csv):
        return targets, denormalized_values 
    
    dict['date'] = data['date'].values.tolist()
    for feature in features:
        dict[feature] = data[feature].values.tolist()

    dict['target'] = data[target].values.tolist()
    df = pd.DataFrame(dict)
    df.to_csv(dataset_csv, index = False, header = True)
    
    return targets, denormalized_values  

def createETThDatasetGeneral(dataset_name, anio, anio_input):
    if anio == 'no_anio':
        targets, denormalized_values = createETThDataset(dataset_name, anio, anio_input)
    else:
        targets, denormalized_values = createETThDatasetAnio(dataset_name, anio, anio_input)

    return targets, denormalized_values

#######################################Greek Scinet dataset - Financial algorithm#############################################################   

greek_scinet_dataset_names = [
    "greek_scinet_dataset_load_load",
    "greek_scinet_dataset_load_athens",
    "greek_scinet_dataset_load_athens_thess"
]

greek_scinet_features = {
    "load_load": ['TOTAL_CONS', 'TOTAL_CONS'], 
    "load_athens": ['TOTAL_CONS', 'Athens_temp'],
    "load_athens_thess": ['TOTAL_CONS', 'Athens_temp','Thessaloniki_temp']
    }

def getGreekScinetDatasetNames():
    return greek_scinet_dataset_names

def getGreekScinetFeatures(dataset):
    substring = "greek_scinet_dataset_"
    option = dataset.split(substring,1)[1]
    
    features = greek_scinet_features[option]
    
    return features

def getGreekScinetInputDim(dataset):
    features = getGreekScinetFeatures(dataset)
    
    return len(features)

def createGreekScinetDataset(dataset):
    csv_name = getOriginalGreekCsvFile()
    data = pd.read_csv(csv_name, nrows= 72336)
    dict = {}
    
    features = getGreekScinetFeatures(dataset)

    for i, feature in enumerate(features):
        dict["column"+str(i)] = data[feature].values.tolist()

    df = pd.DataFrame(dict)
    dataset_csv = os.path.join(getRepoPath(),"datasets","financial",dataset + ".txt")
    df.to_csv(dataset_csv, index = False, header = False)

def createGreekScinetDatasetWithCheck(dataset):
    if (os.path.exists(os.path.join(getRepoPath(),"datasets","financial",dataset + ".txt"))):
        return
    
    createGreekScinetDataset(dataset)
    
