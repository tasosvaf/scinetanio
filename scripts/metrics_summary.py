import argparse
from functools import partial
import os
import sys
from prettytable import PrettyTable

#Adding path to libray
dirpath = os.path.dirname(__file__)
parent_dirpath, _ = os.path.split(dirpath)
sys.path.append(parent_dirpath)

from SCINet.utils.utils_ETTh import checkReadableComponents, createFolderInRepoPath, createMetricsPerEpochComparisonPlot, getDatetimeAsString, getLatestMetricsFilepathsThatBelongToGivenFilters
from SCINet.utils.utils_ETTh import getReadableExperimentName, keepExperimentNameOnly, outputMetricsForETThAlgorithm, print_output_in_specific_file, getMetricsNames, getMetricsTitle
from SCINet.utils.utils_ETTh import metric_filepath_sort

'''
Options:
all_readable_components = ['se', 'la', 'pr', 'lr', 'le', 'dp', 'st', 'ft', 'sh', 'de', 'da', 'ep', 'uk', 'an']
metrics_names_arr = [
    'RSE_full', 'CORR_full', 'Corr_full', 'MAE_full', 'MSE_full', 'RMSE_full', 'MAPE_full', 'MSPE_full',
    'RSE', 'CORR', 'Corr', 'MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE',
    'RSE_red', 'CORR_red', 'Corr_red', 'MAE_red', 'MSE_red', 'RMSE_red', 'MAPE_red', 'MSPE_red',
    'my_mape', 'my_c1', 'my_c2', 'my_c3'
'''

def getMeanAndStdTableString(mean, std, mean_decimal_places = 4, std_decimal_places = 4):
    rounded_mean = round(mean, mean_decimal_places)
    rounded_std = round(std, std_decimal_places)
    mean_std_table_value = f"{rounded_mean:.{mean_decimal_places}f} +/- {rounded_std:.{std_decimal_places}f}"
    return mean_std_table_value

def generateMetricsOutputFiles(readable_components = ['se', 'pr', 'da', 'an'], given_experiment_names = [], and_filters = [], or_filters = [], 
                             given_metrics_names = ['MAPE_full'], metrics_titles_dict = {}, financial_metrics = False, ETTh_metrics = False, 
                             other_metrics = False, my_mape_metric = False, my_C_metrics = False, mean_plots = False, case_in_title = False,
                             mean_decimal_places = 4, std_decimal_places = 4, metric_case = '-1'):
    filepaths = getLatestMetricsFilepathsThatBelongToGivenFilters(given_experiment_names = given_experiment_names, 
                                                                  and_filters = and_filters, or_filters = or_filters)
    
    if len(filepaths) == 0:
        print("No files found")
        return
    
    filepaths = sorted(filepaths, key = lambda filepath: metric_filepath_sort(filepath, readable_components))
    
    checkReadableComponents(readable_components)
    
    metrics_summary_folderpath = createFolderInRepoPath("metrics_summary")
    metrics_tables_folderpath = createFolderInRepoPath("metrics_tables")
    experiment_data = getDatetimeAsString()
    
    metrics_summary_filepath = os.path.join(metrics_summary_folderpath, f"metrics_results_metric_case_{metric_case}_{experiment_data}.txt")
    metrics_tables_filepath = os.path.join(metrics_tables_folderpath, f"metrics_tables__metric_case_{metric_case}_{experiment_data}.txt")
     
    #Define print functions
    print_output = partial(print_output_in_specific_file, filename = metrics_summary_filepath)
    print_in_table_file = partial(print_output_in_specific_file, filename = metrics_tables_filepath)
    
    metrics_names = getMetricsNames(given_metric_names = given_metrics_names, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics,
                    other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics)
    
    table_headers = ['Experiment'] + [getMetricsTitle(metric_name, metrics_titles_dict) for metric_name in metrics_names]
    
    file_metrics_dict = {}    
    cases = ['test']
    for case in cases:
        print_output(f"-----case: {case} -------")
        table = PrettyTable(table_headers)
        table.align['Experiment'] = 'l'
        for filepath in filepaths:
            file = os.path.basename(filepath)
            print_output(f"\nFile: {file}")
            
            metrics_dict = outputMetricsForETThAlgorithm(filepath, case, print_func = print_output, metrics_names = metrics_names, metrics_titles_dict = metrics_titles_dict, plot_graphs = True, mean_plots = mean_plots, case_in_title = case_in_title, metric_case = metric_case)
            
            file_metrics_dict[file] = metrics_dict
            
            experiment_name = keepExperimentNameOnly(file)
            readable_experiment_name = getReadableExperimentName(experiment_name, readable_components)
             
            table_row = [readable_experiment_name]
            for metric_name in metrics_names:
                if mean_plots == True and 'mean' in metric_name:
                    mean_metric_name = metric_name
                    std_metric_name = metric_name.replace('mean', 'std')
                    mean_metric = metrics_dict[mean_metric_name][-1]
                    std_metric = metrics_dict[std_metric_name][-1]
                    mean_std_table_value = getMeanAndStdTableString(mean_metric, std_metric, mean_decimal_places, std_decimal_places)
                    table_row.append(mean_std_table_value)
                else:
                    metric = metrics_dict[metric_name][-1]
                    rounded_metric = round(metric, mean_decimal_places)
                    table_row.append(rounded_metric)
                
            table.add_row(table_row)
        
        print_in_table_file(f"-----case: {case} -------")
        print_in_table_file(str(table))
        
        createMetricsPerEpochComparisonPlot(file_metrics_dict, metrics_titles_dict, case, readable_components, mean_plots, case_in_title, metric_case)

def runMetricCase(args):
    metric_case = args.metric_case
    
    #Test metric cases
    if metric_case == 1:
        # Simple metric case for MAPE_full
        readable_components = ['se', 'pr', 'da', 'an'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = []
        or_filters = []
        given_metrics_names = ['MAPE_mean_5_full', 'MAPE_mean_5', 'MAPE_mean_5_red', 'my_mape', 'my_c1', 'my_c2', 'my_c3']
        metrics_titles_dict = {'MAPE_mean_5_full': 'MAPE_full', 'MAPE_mean_5': 'MAPE', 'MAPE_mean_5_red': 'MAPE_red', 'my_mape': 'my_mape', 'my_c1': 'my_c1', 'my_c2': 'my_c2', 'my_c3': 'my_c3'}
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = False
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 2:
        # Simple metric case for MAPE_full and MAPE
        readable_components = ['se', 'pr', 'da', 'an'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = []
        or_filters = []
        given_metrics_names = ['MAPE_mean_5_full', 'MAPE_mean_5', 'MAPE_mean_5_red', 'my_mape', 'my_c1', 'my_c2', 'my_c3']
        metrics_titles_dict = {'MAPE_mean_5_full': 'MAPE_full', 'MAPE_mean_5': 'MAPE', 'MAPE_mean_5_red': 'MAPE_red', 'my_mape': 'my_mape', 'my_c1': 'my_c1', 'my_c2': 'my_c2', 'my_c3': 'my_c3'}
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = False
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 3:
        # Simple metric case for mean plot for MAPE_mean_5_full
        readable_components = ['se', 'pr', 'da', 'an'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = []
        or_filters = []
        given_metrics_names = ['MAPE_mean_5_full', 'MAPE_mean_5', 'MAPE_mean_5_red', 'my_mape', 'my_c1', 'my_c2', 'my_c3']
        metrics_titles_dict = {'MAPE_mean_5_full': 'MAPE_full', 'MAPE_mean_5': 'MAPE', 'MAPE_mean_5_red': 'MAPE_red', 'my_mape': 'my_mape', 'my_c1': 'my_c1', 'my_c2': 'my_c2', 'my_c3': 'my_c3'}
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = True
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    #Thesis metric cases
    elif metric_case == 1001:
        # Comparison plot to demonstrate that decompose is better that non-decompose
        readable_components = ['se', 'pr', 'da', 'an', 'de'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = ['greek', 'pr24', 'an0', 'dp0.5', 'ftM', 'ukn0']
        or_filters = []
        given_metrics_names = ['MAPE_mean_5_full', 'MAPE_mean_5', 'MAPE_mean_5_red', 'my_mape', 'my_c1', 'my_c2', 'my_c3']
        metrics_titles_dict = {'MAPE_mean_5_full': 'MAPE_full', 'MAPE_mean_5': 'MAPE', 'MAPE_mean_5_red': 'MAPE_red', 'my_mape': 'my_mape', 'my_c1': 'my_c1', 'my_c2': 'my_c2', 'my_c3': 'my_c3'}
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = False
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)     
    elif metric_case == 1002:
        # Metrics table to demonstrate that decompose is better that non-decompose(Same as before but only for the table)
        readable_components = ['se', 'pr', 'da', 'an', 'de'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = ['greek', 'pr24', 'an0', 'dp0.5', 'ftM', 'ukn0']
        or_filters = []
        given_metrics_names = ['MAPE_mean_5_full', 'MAPE_mean_5', 'MAPE_mean_5_red', 'my_mape', 'my_c1', 'my_c2', 'my_c3']
        metrics_titles_dict = {'MAPE_mean_5_full': 'MAPE_full', 'MAPE_mean_5': 'MAPE', 'MAPE_mean_5_red': 'MAPE_red', 'my_mape': 'my_mape', 'my_c1': 'my_c1', 'my_c2': 'my_c2', 'my_c3': 'my_c3'}
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = True
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 1003:
        # Comparison plot to demonstrate that dropout 0.25 is marginally - mainly for big history values
        readable_components = ['se', 'pr', 'da', 'an', 'dp'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = ['greek', 'pr24', 'an0', 'decYes', 'ftM', 'ukn0']
        # remove an0 if you want to get all the results
        # and_filters = ['greek', 'pr24', 'decYes']
        or_filters = []
        given_metrics_names = ['MAPE_mean_5_full', 'MAPE_mean_5', 'MAPE_mean_5_red', 'my_mape', 'my_c1', 'my_c2', 'my_c3']
        metrics_titles_dict = {'MAPE_mean_5_full': 'MAPE_full', 'MAPE_mean_5': 'MAPE', 'MAPE_mean_5_red': 'MAPE_red', 'my_mape': 'my_mape', 'my_c1': 'my_c1', 'my_c2': 'my_c2', 'my_c3': 'my_c3'}
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = False
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 1004:
        # Metrics table to demonstrate that dropout 0.25 is marginally better - mainly for big history values(Same as before but only for the table)
        readable_components = ['se', 'pr', 'da', 'an', 'dp'] # sequence, prediction, data , anio
        given_experiment_names = []
        # and_filters = ['greek', 'pr24', 'an0', 'decYes', 'ftM', 'ukn0'] 
        # remove an0 if you want to get all the results
        and_filters = ['greek', 'pr24', 'decYes', 'ftM', 'ukn0']
        or_filters = []
        given_metrics_names = ['MAPE_mean_5_full', 'MAPE_mean_5', 'MAPE_mean_5_red', 'my_mape', 'my_c1', 'my_c2', 'my_c3']
        metrics_titles_dict = {'MAPE_mean_5_full': 'MAPE_full', 'MAPE_mean_5': 'MAPE', 'MAPE_mean_5_red': 'MAPE_red', 'my_mape': 'my_mape', 'my_c1': 'my_c1', 'my_c2': 'my_c2', 'my_c3': 'my_c3'}
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = True
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 1005:
        # Comparison plot to check history values and anio for pr24
        # Result : anio4 is the worst - an1, an2, an3 and an5 are almost the same - an1 is the best marginally 
        # Result: anio input is marginally better for small history e.g. his96 but marginally better for bigger history e.g. hist192, hist336
        # Result: metric improvement from 0.382 -> 0.336 from hist 96 to hist 192 and then 0.336 -> 0.320 for hist 192 to hist 336
        readable_components = ['se', 'pr', 'da', 'an', 'dp'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = ['greek', 'pr24', 'an1', 'decYes', 'dp0.25', 'ftM', 'ukn0']
        # and_filters = ['greek', 'pr24', 'decYes', 'dp0.25', 'ftM', 'ukn0']
        or_filters = []
        given_metrics_names = ['MAPE_mean_5_full', 'MAPE_mean_5', 'MAPE_mean_5_red', 'my_mape', 'my_c1', 'my_c2', 'my_c3']
        metrics_titles_dict = {'MAPE_mean_5_full': 'MAPE_full', 'MAPE_mean_5': 'MAPE', 'MAPE_mean_5_red': 'MAPE_red', 'my_mape': 'my_mape', 'my_c1': 'my_c1', 'my_c2': 'my_c2', 'my_c3': 'my_c3'}
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = False
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 1006:
        # Metrics table to check history values and anio for pr24 (Same as before but only for the table)
        # Result : anio4 is the worst - an1, an2, an3 and an5 are almost the same - an1 is the best marginally 
        # Result: anio input is marginally better for small history e.g. his96 but marginally better for bigger history e.g. hist192, hist336
        # Result: metric improvement from 0.382 -> 0.336 from hist 96 to hist 192 and then 0.336 -> 0.320 for hist 192 to hist 336
        readable_components = ['se', 'pr', 'da', 'an', 'dp'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = ['greek', 'pr24', 'decYes', 'dp0.25', 'ftM', 'ukn0']
        or_filters = []
        given_metrics_names = ['MAPE_mean_5_full', 'MAPE_mean_5', 'MAPE_mean_5_red', 'my_mape', 'my_c1', 'my_c2', 'my_c3']
        metrics_titles_dict = {'MAPE_mean_5_full': 'MAPE_full', 'MAPE_mean_5': 'MAPE', 'MAPE_mean_5_red': 'MAPE_red', 'my_mape': 'my_mape', 'my_c1': 'my_c1', 'my_c2': 'my_c2', 'my_c3': 'my_c3'}
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = True
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 1007:
        # Comparison plot to check history values and anio for pr48
        # Result : anio4 is the worst - an1, an2, an3 and an5 are almost the same - an1 is the best marginally 
        # Result: anio input is marginally better for small history e.g. his96 but marginally better for bigger history e.g. hist192, hist336
        # Result: metric improvement from 0.0382 -> 0.0336 from hist 96 to hist 192 and then 0.0336 -> 0.0320 for hist 192 to hist 336
        readable_components = ['se', 'pr', 'da', 'an', 'dp'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = ['greek', 'pr48', 'an1', 'decYes', 'dp0.25', 'ftM', 'ukn0']
        # and_filters = ['greek', 'pr48', 'decYes', 'dp0.25', 'ftM', 'ukn0']
        or_filters = []
        given_metrics_names = ['MAPE_mean_5_full', 'MAPE_mean_5', 'MAPE_mean_5_red', 'my_mape', 'my_c1', 'my_c2', 'my_c3']
        metrics_titles_dict = {'MAPE_mean_5_full': 'MAPE_full', 'MAPE_mean_5': 'MAPE', 'MAPE_mean_5_red': 'MAPE_red', 'my_mape': 'my_mape', 'my_c1': 'my_c1', 'my_c2': 'my_c2', 'my_c3': 'my_c3'}
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = False
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 1008:
        # Metrics table to check history values and anio for pr48 (Same as before but only for the table)
        # Result : anio4 is the worst - an1, an2, an3 and an5 are almost the same - an1 is the best marginally 
        # Result: anio input is marginally better for small history e.g. his96 but marginally better for bigger history e.g. hist192, hist336
        # Result: metric improvement from 0.0436 -> 0.0384 from hist 96 to hist 192 and then 0.0384 -> 0.0370 for hist 192 to hist 336
        readable_components = ['se', 'pr', 'da', 'an', 'dp'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = ['greek', 'pr48', 'decYes', 'dp0.25', 'ftM', 'ukn0']
        or_filters = []
        given_metrics_names = ['MAPE_mean_5_full', 'MAPE_mean_5', 'MAPE_mean_5_red', 'my_mape', 'my_c1', 'my_c2', 'my_c3']
        metrics_titles_dict = {'MAPE_mean_5_full': 'MAPE_full', 'MAPE_mean_5': 'MAPE', 'MAPE_mean_5_red': 'MAPE_red', 'my_mape': 'my_mape', 'my_c1': 'my_c1', 'my_c2': 'my_c2', 'my_c3': 'my_c3'}
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = True
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 1009:
        # Comparison plot to check history values and anio for pr168, pr336 and pr720
        # Result : anio4 is the worst - an1, an2, an3 and an5 are almost the same - an1 is the best marginally 
        # Result: anio input is marginally better for small history e.g. his96 but marginally better for bigger history e.g. hist192, hist336
        # Result: metric improvement from 0.0382 -> 0.0336 from hist 96 to hist 192 and then 0.0336 -> 0.0320 for hist 192 to hist 336
        readable_components = ['se', 'pr', 'da', 'an', 'dp'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = ['greek', 'an4', 'decYes', 'dp0.25', 'ftM', 'ukn0']
        # and_filters = ['greek', 'decYes', 'dp0.25']
        or_filters = ['pr168', 'pr336', 'pr720']
        given_metrics_names = ['MAPE_mean_5_full', 'MAPE_mean_5', 'MAPE_mean_5_red', 'my_mape', 'my_c1', 'my_c2', 'my_c3']
        metrics_titles_dict = {'MAPE_mean_5_full': 'MAPE_full', 'MAPE_mean_5': 'MAPE', 'MAPE_mean_5_red': 'MAPE_red', 'my_mape': 'my_mape', 'my_c1': 'my_c1', 'my_c2': 'my_c2', 'my_c3': 'my_c3'}
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = False
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 1010:
        # Metrics table to check history values and anio for pr168, pr336 and pr720 (Same as before but only for the table)
        # Result : anio4 is the best by far - an1, an2, an3 and an5 are almost the same - an2 is the best marginally of them
        # Result: anio input is marginally better for all cases since we have bigger history in this case
        # Result: More history is needed as we go to bigger prediction values but then the algorithm becomes very slow
        # Result: Anio helps dramatically more for bigger prediction values e.g. pr720 -> 0.1040 -> 0.059 and 0.0437 for anio4 with input
        readable_components = ['se', 'pr', 'da', 'an', 'dp'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = ['greek', 'decYes', 'dp0.25', 'ftM', 'ukn0']
        or_filters = ['pr168', 'pr336', 'pr720']
        given_metrics_names = ['MAPE_mean_5_full', 'MAPE_mean_5', 'MAPE_mean_5_red', 'my_mape', 'my_c1', 'my_c2', 'my_c3']
        metrics_titles_dict = {'MAPE_mean_5_full': 'MAPE_full', 'MAPE_mean_5': 'MAPE', 'MAPE_mean_5_red': 'MAPE_red', 'my_mape': 'my_mape', 'my_c1': 'my_c1', 'my_c2': 'my_c2', 'my_c3': 'my_c3'}
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = True
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 1011:
        # Comparison plot to check history values for all pr with single anio0
        # Result The metrics improve as we increase but after a point there is no singificant improvement and the time increases dramatically
        readable_components = ['se', 'pr', 'da', 'an', 'dp'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = ['greek', 'an0', 'decYes', 'dp0.25', 'ftM', 'ukn0']
        or_filters = []
        given_metrics_names = ['MAPE_mean_5_full', 'MAPE_mean_5', 'MAPE_mean_5_red', 'my_mape', 'my_c1', 'my_c2', 'my_c3']
        metrics_titles_dict = {'MAPE_mean_5_full': 'MAPE_full', 'MAPE_mean_5': 'MAPE', 'MAPE_mean_5_red': 'MAPE_red', 'my_mape': 'my_mape', 'my_c1': 'my_c1', 'my_c2': 'my_c2', 'my_c3': 'my_c3'}
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = False
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 1012:
        # Metrics table to check history values for all pr with single anio0 (Same as before but only for the table)
        # Result The metrics improve as we increase but after a point there is no singificant improvement and the time increases dramatically
        readable_components = ['se', 'pr', 'da', 'an', 'dp'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = ['greek', 'an0', 'decYes', 'dp0.25', 'ftM', 'ukn0']
        or_filters = []
        given_metrics_names = ['MAPE_mean_5_full', 'MAPE_mean_5', 'MAPE_mean_5_red', 'my_mape', 'my_c1', 'my_c2', 'my_c3']
        metrics_titles_dict = {'MAPE_mean_5_full': 'MAPE_full', 'MAPE_mean_5': 'MAPE', 'MAPE_mean_5_red': 'MAPE_red', 'my_mape': 'my_mape', 'my_c1': 'my_c1', 'my_c2': 'my_c2', 'my_c3': 'my_c3'}
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = True
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 1013:
        # Metrics table to check the difference between multivariate vs univariate
        readable_components = ['se', 'pr', 'da', 'an'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = []
        or_filters = ['seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an0No',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an1Yes',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an1No',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an4Yes',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an4No',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an0No',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an1Yes',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an1No',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an4Yes',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an4No',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an0No',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an1Yes',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an1No',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an4Yes',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an4No',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an0No',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an1Yes',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an1No',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an4Yes',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an4No',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftS_shifYes_decNo_datgreek_ep300_ukn0_an0No',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftS_shifYes_decNo_datgreek_ep300_ukn0_an1Yes',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftS_shifYes_decNo_datgreek_ep300_ukn0_an1No',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftS_shifYes_decNo_datgreek_ep300_ukn0_an4Yes',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftS_shifYes_decNo_datgreek_ep300_ukn0_an4No',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftS_shifYes_decNo_datgreek_ep300_ukn0_an0No',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftS_shifYes_decNo_datgreek_ep300_ukn0_an1Yes',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftS_shifYes_decNo_datgreek_ep300_ukn0_an1No',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftS_shifYes_decNo_datgreek_ep300_ukn0_an4Yes',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftS_shifYes_decNo_datgreek_ep300_ukn0_an4No',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftS_shifYes_decNo_datgreek_ep300_ukn0_an0No',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftS_shifYes_decNo_datgreek_ep300_ukn0_an1Yes',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftS_shifYes_decNo_datgreek_ep300_ukn0_an1No',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftS_shifYes_decNo_datgreek_ep300_ukn0_an4Yes',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftS_shifYes_decNo_datgreek_ep300_ukn0_an4No',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftS_shifYes_decNo_datgreek_ep300_ukn0_an0No',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftS_shifYes_decNo_datgreek_ep300_ukn0_an1Yes',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftS_shifYes_decNo_datgreek_ep300_ukn0_an1No',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftS_shifYes_decNo_datgreek_ep300_ukn0_an4Yes',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftS_shifYes_decNo_datgreek_ep300_ukn0_an4No']
        given_metrics_names = ['MAPE_mean_5_full', 'MAPE_mean_5', 'MAPE_mean_5_red', 'my_mape', 'my_c1', 'my_c2', 'my_c3']
        metrics_titles_dict = {'MAPE_mean_5_full': 'MAPE_full', 'MAPE_mean_5': 'MAPE', 'MAPE_mean_5_red': 'MAPE_red', 'my_mape': 'my_mape', 'my_c1': 'my_c1', 'my_c2': 'my_c2', 'my_c3': 'my_c3'}
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = True
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 1014:
        # Metrics table to check the difference between unknown days
        readable_components = ['se', 'pr', 'da', 'an'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = []
        or_filters = ['seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an0No',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an1Yes',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an1No',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an4Yes',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an4No',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an0No',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an1Yes',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an1No',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an4Yes',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an4No',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an0No',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an1Yes',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an1No',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an4Yes',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an4No',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an0No',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an1Yes',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an1No',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an4Yes',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn0_an4No',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn1_an0No',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn1_an1Yes',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn1_an1No',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn1_an4Yes',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn1_an4No',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn1_an0No',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn1_an1Yes',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn1_an1No',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn1_an4Yes',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn1_an4No',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn1_an0No',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn1_an1Yes',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn1_an1No',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn1_an4Yes',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn1_an4No',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn1_an0No',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn1_an1Yes',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn1_an1No',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn1_an4Yes',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn1_an4No',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn4_an0No',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn4_an1Yes',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn4_an1No',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn4_an4Yes',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn4_an4No',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn4_an0No',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn4_an1Yes',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn4_an1No',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn4_an4Yes',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn4_an4No',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn4_an0No',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn4_an1Yes',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn4_an1No',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn4_an4Yes',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn4_an4No',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn4_an0No',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn4_an1Yes',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn4_an1No',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn4_an4Yes',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn4_an4No',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn10_an0No',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn10_an1Yes',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn10_an1No',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn10_an4Yes',
                      'seq336_lab24_pr24_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn10_an4No',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn10_an0No',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn10_an1Yes',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn10_an1No',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn10_an4Yes',
                      'seq336_lab48_pr48_lr1e-05_lev3_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn10_an4No',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn10_an0No',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn10_an1Yes',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn10_an1No',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn10_an4Yes',
                      'seq336_lab336_pr336_lr1e-05_lev4_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn10_an4No',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn10_an0No',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn10_an1Yes',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn10_an1No',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn10_an4Yes',
                      'seq736_lab720_pr720_lr1e-05_lev5_dp0.25_st1_ftM_shifYes_decNo_datgreek_ep300_ukn10_an4No',]
        given_metrics_names = ['MAPE_mean_5_full', 'MAPE_mean_5', 'MAPE_mean_5_red', 'my_mape', 'my_c1', 'my_c2', 'my_c3']
        metrics_titles_dict = {'MAPE_mean_5_full': 'MAPE_full', 'MAPE_mean_5': 'MAPE', 'MAPE_mean_5_red': 'MAPE_red', 'my_mape': 'my_mape', 'my_c1': 'my_c1', 'my_c2': 'my_c2', 'my_c3': 'my_c3'}
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = True
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 1101:
        # Comparison plot to demonstrate that decompose is better that non-decompose in ETTh1
        readable_components = ['se', 'pr', 'da', 'an', 'de'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = ['ETTh1', 'an0', 'dp0.5', 'ftM']
        or_filters = []
        given_metrics_names = ['MSE_mean_5_full', 'MSE_mean_5', 'MSE_mean_5_red', 'MAE_mean_5_full', 'MAE_mean_5', 'MAE_mean_5_red']
        metrics_titles_dict = {'MSE_mean_5_full': 'MSE_full', 'MSE_mean_5': 'MSE', 'MSE_mean_5_red': 'MSE_red', 'MAE_mean_5_full': 'MAE_full', 'MAE_mean_5': 'MAE', 'MAE_mean_5_red': 'MAE_red' }
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = False
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)     
    elif metric_case == 1102:
        # Metrics table to demonstrate that decompose is better that non-decompose in ETTh1(Same as before but only for the table)
        readable_components = ['se', 'pr', 'da', 'an', 'de'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = ['ETTh1', 'an0', 'dp0.5', 'ftM']
        or_filters = []
        given_metrics_names = ['MSE_mean_5_full', 'MSE_mean_5', 'MSE_mean_5_red', 'MAE_mean_5_full', 'MAE_mean_5', 'MAE_mean_5_red']
        metrics_titles_dict = {'MSE_mean_5_full': 'MSE_full', 'MSE_mean_5': 'MSE', 'MSE_mean_5_red': 'MSE_red', 'MAE_mean_5_full': 'MAE_full', 'MAE_mean_5': 'MAE', 'MAE_mean_5_red': 'MAE_red' }
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = True
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 1103:
        # Comparison plot to demonstrate that there are not sufficient data to compare dropout 0.5 and 0.25 in ETTh1. paper uses 0.25 for specific values but it seems the effect is marginal
        readable_components = ['se', 'pr', 'da', 'an', 'dp'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = ['ETTh1', 'an0', 'decYes']
        # remove an0 if you want to get all the results
        # and_filters = ['ETTh1', 'pr24', 'decYes']
        or_filters = []
        given_metrics_names = ['MSE_mean_5_full', 'MSE_mean_5', 'MSE_mean_5_red', 'MAE_mean_5_full', 'MAE_mean_5', 'MAE_mean_5_red']
        metrics_titles_dict = {'MSE_mean_5_full': 'MSE_full', 'MSE_mean_5': 'MSE', 'MSE_mean_5_red': 'MSE_red', 'MAE_mean_5_full': 'MAE_full', 'MAE_mean_5': 'MAE', 'MAE_mean_5_red': 'MAE_red' }
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = False
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 1104:
        # demonstrate that there are not sufficient data to compare dropout 0.5 and 0.25 in ETTh1. paper uses 0.25 for specific values but it seems the effect is marginal(Same as before but only for the table)
        readable_components = ['se', 'pr', 'da', 'an', 'dp'] # sequence, prediction, data , anio
        given_experiment_names = []
        # and_filters = ['ETTh1', 'an0', 'decYes'] 
        # remove an0 if you want to get all the results
        and_filters = ['ETTh1', 'pr24', 'decYes']
        or_filters = []
        given_metrics_names = ['MSE_mean_5_full', 'MSE_mean_5', 'MSE_mean_5_red', 'MAE_mean_5_full', 'MAE_mean_5', 'MAE_mean_5_red']
        metrics_titles_dict = {'MSE_mean_5_full': 'MSE_full', 'MSE_mean_5': 'MSE', 'MSE_mean_5_red': 'MSE_red', 'MAE_mean_5_full': 'MAE_full', 'MAE_mean_5': 'MAE', 'MAE_mean_5_red': 'MAE_red' }
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = True
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 1105:
        # Comparison plot to check history values and anio in ETTh1
        # Result : no anio is the best - scinet/anio method doesn't work for ETTh1 
        # Result : multivariate makes the result more smooth sometimes and improves the metrics
        # Result: metric improvement with history window increase
        readable_components = ['se', 'pr', 'da', 'an', 'dp', 'ft'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = ['ETTh1', 'an0', 'decYes']
        or_filters = []
        given_metrics_names = ['MSE_mean_5_full', 'MSE_mean_5', 'MSE_mean_5_red', 'MAE_mean_5_full', 'MAE_mean_5', 'MAE_mean_5_red']
        metrics_titles_dict = {'MSE_mean_5_full': 'MSE_full', 'MSE_mean_5': 'MSE', 'MSE_mean_5_red': 'MSE_red', 'MAE_mean_5_full': 'MAE_full', 'MAE_mean_5': 'MAE', 'MAE_mean_5_red': 'MAE_red' }
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = False
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 1106:
        # Metrics table to check history values and anio in ETTh1(Same as before but only for the table)
        # Result : no anio is the best - scinet/anio method doesn't work for ETTh1 
        # Result : multivariate makes the result more smooth sometimes and improves the metrics
        # Result: metric improvement with history window increase
        readable_components = ['se', 'pr', 'da', 'an', 'dp', 'ft'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = ['ETTh1', 'an0', 'decYes']
        or_filters = []
        given_metrics_names = ['MSE_mean_5_full', 'MSE_mean_5', 'MSE_mean_5_red', 'MAE_mean_5_full', 'MAE_mean_5', 'MAE_mean_5_red']
        metrics_titles_dict = {'MSE_mean_5_full': 'MSE_full', 'MSE_mean_5': 'MSE', 'MSE_mean_5_red': 'MSE_red', 'MAE_mean_5_full': 'MAE_full', 'MAE_mean_5': 'MAE', 'MAE_mean_5_red': 'MAE_red' }
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = True
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 1107:
        # Metrics table to check if results are same as the paper for ETTh1.
        # Result They are not the same as the paper. Most values are around [10,30] and the difference is 3-4 depending are 1.5 for MAE and 3 for MSE. 1.5/30 is 5% so this could be 0.05.
        # Result : The only single way the reulsts might be correct is to use the percentage of the values.
        readable_components = ['se', 'pr', 'da', 'an', 'dp', 'ft'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = ['ETTh1', 'an0', 'decYes']
        or_filters = []
        given_metrics_names = ['MSE', 'MSE_mean_5_full', 'MSE_mean_5', 'MSE_mean_5_red', 'MAE', 'MAE_mean_5_full', 'MAE_mean_5', 'MAE_mean_5_red']
        metrics_titles_dict = {'MSE': 'MSE_paper', 'MSE_mean_5_full': 'MSE_full', 'MSE_mean_5': 'MSE', 'MSE_mean_5_red': 'MSE_red', 'MAE': 'MAE_paper', 'MAE_mean_5_full': 'MAE_full', 'MAE_mean_5': 'MAE', 'MAE_mean_5_red': 'MAE_red' }
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = True
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 1201:
        # Comparison plot to demonstrate that decompose is better that non-decompose in ETTh2
        readable_components = ['se', 'pr', 'da', 'an', 'de'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = ['ETTh2', 'an0', 'dp0.5', 'ftM']
        or_filters = []
        given_metrics_names = ['MSE_mean_5_full', 'MSE_mean_5', 'MSE_mean_5_red', 'MAE_mean_5_full', 'MAE_mean_5', 'MAE_mean_5_red']
        metrics_titles_dict = {'MSE_mean_5_full': 'MSE_full', 'MSE_mean_5': 'MSE', 'MSE_mean_5_red': 'MSE_red', 'MAE_mean_5_full': 'MAE_full', 'MAE_mean_5': 'MAE', 'MAE_mean_5_red': 'MAE_red' }
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = False
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)     
    elif metric_case == 1202:
        # Metrics table to demonstrate that decompose is better that non-decompose in ETTh2(Same as before but only for the table)
        readable_components = ['se', 'pr', 'da', 'an', 'de'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = ['ETTh1', 'an0', 'dp0.5', 'ftM']
        or_filters = []
        given_metrics_names = ['MSE_mean_5_full', 'MSE_mean_5', 'MSE_mean_5_red', 'MAE_mean_5_full', 'MAE_mean_5', 'MAE_mean_5_red']
        metrics_titles_dict = {'MSE_mean_5_full': 'MSE_full', 'MSE_mean_5': 'MSE', 'MSE_mean_5_red': 'MSE_red', 'MAE_mean_5_full': 'MAE_full', 'MAE_mean_5': 'MAE', 'MAE_mean_5_red': 'MAE_red' }
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = True
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 1203:
        # Comparison plot to demonstrate that dropout 0.0 is better than 0.25 and 0.5 for some cases. For some cases the paper uses dropout 0.5 in ETTh2
        readable_components = ['se', 'pr', 'da', 'an', 'dp', 'ft'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = ['ETTh2', 'an0', 'decYes']
        # remove an0 if you want to get all the results
        # and_filters = ['ETTh2', 'pr24', 'decYes']
        or_filters = []
        given_metrics_names = ['MSE_mean_5_full', 'MSE_mean_5', 'MSE_mean_5_red', 'MAE_mean_5_full', 'MAE_mean_5', 'MAE_mean_5_red']
        metrics_titles_dict = {'MSE_mean_5_full': 'MSE_full', 'MSE_mean_5': 'MSE', 'MSE_mean_5_red': 'MSE_red', 'MAE_mean_5_full': 'MAE_full', 'MAE_mean_5': 'MAE', 'MAE_mean_5_red': 'MAE_red' }
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = False
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 1204:
        # Metrics table to demonstrate that dropout 0.0 is better than 0.25 and 0.5 for some cases. For some cases the paper uses dropout 0.5 in ETTh2(Same as before but only for the table)
        readable_components = ['se', 'pr', 'da', 'an', 'dp'] # sequence, prediction, data , anio
        given_experiment_names = []
        # and_filters = ['ETTh2', 'an0', 'decYes'] 
        # remove an0 if you want to get all the results
        and_filters = ['ETTh2', 'pr24', 'decYes']
        or_filters = []
        given_metrics_names = ['MSE_mean_5_full', 'MSE_mean_5', 'MSE_mean_5_red', 'MAE_mean_5_full', 'MAE_mean_5', 'MAE_mean_5_red']
        metrics_titles_dict = {'MSE_mean_5_full': 'MSE_full', 'MSE_mean_5': 'MSE', 'MSE_mean_5_red': 'MSE_red', 'MAE_mean_5_full': 'MAE_full', 'MAE_mean_5': 'MAE', 'MAE_mean_5_red': 'MAE_red' }
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = True
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 1205:
        # Comparison plot to check history values and anio in ETTh2
        # Result : no anio is the best - scinet/anio method doesn't work for ETTh2
        # Result : multivariate makes the result more smooth sometimes and improves the metrics 
        # Result: metric improvement with history window increase
        readable_components = ['se', 'pr', 'da', 'an', 'dp', 'ft'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = ['ETTh2', 'an0', 'decYes']
        or_filters = []
        given_metrics_names = ['MSE_mean_5_full', 'MSE_mean_5', 'MSE_mean_5_red', 'MAE_mean_5_full', 'MAE_mean_5', 'MAE_mean_5_red']
        metrics_titles_dict = {'MSE_mean_5_full': 'MSE_full', 'MSE_mean_5': 'MSE', 'MSE_mean_5_red': 'MSE_red', 'MAE_mean_5_full': 'MAE_full', 'MAE_mean_5': 'MAE', 'MAE_mean_5_red': 'MAE_red' }
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = False
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 1206:
        # Metrics table to check history values and anio in ETTh2(Same as before but only for the table)
        # Result : no anio is the best - scinet/anio method doesn't work for ETTh2
        # Result : multivariate makes the result more smooth sometimes and improves the metrics 
        # Result: metric improvement with history window increase
        readable_components = ['se', 'pr', 'da', 'an', 'dp', 'ft'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = ['ETTh2', 'an0', 'decYes']
        or_filters = []
        given_metrics_names = ['MSE_mean_5_full', 'MSE_mean_5', 'MSE_mean_5_red', 'MAE_mean_5_full', 'MAE_mean_5', 'MAE_mean_5_red']
        metrics_titles_dict = {'MSE_mean_5_full': 'MSE_full', 'MSE_mean_5': 'MSE', 'MSE_mean_5_red': 'MSE_red', 'MAE_mean_5_full': 'MAE_full', 'MAE_mean_5': 'MAE', 'MAE_mean_5_red': 'MAE_red' }
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = True
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)
    elif metric_case == 1207:
        # Metrics table to check if results are same as the paper for ETTh2.
        # Result They are not the same as the paper. Most values are around [10,40] and the difference is 2-4 depending are 2.1 for MAE and 8 for MSE. 2.1/50 is 4% so this could be 0.04.
        # Result : The only single way the reulsts might be correct is to use the percentage of the values.
        readable_components = ['se', 'pr', 'da', 'an', 'dp', 'ft'] # sequence, prediction, data , anio
        given_experiment_names = []
        and_filters = ['ETTh2', 'an0', 'decYes']
        or_filters = []
        given_metrics_names = ['MSE', 'MSE_mean_5_full', 'MSE_mean_5', 'MSE_mean_5_red', 'MAE', 'MAE_mean_5_full', 'MAE_mean_5', 'MAE_mean_5_red']
        metrics_titles_dict = {'MSE': 'MSE_paper', 'MSE_mean_5_full': 'MSE_full', 'MSE_mean_5': 'MSE', 'MSE_mean_5_red': 'MSE_red', 'MAE': 'MAE_paper', 'MAE_mean_5_full': 'MAE_full', 'MAE_mean_5': 'MAE', 'MAE_mean_5_red': 'MAE_red' }
        financial_metrics = False
        ETTh_metrics = False
        other_metrics = False
        my_mape_metric = False
        my_C_metrics = False
        mean_plots = True
        case_in_title = False
        mean_decimal_places = 4
        std_decimal_places = 4
    
        generateMetricsOutputFiles(readable_components = readable_components, given_experiment_names = given_experiment_names, and_filters = and_filters, or_filters = or_filters, 
                                   given_metrics_names = given_metrics_names, metrics_titles_dict = metrics_titles_dict, financial_metrics = financial_metrics, ETTh_metrics = ETTh_metrics, 
                                   other_metrics = other_metrics, my_mape_metric = my_mape_metric, my_C_metrics = my_C_metrics, mean_plots = mean_plots, case_in_title = case_in_title,
                                   mean_decimal_places = mean_decimal_places, std_decimal_places = std_decimal_places, metric_case = metric_case)

def main():
    parser = argparse.ArgumentParser(description='Metric summary script')

    parser.add_argument('--metric_case', type=int, default=1)
    parser.add_argument('--python_name',type=str, default='python', choices=['python', 'python3'])

    args = parser.parse_args()

    print(f"metric_case = {args.metric_case}")
    
    runMetricCase(args)
      
if __name__ == "__main__":
    main()

