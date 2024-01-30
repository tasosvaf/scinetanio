# tasosthesis

This is the code for Tasos Vafeidis thesis that tries to extend SCINet method to do the following:

- Predict days ahead
- Combine method with anio methodology
- Apply method to greek electricity dataset

## Utility code and scripts

A lot of utility code has been added to the project.
Utility code can be found in :

- SCINet/utils/utils_ETTh.py file : Library with utility functionality
- scripts folder :
  - mult_run.py : Script to enable us run multiple experiments one afte the other.
  - group_run.py : Script to enable us run mult_run.py file multiple times in parallel for different arguments.
  - Multi_cities_Deh_Dataset.csv : Greek electricity dataset
  - create_datasets.py : Create greek energy and ETTh datasets (This is optional before running the algorithm. The algorithm looks for the datasets and if they don't exist it created them once).
  - delete_datasets.py : Delete greek energy and ETTh datasets
  - delete_results.py : Delete all the results produced by running the algorithm
  - metrics_summary.py : Script to generate metrics summary, metrics tables, plots and comparison plots. There are many different metric cases inside that enable the user to generate different types of plots, tables and summaries.
  - delete_metrics_ouput.py : Script to delete metrics related output(metrics_summary, metrics tables, plots and comparison plots)
  - experiment_runs.txt : Helping commands for running mult_run.py for various cases in Linux environment

## Run multiple runs

For convenience and creating many different experiments to compare results a script(scripts/mult_run.py) has been added.

### Arguments

- run_cases : A list of integers where each integer represents a run_case that has been defined inside scripts/mult_run.py script. Each run_case contains many different experiments.(Default value : [-2])
- python_name : This value can be either 'python' or 'python3' depending on the name of python version 3 in the machine. (Default value : 'python')
- log : This can be either 'Yes' or 'No'. log = 'Yes' means that the log will be stored in files. (Default value : 'Yes')
- gpu : This sets in which gpu the script will run. The gpu is being set only for that script. The user can select one of the following values ['', '0', '1', '2', '3'] (Default value : '')
- run_same_experiment : This sets if the experiment will run twice if there is already result from older experiment run that is identical. This can be either 'Yes' or 'No' (Default value : 'No')

### Examples

```
python mult_run.py --run_cases 1 2 3 --log Yes  --gpu 2
```

```
python3 mult_run.py --run_cases 1 2 3 --python_name python3 --log Yes
```

```
python mult_run.py --run_cases 1 -3 15 --gpu 0  --run_same_experiment No
```

## Run group runs

For convenience and creating a group containing of many different experiments a script(scripts/group_run.py) has been added.

### Arguments

- group_case : Each group case defines a list of arguments and setup up for different runs of mult_run.py script that will run in parallel (Default value : 1)
- python_name : This value can be either 'python' or 'python3' depending on the name of python version 3 in the machine. (Default value : 'python')

### Examples

```
python group_run.py --group_case 1
```

```
python3 scripts/group_run.py --group_case 4 --python_name python3
```

## Run metrics summary

For convenience a script has been added to generate metrics summary, metrics tables, plots and comparison plots. Inside the script the user can specify
what it wants as a metric cases in order the metrics output files to be done based on that selection.

### Arguments

- metric_case : Each metric case defines a list of metrics and a list of experiments the metrics summary algorithm should consider. (Default value : 1)
- python_name : This value can be either 'python' or 'python3' depending on the name of python version 3 in the machine. (Default value : 'python')

### Examples

```
python metrics_summary.py --metric_case 1
```

```
python3 metrics_summary.py --metric_case 2 --python_name python3
```
