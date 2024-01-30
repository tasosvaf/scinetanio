import os
import torch
from datetime import datetime
from utils.utils_ETTh import getDatetimeAsString
from utils.utils_ETTh import createGreekScinetDatasetWithCheck
from utils.utils_ETTh import createFolderInSciNetFolder, getGreekScinetDatasetNames
from experiments.exp_financial import Exp_financial
import argparse
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='SCINet on financial datasets')
### -------  dataset settings --------------
parser.add_argument('--dataset_name', type=str, default= 'greek_scinet_dataset_load_load', choices=['electricity', 'solar_AL', 'exchange_rate', 'traffic']+ getGreekScinetDatasetNames())
parser.add_argument('--data', type=str, default='./datasets/' + 'greek_scinet_dataset_load_load' + '.txt',
                    help='location of the data file')
parser.add_argument('--normalize', type=int, default=2)

### -------  device settings --------------
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--gpu', type=int, default=0, help='gpu')

### -------  input/output length settings --------------                                                                            
parser.add_argument('--window_size', type=int, default=168, help='input length')
parser.add_argument('--horizon', type=int, default=24, help='prediction length')
parser.add_argument('--future_unknown_days', type=int, default=0, help='future unknown days')
parser.add_argument('--concat_len', type=int, default=165)
parser.add_argument('--single_step', type=int, default=1, help='only supervise the final setp')
parser.add_argument('--single_step_output_One', type=int, default=0, help='only output the single final step')
parser.add_argument('--lastWeight', type=float, default=1.0,help='Loss weight lambda on the final step')

### -------  training settings --------------  
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--evaluate', type=bool, default=True)
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--num_nodes',type=int,default=8,help='number of nodes/variables')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--lr',type=float,default=5e-3,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=2,help='')
parser.add_argument('--lradj', type=int, default=1,help='adjust learning rate')
parser.add_argument('--save_path', type=str, default='exp/financial_checkpoints/')
parser.add_argument('--model_name', type=str, default='SCINet')

### -------  model settings --------------  
parser.add_argument('--hidden-size', default=8.0, type=float, help='hidden channel of module')# H, EXPANSION RATE
parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
parser.add_argument('--kernel', default=5, type=int, help='kernel size')#k kernel size
parser.add_argument('--dilation', default=1, type=int, help='dilation')
parser.add_argument('--positionalEcoding', type = bool , default=False)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--groups', type=int, default=1)
parser.add_argument('--levels', type=int, default=3)
parser.add_argument('--num_decoder_layer', type=int, default=3)
parser.add_argument('--stacks', type=int, default=2)
parser.add_argument('--long_term_forecast', action='store_true', default=False)
parser.add_argument('--RIN', type=bool, default=False)
parser.add_argument('--decompose', type=str, default='No', choices=['Yes', 'No'])

args = parser.parse_args()

if not args.long_term_forecast:
    args.concat_len = args.window_size - args.horizon

if args.dataset_name.startswith("greek_scinet_dataset_"):
    createGreekScinetDatasetWithCheck(args.dataset_name)

if __name__ == '__main__':

    torch.manual_seed(4321)  # reproducible
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
    torch.backends.cudnn.enabled = True

    Exp=Exp_financial
    exp=Exp(args)
    
    experiment_date = getDatetimeAsString()
    pred_folder_name = f"financ_predictions_{experiment_date}"
    
    pred_folder_path = createFolderInSciNetFolder(pred_folder_name)
    
    
    pred_file_name = f"{str(args.epochs)}epochs-{str(args.future_unknown_days)}unknown_days-predictions.csv"
    pred_file_path = os.path.join(pred_folder_path, pred_file_name)
    train_pred_file_name = f"{str(args.epochs)}epochs-{str(args.future_unknown_days)}unknown_days-training_predictions.csv"
    train_pred_file_path = os.path.join(pred_folder_path, train_pred_file_name)  
    if args.train or args.resume:
        data=exp._get_data()
        before_train = datetime.now().timestamp()
        print("===================Normal-Start=========================")
        normalize_statistic = exp.train(train_pred_file_path)
        after_train = datetime.now().timestamp()
        print(f'Training took {(after_train - before_train) / 60} minutes')
        print("===================Normal-End=========================")
        exp.validate(data,data.test[0],data.test[1], evaluate=True)

    if args.evaluate:
        data=exp._get_data()
        before_evaluation = datetime.now().timestamp()
        if args.stacks == 1:
            predictions, y_test, rse, rae, correlation, mse, mae= exp.validate(data,data.test[0],data.test[1], evaluate=True)
        else:
            predictions, y_test, rse, rae, correlation, rse_mid, rae_mid, correlation_mid, mse, mae = exp.validate(data,data.test[0],data.test[1], evaluate=True)
        after_evaluation = datetime.now().timestamp()
        print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')
        df = pd.DataFrame()
        pred = []
        true = []
        predictions *= data.get_max()
        y_test *= data.get_max()
        for i in range(0, args.groups):
            pred.append([])
        for predict in predictions:
            for i in range(0, args.groups):
                pred[i].append(predict[i])
        for i in range(0, args.groups):
            df['predictions_' + str(i)] = pred[i]
        for tr in y_test:
            true.append(tr[0])
        df['true'] = true
        df.to_csv(pred_file_path, index = False)





