import os
import json
import time
import glob

import numpy as np
import pandas as pd



path = './savemodel2/InceptBackbone/'
dataset_list = os.listdir(path)

outpath ='savemodel2_csv/' #savemodel5_
os.makedirs(outpath,exist_ok=True)

# dataset_list =['Handwriting']



for dataset_name in dataset_list:
    print(dataset_name)
    dataset_path = path + dataset_name

    exp_list = os.listdir(dataset_path)
    
    df_list = []
    for exp in exp_list:
        print(' '+exp)
        exp_path = os.path.join(dataset_path, exp)

        '''
        option.txt
        
        '''
        option_txt = glob.glob(os.path.join(exp_path, '*.txt'))[0]
    
        with open(option_txt, 'r') as file:
            lines = file.readlines()
        
        data_dict = {}
        
        for line in lines:
            parts = line.strip().split(': ')
            if len(parts) == 2:
                key, value = parts
                data_dict[key] = [value]
        
        df = pd.DataFrame(data_dict)
        df.index = [exp]

        # df = df.rename_axis(exp)

        # df = df.transpose()
        # df.reset_index(inplace=True)
        # df.columns = ['Parameters', 'Value']
        
        # Print the DataFrame
        # print(df)
    
    
    
        '''
        Train_log.log
        
        '''
        Train_log = glob.glob(os.path.join(exp_path, '*.log'))[0]
    
        data = []
        
        with open(Train_log, 'r') as file:
            lines = file.readlines()
        
        for i in range(len(lines)):
            if "Best model saved at:" in lines[i]:
            # if True:    
                info_line = lines[i].strip()  # Remove leading/trailing whitespace
                info_parts = info_line.split("Best model saved at: ")[1]
                data_parts = info_parts.split(" Epoch ")[0].split(" test loss: ")
                print([i])
                params = lines[i-1]
                print(lines[i])
                parts = params.strip().split()
                # for j in range(len(parts)):
                #     print(j, parts[j])
                
                print(parts)
                epoch = int(parts[1].split('/')[0].split('[')[1])
                # total_epochs = int(parts[2].split('/')[1].split(']')[0])
                train_loss = float(parts[4])
                test_loss = float(parts[7][:-1])  # Remove the trailing comma
                
                parts[1:]=parts[:-1]
                
                average_score = float(parts[10][:-1])  # Remove the trailing comma

                bal_average = float(parts[14][:-1])
                f1_marco = float(parts[17])
                f1_mirco = float(parts[20])
                p_marco = float(parts[23])
                p_mirco = float(parts[26])
                r_marco = float(parts[29])
                r_mirco = float(parts[32])
                roc_auc_ovo_marco = float(parts[36])
                roc_auc_ovo_mirco = float(parts[40])
                roc_auc_ovr_marco = float(parts[44])
                roc_auc_ovr_mirco = 0.#float(parts[48])
        
                info_dict = {
                    "Best model": 'yes',
                    "Epoch": epoch,
                    "Train Loss": train_loss,
                    "Test Loss": test_loss,
                    "Average Score": average_score,

                    "bal_average": bal_average,
                    "f1_marco": f1_marco,
                    "f1_mirco": f1_mirco,
                    "p_marco": p_marco,
                    "p_mirco": p_mirco,
                    "r_marco": r_marco,
                    "r_mirco": r_mirco,
                    "roc_auc_ovo_marco": roc_auc_ovo_marco,
                    "roc_auc_ovo_mirco": roc_auc_ovo_mirco,
                    "roc_auc_ovr_marco": roc_auc_ovr_marco,
                    "roc_auc_ovr_mirco": roc_auc_ovr_mirco,
                }
        
                data.append(info_dict)
        
        Train_log_df = pd.DataFrame(data)
        
        # Print the DataFrame
        # print(Train_log_df)
    
        best_model_df = pd.DataFrame([info_dict])
        best_model_df.index = [exp]

        # best_model_df = best_model_df.transpose()
        # best_model_df.reset_index(inplace=True)
        # best_model_df.columns = ['Parameters', 'Value']
        # print(best_model_df)
    
        df = pd.concat([best_model_df, df], axis=1)
        # df.reset_index(drop=True, inplace=True)
    
        # print(df)
    
        globals()[exp] = df
        df_list.append(globals()[exp])


    # if len(exp_list) > 1:
    df = pd.concat(df_list)
    
    # df = df.sort_index()
    
    selected_columns = [
    
                        'Average Score',
                        'batchsize',
                        'dropout_patch', 
                        'epoch_des',
                        'dropout_node',
                        'Train Loss',
                        'Test Loss',
                        'Epoch',
                        "bal_average",
                        "f1_marco",
                        "f1_mirco",
                        "p_marco",
                        "p_mirco",
                        "r_marco",
                        "r_mirco",
                        "roc_auc_ovo_marco",
                        "roc_auc_ovo_mirco",
                        "roc_auc_ovr_marco",
                        "roc_auc_ovr_mirco",]
    
    df = df[selected_columns]
    
    df = df.sort_values(by=['Average Score', 'batchsize'])
    
    # df.to_excel('csv/'+dataset_name+'_params_'+'.xlsx', index=True)
    df.to_csv(outpath+dataset_name+'_params_'+'.csv', index=True)
        
    
    # with pd.ExcelWriter(dataset_name+'_params_'+'.xlsx', engine='openpyxl') as writer:
    #     for exp_name in exp_list:
    #         globals()[exp_name].to_excel(writer, sheet_name=exp_name, index=False)
    
        











