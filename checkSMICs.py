import config
import pandas as pd
import os
import numpy as np
import random

def printmenu():
    print('You can choose from the following options')
    print('1. Do some Manual Checking')
    print('2. Show invalid SMICs')
    print('3. Exit')

def show_record(record, smc_data):
    record = record.to_dict(orient='records')
    record = record[0]
    judgeid = record['judgeid']
    sourceid = record['sourceid']

    smc_record = smc_data[((smc_data['judgeid']==judgeid) & (smc_data['sourceid']==sourceid))]
    smc_record = smc_record.to_dict(orient='records')

    if(len(smc_record)>1):
        raise Exception('More than one value find in SMC corpus')
    smc_record = smc_record[0]

    print('\nRule id: %d' % (record['ruleid']))
    print('\nSOURCE')
    print(smc_record['source'])
    print('\nSMIC')
    print(record['smic'])
    print('\nSMC')
    print(smc_record['system'])
    print('\n')

def manualcheck(smic_data):
    #number of smics which need to be labelled for each rule
    goal = np.array([5, 6, 6, 5, 5, 5, 5, 5])
    progress = smic_data[['ruleid', 'valid_smic']]
    progress = progress.groupby('ruleid').count()
    print(progress)
    smc_data = pd.read_csv(config.smc_final, usecols=['sourceid', 'judgeid', 'system', 'source', 'gold'], dtype={'sourceid': object, 'judgeid': object, 'system': object, 'source': object, 'gold': object})

    current_prog = goal - np.array(list(progress['valid_smic']))
    #remove already labelled data

    data = smic_data[smic_data['valid_smic'].isnull()]
    #print(data['valid_smic'])
    ruleid = 1
    quit = 0
    for rule_prog in current_prog:

        if(rule_prog==0):
            ruleid += 1
            continue

        ids = list(data[data['ruleid']==ruleid]['smic_id'])
        selectedids = random.sample(ids, rule_prog)

        for smic_id in selectedids:
            selected_record = data[data['smic_id']==smic_id]


            #print(selected_record)
            show_record(selected_record, smc_data)

            option_sel = raw_input("Enter 0 if invalid or 1 if valid (anything else to go back to menu): ")
            if(option_sel=='1'):
                print('Labeling as 1')
                smic_data.loc[smic_data.smic_id == smic_id, 'valid_smic'] = 1
                #smic_data[smic_data['smic_id']==smic_id]['valid_smic'] = 1
            elif(option_sel=='0'):
                print('Labeling as 0')
                smic_data.loc[smic_data.smic_id == smic_id, 'valid_smic'] = 0
                #smic_data[smic_data['smic_id']==smic_id]['valid_smic'] = 0
            else:
                print('Quitting without Labeling')
                quit = 1
                break
        if(quit):
            break
        ruleid += 1

    #save file in the end
    smic_data.to_csv(config.non_differentiable_smic_manual_check, index=False, na_rep='')

def showInvalidSMICs(smic_data):
    smc_data = pd.read_csv(config.smc_final, usecols=['sourceid', 'judgeid', 'system', 'source', 'gold'], dtype={'sourceid': object, 'judgeid': object, 'system': object, 'source': object, 'gold': object})
    invalid_smics = smic_data[smic_data['valid_smic']==0]
    before_len = (len(invalid_smics))
    invalid_smics = pd.merge(invalid_smics, smc_data, left_on=['sourceid', 'judgeid'], right_on=['sourceid', 'judgeid'], how='left')
    after_len = (len(invalid_smics))

    if(before_len!=after_len):
        raise Exception('Problem in merging')
    print('Total invalid SMICs are %d', after_len)

    for i, row in invalid_smics.iterrows():
        print('SMC sourceid: %s judgeid: %s ruleid: %d'%(row['sourceid'], row['judgeid'], row['ruleid']))
        print('SMC: %s' % row['system'])
        print('SMIC: %s' % row['smic'])
        print('\n')

print('Reading Data from File')
#check if some records were manually verified earlier or not
#if not Initialize a file
if(os.path.exists(config.non_differentiable_smic_manual_check) is False):
    smic_data = pd.read_csv(config.non_differentiable_smic, dtype={"judgeid": object, 'rouge1_f': np.float64, u'rouge1_p': np.float64, 'rouge1_r': np.float64, 'rouge2_f': np.float64, 'rouge2_p': np.float64, 'rouge2_r': np.float64, 'rougel_f': np.float64, 'rougel_p': np.float64, 'rougel_r': np.float64, 'ruleid': np.int32, 'smic': object, 'sourceid': object, 'smic_lmscore': np.float64, 'smic_id': np.int32, 'gtp_log_prob_smic': np.float64, 'gtp_avg_log_prob_smic': np.float64, 'oov_words': np.bool_, 'smc_lmscore': np.float64, 'gtp_log_prob_smc': np.float64, 'gtp_avg_log_prob_smc': np.float64, 'smic_better': np.bool_, 'smic_better_avg':np.bool_})
    #valid smic will contain manual verification of smic
    #default values will be NaN which means no verification done
    #a value 1 indicates a valid smic
    #a value 0 indicates an invalid smic
    smic_data['valid_smic'] = np.nan
    print(len(smic_data))
    #print(smic_data)
else:
    #read previous data
    smic_data = pd.read_csv(config.non_differentiable_smic_manual_check, dtype={"judgeid": object, 'rouge1_f': np.float64, u'rouge1_p': np.float64, 'rouge1_r': np.float64, 'rouge2_f': np.float64, 'rouge2_p': np.float64, 'rouge2_r': np.float64, 'rougel_f': np.float64, 'rougel_p': np.float64, 'rougel_r': np.float64, 'ruleid': np.int32, 'smic': object, 'sourceid': object, 'smic_lmscore': np.float64, 'smic_id': np.int32, 'gtp_log_prob_x': np.float64, 'gtp_avg_log_prob_x': np.float64, 'oov_words': np.bool_, 'smc_lmscore': np.float64, 'gtp_log_prob_y': np.float64, 'gtp_avg_log_prob_y': np.float64, 'non_d': np.bool_, 'non_d_avg': np.bool_, 'valid_smic': np.float64})
    print(len(smic_data))
    #print(smic_data[smic_data['valid_smic']==0])
exit_code = False

while(exit_code is False):
    printmenu()
    option_sel = raw_input("Choose from [1 to 3]: ")

    if(option_sel=='1'):
        manualcheck(smic_data)
    elif(option_sel=='2'):
        #show invalid smics
        showInvalidSMICs(smic_data)


    elif(option_sel=='3'):
        print('Exiting code')
        exit_code = True
    else:
        print('Wrong option selected')
