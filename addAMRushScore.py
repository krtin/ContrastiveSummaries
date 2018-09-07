import config
import pandas as pd
import pickle as pkl
import numpy as np
import os
import unicodedata

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


#get to the point smc scores

print('Reading SMC data and its probabilities')

smc_data = pd.read_csv(config.preprocessed_with_lm)
#print(len(smc_data))

smc_prob_file = "../summarization_models/namas_trained/probs/smc_toutanova.txt"

with open(smc_prob_file, 'r') as f:
    prob_smc_data = f.read().split('\n')

if(prob_smc_data[-1]==''):
    prob_smc_data.pop()


#print(len(prob_smc_data))
if(len(smc_data)!=len(prob_smc_data)):
    raise Exception("Number of generated probabilities and number of smcs are not equal")
else:
    #print(smc_data.columns)
    print('Adding GTP probabilities to smc data')

    smc_data['smc_prob'] = prob_smc_data


    smc_data.to_csv('data/smc_final_amrush_tout.csv', index=False)


print('Reading SMIC probability data')

totalfiles = 3
smic_prob_file = "../summarization_models/namas_trained/probs/smic_toutanova_"

probs_smic = []

for fileno in range(totalfiles):
    current_file = smic_prob_file+str(fileno+1)+".txt"
    with open(current_file, 'r') as f:
        data = f.read().split('\n')

    if(data[-1]==''):
        data.pop()

    probs_smic.extend(data)



print(len(probs_smic))

print('Reading SMIC data')
smic_data = pd.read_csv(config.smiccorpus_filtered)
#print(len(smic_data))

f = open(config.smicidfile_filtered_gtp, 'r')
smicids = f.read().split('\n')
f.close()
smic_ids_len = len(smicids)

if(smic_ids_len!=len(probs_smic)):
    raise Exception("Length of smicids and generated smic probs are not equal")

prob_smic_data = pd.DataFrame()
prob_smic_data['smic_id'] = np.array(smicids).astype(int)
prob_smic_data['smic_prob'] = probs_smic

#print(prob_smic_data)


orig_smic_len = len(smic_data)

if((orig_smic_len)!=len(prob_smic_data)):
    raise Exception("smic data length is not equal to the probs length in dataframe")

smic_data = pd.merge(smic_data, prob_smic_data, left_on='smic_id', right_on='smic_id', how='left')



if(len(smic_data)!=orig_smic_len):
    raise Exception("Some issue in joining")
#print(smic_data)
smic_data.to_csv('data/smic_final_amrush_tout.csv', index=False)
