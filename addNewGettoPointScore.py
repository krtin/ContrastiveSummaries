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

smc_data = pd.read_csv(config.smc_final)
#print(len(smc_data))


prob_smc_data = pkl.load(open(config.new_gtp_smc_probfile, 'rb'))

#print(len(prob_smc_data))
if(len(smc_data)!=len(prob_smc_data)):
    raise Exception("Number of generated probabilities and number of smcs are not equal")
else:
    #print(smc_data.columns)
    print('Adding GTP probabilities to smc data')
    prob_smc_data = pd.DataFrame(prob_smc_data, columns=['counter', 'log_prob', 'avg_log_prob', 'gtp_target', 'gtp_output'])
    #print(len(prob_smc_data))
    #print(prob_smc_data['gtp_target'])
    smc_data['new_gtp_oov_words'] = prob_smc_data['gtp_target']==prob_smc_data['gtp_output']
    smc_data['new_gtp_log_prob'] = prob_smc_data['log_prob']
    smc_data['new_gtp_avg_log_prob'] = prob_smc_data['avg_log_prob']
    #print(smc_data['gtp_log_prob'])
    #print(prob_smc_data['log_prob'])
    #print(smc_data['system'])
    #print(prob_smc_data['gtp_target'])

    smc_data.to_csv(config.smc_final, index=False)


print('Reading SMIC probability data')

probfiles = [f for f in os.listdir(config.new_gtp_smic_probdir) if os.path.isfile(os.path.join(config.new_gtp_smic_probdir, f))]

prob_smic_data = {}

for filename in probfiles:
    filename_parts = filename.split('_')

    if(len(filename_parts)==3):
        filename_parts[2] = filename_parts[2].split('.')[0]
        if(is_number(filename_parts[1]) and is_number(filename_parts[2])):
            start = int(filename_parts[1])
            end = int(filename_parts[2])
            prob_smic_data[start] = pkl.load(open(os.path.join(config.new_gtp_smic_probdir, filename), 'rb'))

tmp_data = []
for key in sorted(prob_smic_data.iterkeys()):
    #if(key>810000):
        #tmp = pd.DataFrame(prob_smic_data[key], columns=['counter', 'gtp_log_prob', 'gtp_avg_log_prob', 'gtp_target', 'gtp_output'])
        #print(tmp)
        #print('Null count', len(tmp['counter'].isnull()))
        #break
    #print(key)
    print(key, len(prob_smic_data[key]))
    tmp_data.extend(prob_smic_data[key])

prob_smic_data = tmp_data
del tmp_data

print(len(prob_smic_data))

print('Reading SMIC data')
smic_data = pd.read_csv(config.smic_final)
#print(len(smic_data))

f = open(config.smicidfile_filtered_new_gtp, 'r')
smicids = f.read().split('\n')
f.close()
orig_len = len(smicids)
#prob_smic_data = prob_smic_data[0:orig_len]
smicids = smicids[0:len(prob_smic_data)]
cut_len = len(smicids)
print('Total available data: %d out of %d' % (cut_len, orig_len))

prob_smic_data = pd.DataFrame(prob_smic_data, columns=['counter', 'new_gtp_log_prob', 'new_gtp_avg_log_prob', 'gtp_target', 'gtp_output'])
prob_smic_data['smic_id'] = np.array(smicids).astype(int)
prob_smic_data['new_oov_words'] = prob_smic_data['gtp_target']!=prob_smic_data['gtp_output']
del prob_smic_data['gtp_target']
del prob_smic_data['gtp_output']
del prob_smic_data['counter']

orig_smic_len = len(smic_data)
smic_data = pd.merge(smic_data, prob_smic_data, left_on='smic_id', right_on='smic_id', how='left')
prob_added_len = smic_data['new_gtp_log_prob'].count()

if(prob_added_len!=(cut_len)):
    raise Exception("Some ids are not matching during join")
if(len(smic_data)!=orig_smic_len):
    raise Exception("Some issue in joining")
#print(smic_data)
smic_data.to_csv(config.smic_final, index=False)



'''
#get to the point smic scores
print('Reading SMIC data and its probabilities')
smic_data = pd.read_csv(config.smiccorpus_filtered)
print(smic_data.columns)
prob_smic_data = pkl.load(open(config.gtp_smic_probfile, 'rb'))
f = open(config.smicidfile_filtered_gtp, 'r')
smicids = f.read().split('\n')
f.close()

if(len(smicids)==len(prob_smic_data) and len(smicids)==len(smic_data)):
    prob_smic_data = pd.DataFrame(prob_smic_data, columns=['counter', 'gtp_log_prob', 'gtp_avg_log_prob', 'gtp_target', 'gtp_output'])
    prob_smic_data['smic_id'] = np.array(smicids).astype(int)
    prob_smic_data['oov_words'] = prob_smic_data['gtp_target']==prob_smic_data['gtp_output']
    del prob_smic_data['gtp_target']
    del prob_smic_data['gtp_output']
    del prob_smic_data['counter']
    #print(len(smic_data))
    smic_data = pd.merge(smic_data, prob_smic_data, left_on='smic_id', right_on='smic_id', how='left')
    #print(len(smic_data))
    #print(smic_data['gtp_log_prob'].count())
    if(smic_data['gtp_log_prob'].count()!=len(smic_data)):
        raise Exception("Some ids are not matching during join")
    #print(smic_data)
    smic_data.to_csv(config.smic_final, index=False)
    #print(prob_smic_data)
else:
    raise Exception("SMIC length does not match")
'''
