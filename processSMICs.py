import pandas as pd
import config
import os
import time
import lm
import pickle as pkl
import numpy as np

#############################

# This file will add LM scores for the generated SMICs

############################
starttime = time.time()


if(os.path.exists(config.smicfile+'_lm.pkl') is False):
    print("Generating LM scores")
    lm.generateLMScore(config.smicfile, config.smicfile+'_lm.pkl')

print("Loading LM scores")
smicprobs = pkl.load(open(config.smicfile+'_lm.pkl', 'rb'))
print('Added %d smic scores' % (len(smicprobs)))

#read data file
data = pd.read_csv(config.smiccorpus, dtype={"judgeid": object, 'rouge1_f': np.float64, u'rouge1_p': np.float64, 'rouge1_r': np.float64, 'rouge2_f': np.float64, 'rouge2_p': np.float64, 'rouge2_r': np.float64, 'rougel_f': np.float64, 'rougel_p': np.float64, 'rougel_r': np.float64, 'ruleid': np.int32, 'smic': object, 'sourceid': object, 'smic_lmscore': np.float64})

#verify if the length is equal
if(len(smicprobs)!=len(data)):
    raise Exception("Data length of SMIC corpus and LM scores don't match")

#add the generated scores to data and write to file
data['smic_lmscore'] = pd.Series(smicprobs, index=data.index)
data['smic_id'] = data.index
data.to_csv(config.smiccorpus_with_lm, index=False)
#print(data)

endtime = time.time()

print("LM scores added to SMICs it took: %s" % (time.strftime("%H:%M:%S", time.gmtime(endtime - starttime))))
