import pandas as pd
from rouge import Rouge
import config
import os
import time
import lm
import pickle as pkl
import numpy as np

def getRougeScore_todict(gold, sys):
    rouge = Rouge()
    scores = rouge.get_scores(gold, sys)
    scores = scores[0]
    scores = pd.Series({'rouge1_f':scores['rouge-1']['f'], 'rouge1_p':scores['rouge-1']['p'], 'rouge1_r':scores['rouge-1']['r'], 'rouge2_f':scores['rouge-2']['f'], 'rouge2_p':scores['rouge-2']['p'], 'rouge2_r':scores['rouge-2']['r'], 'rougel_f':scores['rouge-l']['f'], 'rougel_p':scores['rouge-l']['p'], 'rougel_r':scores['rouge-l']['r'] })
    return scores

def processRawData(filename, processedfilename):
    cols = ['sourceid', 'category', 'source', 'summary', 'judgeid', 'meaning', 'grammar', 'sents', 'rank']
    data = pd.read_csv(filename, names=cols, index_col=0)
    goldset = data[data['rank']==1][['sourceid','summary']]
    print("Number of Gold Standard found %d"%(len(goldset)))
    print("Number of Gold Standard found %d"%(len(goldset['sourceid'].unique())))
    goldset.columns = ['sourceid', 'gold']
    systemset = data[data['rank']!=1][['sourceid','summary', 'source', 'category', 'judgeid', 'meaning', 'grammar', 'sents', 'rank']]

    systemset.columns = ['sourceid', 'system', 'source', 'category', 'judgeid', 'meaning', 'grammar', 'sents', 'rank']

    data = pd.merge(systemset, goldset, how='inner', on='sourceid')
    print("Number of Gold Standard after merge %d"%(len(data['sourceid'].unique())))
    data = data.merge(data.apply(lambda row: getRougeScore_todict(row['gold'], row['system']), axis=1), left_index=True, right_index=True)
    #print(data)
    #data.to_csv(processedfilename)

processRawData(config.rawCorpus, config.processedCorpus)
'''
starttime = time.time()

#process data if not already processed or force_process option is set to true
if(os.path.exists(config.processedCorpus) is False or config.force_process):
    print('Processing Raw data and writing to processed file')
    #process raw corpus to specific format for smic generator
    processRawData(config.rawCorpus, config.processedCorpus)

print("Reading processed file")
#read processed file
data = pd.read_csv(config.processedCorpus, index_col=0)
data.sort_values('sourceid', inplace=True)
print(data)


print("Extracting gold and system summaries from processed file")
#write gold and smcs/system summaries to file separated by newline
#this file will be used to get the lmscores
prevsourceid = -1
system_sents = []
gold_sents = []
unique_sourceids = []

for i, row in data.iterrows():
    system = row['system']
    gold = row['gold']
    sourceid = row['sourceid']

    if(prevsourceid!=sourceid):
        #if its a new sourceid then add the gold to list
        gold_sents.append(gold)
        unique_sourceids.append(sourceid)

    #always add system to list
    system_sents.append(system)

    prevsourceid = sourceid



#just do some verifications before writing to file
noof_distinct = data['sourceid'].nunique()

if(noof_distinct != len(gold_sents)):
    raise Exception('Length of distinct source ids and number of gold sentences should be equal')

print("Writing gold and system summaries to files")

with open(config.smcfile, 'w') as f:
    f.write('\n'.join(system_sents))

with open(config.goldfile, 'w') as f:
    f.write('\n'.join(gold_sents))



print("Generating LM scores if required")
if(os.path.exists(config.goldfile+'_lm.pkl') is False):
    lm.generateLMScore(config.goldfile, config.goldfile+'_lm.pkl')

if(os.path.exists(config.smcfile+'_lm.pkl') is False):
    lm.generateLMScore(config.smcfile, config.smcfile+'_lm.pkl')

print('Adding LM scores to Input file')
goldprobs = pkl.load(open(config.goldfile+'_lm.pkl', 'rb'))

smcprobs = pkl.load(open(config.smcfile+'_lm.pkl', 'rb'))
print('Added %d gold scores and %d smc scores' % (len(goldprobs), len(smcprobs)))

#add the generated scores to data and write to file
data['smc_lmscore'] = pd.Series(smcprobs, index=data.index)
golddata = pd.DataFrame(np.array([unique_sourceids, goldprobs]).transpose(), columns=['sourceid', 'gold_lmscore'])
data = pd.merge(data, golddata, left_on='sourceid', right_on='sourceid', how='left')

#print(data)
data.to_csv(config.preprocessed_with_lm, index=False)

with open(config.smcfile, 'w') as f:
    f.write('\n'.join(list(data["system"])))

with open(config.sourcefile, 'w') as f:
    f.write('\n'.join(list(data["source"])))

endtime = time.time()

print("Preprocessing Step Completed it took: %s" % (time.strftime("%H:%M:%S", time.gmtime(endtime - starttime))))

'''
