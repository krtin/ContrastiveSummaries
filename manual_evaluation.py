import config
import pandas as pd
import numpy as np

from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')

smc_data = pd.read_csv(config.smc_final, usecols=['sourceid', 'judgeid', 'source', 'system', 'gtp_log_prob', 'meaning', 'grammar'], dtype={'sourceid': object, 'judgeid': object, 'source': object, 'system': object, 'gtp_log_prob': np.float64, 'meaning': np.float64, 'grammar': np.float64, })
smc_data.rename(columns={'gtp_log_prob': 'gtp_log_prob_smc'}, inplace=True)

smic_data = pd.read_csv(config.smic_final, usecols=['smic_id', 'sourceid', 'judgeid', 'smic', 'ruleid', 'gtp_log_prob'], dtype={'smic_id': np.int32, 'sourceid': object, "judgeid": object, 'smic': object, 'ruleid': np.int32, 'gtp_log_prob': np.float64, })
smic_data.rename(columns={'gtp_log_prob': 'gtp_log_prob_smic'}, inplace=True)

len_before_merge = len(smic_data)
smic_data = pd.merge(smic_data, smc_data, left_on=['sourceid', 'judgeid'], right_on=['sourceid', 'judgeid'], how='left')
smic_data['smic_better'] = (smic_data['gtp_log_prob_smc'] < smic_data['gtp_log_prob_smic']).astype(int)

if(len(smic_data)!=len_before_merge):
    raise Exception("Length before and after join unequal")

rules = smic_data['ruleid'].unique()

#this is fixed
samples_per_source = 1
#this is fixed
samples_per_source_per_smc = 1
#should be even
source_samples_per_rule = 50
total_data_per_rule = source_samples_per_rule*samples_per_source_per_smc*samples_per_source
total_rules = len(rules)
total_data = total_rules*total_data_per_rule

np.random.seed(390)
output = []
for rule in rules:
    data = smic_data[smic_data['ruleid']==rule]
    data = data[data['meaning']>=3.0]
    data = data[data['grammar']>=3.0]
    data_smic_better = data[data['smic_better']==1]
    data_smc_better = data[data['smic_better']==0]
    #get distinct source ids
    sourceids_smic_better = list(data_smic_better['sourceid'].unique())
    sourceids_smc_better = list(data_smc_better['sourceid'].unique())

    #pick source_samples_per_rule worth sourceids
    sourceids_smic_better = np.random.choice(sourceids_smic_better, int(source_samples_per_rule/2), replace=False)
    sourceids_smc_better = np.random.choice(sourceids_smc_better, int(source_samples_per_rule/2), replace=False)

    print(rule, len(data_smic_better))
    print(rule, len(data_smc_better))

    for sourceid in sourceids_smic_better:
        source_data = data[data["sourceid"]==sourceid]
        judgeids = list(source_data["judgeid"].unique())
        #select a unique random judge
        judgeid = np.random.choice(judgeids, replace=False)
        source_data = source_data[source_data["judgeid"]==judgeid]
        length = len(source_data)
        idx = np.random.choice(range(length), replace=False)
        source_data = source_data[idx:idx+1]
        source_data['better'] = 1
        smc = list(source_data['system'])[0]


        smic = list(source_data['smic'])[0]
        #smc = smc.decode('utf-8', 'ignore').lower()
        #print(smc)
        #smc = smc.encode("utf-8", 'ignore')
        smc_parse = nlp.annotate(smc, properties={
          'annotators': 'tokenize,ssplit',
          'outputFormat': 'json'
        })
        smc = []
        smic_words = smic.split(" ")
        word_counter = 0
        for sent in smc_parse["sentences"]:
            tokens = sent["tokens"]
            for token in tokens:
                smc.append(token["word"])
                if(token["word"].lower()!=smic_words[word_counter]):
                    smic_words[word_counter] = "<strong>"+smic_words[word_counter]+"</strong>"
                word_counter += 1
        smic = " ".join(smic_words)
        smc = " ".join(smc).lower().strip()

        if(len(smic.split(" "))!=len(smc.split(" "))):
            print(smic)
            print(smc)

            #sys_parse = nlp.annotate(smc, properties={
            #  'annotators': 'tokenize,ssplit',
            #  'outputFormat': 'json'
            #})
            #print(sys_parse)
            raise Exception("smic and smc length don't match")
        source_data['smic'] = smic
        #print(source_data)
        if(len(output)==0):
            output = source_data
        else:
            output = pd.concat([output, source_data])

    for sourceid in sourceids_smc_better:
        source_data = data[data["sourceid"]==sourceid]
        judgeids = list(source_data["judgeid"].unique())
        #select a unique random judge
        judgeid = np.random.choice(judgeids, replace=False)
        source_data = source_data[source_data["judgeid"]==judgeid]
        length = len(source_data)
        idx = np.random.choice(range(length), replace=False)
        source_data = source_data[idx:idx+1]
        source_data['better'] = 0
        smc = list(source_data['system'])[0]


        smic = list(source_data['smic'])[0]
        #smc = smc.decode('utf-8', 'ignore').lower()
        #print(smc)
        #smc = smc.encode("utf-8", 'ignore')
        smc_parse = nlp.annotate(smc, properties={
          'annotators': 'tokenize,ssplit',
          'outputFormat': 'json'
        })
        smc = []
        smic_words = smic.split(" ")
        word_counter = 0
        for sent in smc_parse["sentences"]:
            tokens = sent["tokens"]
            for token in tokens:
                smc.append(token["word"])
                if(token["word"].lower()!=smic_words[word_counter]):
                    smic_words[word_counter] = "<strong>"+smic_words[word_counter]+"</strong>"
                word_counter += 1
        smic = " ".join(smic_words)
        smc = " ".join(smc).lower().strip()

        if(len(smic.split(" "))!=len(smc.split(" "))):
            print(smic)
            print(smc)

            #sys_parse = nlp.annotate(smc, properties={
            #  'annotators': 'tokenize,ssplit',
            #  'outputFormat': 'json'
            #})
            #print(sys_parse)
            raise Exception("smic and smc length don't match")
        source_data['smic'] = smic

        if(len(output)==0):
            output = source_data
        else:
            output = pd.concat([output, source_data])




output = output.sample(frac=1)
output = output.reset_index()
output["system"] = output["system"].str.lower()
collisions = len(output[output["system"]==output["smic"]])
repeats = len(output) - len(output["smic"].unique())
print("Repeats: %d Collisions: %d"%(repeats, collisions))

output.drop(['index', 'judgeid', 'ruleid', 'sourceid'], axis=1, inplace=True)
final = pd.DataFrame(columns=['id', 'Source', 'SMIC', 'SMC'])
final['id'] = output['smic_id']
final['Source'] = output['source']
final['SMIC'] = output['smic']
final['SMC'] = output['system']
final['Better'] = output['better']
print(len(final))
final.to_csv("data/smic_manualeval.csv", index=False)
