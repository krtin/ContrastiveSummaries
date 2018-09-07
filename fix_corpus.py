from pycorenlp import StanfordCoreNLP
import pandas as pd
import numpy as np
import config

nlp = StanfordCoreNLP('http://localhost:9000')

smc_data = pd.read_csv(config.smc_final, dtype={'sourceid': object, 'judgeid': object, 'system': object, 'source': object}, usecols=['sourceid', 'judgeid', 'system', 'source'])

smic_data = pd.read_csv(config.smic_final, dtype={"judgeid": object, 'rouge1_f': np.float64, u'rouge1_p': np.float64, 'rouge1_r': np.float64, 'rouge2_f': np.float64, 'rouge2_p': np.float64, 'rouge2_r': np.float64, 'rougel_f': np.float64, 'rougel_p': np.float64, 'rougel_r': np.float64, 'ruleid': np.int32, 'smic': object, 'sourceid': object, 'smic_lmscore': np.float64, 'smic_id': np.int32, 'gtp_log_prob': np.float64, 'gtp_avg_log_prob': np.float64, 'oov_words': np.bool_, 'new_gtp_log_prob': np.float64, 'new_gtp_avg_log_prob': np.float64, 'new_gtp_oov_words': np.bool_, 'new_rev_gtp_log_prob': np.float64, 'new_rev_gtp_avg_log_prob': np.float64, 'new_rev_gtp_oov_words': np.bool_})

sourceids_unique = list(smc_data['sourceid'].unique())

def tokenize(sents):
    parse = nlp.annotate(sents, properties={
      'annotators': 'tokenize,ssplit',
      'outputFormat': 'json'
    })

    sents = []
    for sent in parse["sentences"]:
        tokens = sent["tokens"]
        for token in tokens:

            sents.append(token["word"])
    sents = " ".join(sents)

    return sents
error_counts = 0
counter=0
save_len = len(smic_data)
df = []
for i, row in smc_data.iterrows():
        sourceid = row['sourceid']
        judgeid = row['judgeid']

        #source = tokenize(row['source']).lower()
        smc = tokenize(row['system']).lower()
        smics = smic_data[smic_data['sourceid']==str(sourceid)]
        smics = smics[smics['judgeid']==str(judgeid)]

        for j, smic_row in smics.iterrows():
            counter += 1
            smic = smic_row['smic']
            if(smc==str(smic)):
                error_counts+=1
                #smic_data = smic_data.drop(smic_data.index[j])
                #print(j)
                #print("new length", len(smic_data))
                continue

            df.append(smic_row.to_dict())





        print('\r Completed %d out of %d errors %d'%(counter, save_len, error_counts), end='')

smic_data = pd.DataFrame(df)
print('Original Length: %d New Length: %d Dropped Length: %d'%(save_len, len(smic_data), error_counts ))

smic_data.to_csv(config.smic_final_new, index=False)
