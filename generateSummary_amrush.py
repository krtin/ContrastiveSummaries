import config
import pandas as pd
import numpy as np
import json
from scipy.stats import pearsonr

smc_data = pd.read_csv('data/smc_final_amrush_tout.csv', dtype={'sourceid': object, 'judgeid': object, 'smc_lmscore': np.float64, 'smc_prob': np.float64}, usecols=['sourceid', 'judgeid', 'smc_lmscore', 'smc_prob'])

classes = pd.read_csv('data/tutanova_smc_with_classes.csv', dtype={'sourceid': object, 'judgeid': object, 'nov1_classes': np.int32, 'nov2_classes': np.int32}, usecols=['sourceid', 'judgeid', 'nov1_classes', 'nov2_classes'])

prevlen=len(smc_data)
smc_data = pd.merge(smc_data, classes, left_on=['sourceid', 'judgeid'], right_on=['sourceid', 'judgeid'], how='left')

if(len(smc_data)!=prevlen):
    raise Exception("Error in joining")

smc_data = smc_data[smc_data['nov1_classes']==0]


noof_real = len(smc_data)
print("Number of Real Summaries %d"%(noof_real))
noof_source = len(smc_data['sourceid'].unique())
print("Number of Sources %d"%(noof_source))

valid_ids = pd.read_csv(config.smic_final_new, dtype={'smic_id': np.int32}, usecols=['smic_id'])
valid_ids = list(valid_ids['smic_id'].unique())

smic_data = pd.read_csv('data/smic_final_amrush_tout.csv', dtype={"judgeid": object, 'rouge1_f': np.float64, u'rouge1_p': np.float64, 'rouge1_r': np.float64, 'rouge2_f': np.float64, 'rouge2_p': np.float64, 'rouge2_r': np.float64, 'rougel_f': np.float64, 'rougel_p': np.float64, 'rougel_r': np.float64, 'ruleid': np.int32, 'smic': object, 'sourceid': object, 'smic_lmscore': np.float64, 'smic_id': np.int32, 'smic_prob': np.float64})

smic_data = smic_data[smic_data['smic_id'].isin(valid_ids)]


before_join_len = (len(smic_data))
smic_data = pd.merge(smic_data, smc_data, left_on=['sourceid', 'judgeid'], right_on=['sourceid', 'judgeid'], how='left')

if(before_join_len!=len(smic_data)):
    raise Exception('Something wrong with join')
print("Total data found %d" % before_join_len)


#smic_data = smic_data.replace([np.inf, -np.inf], np.nan)
#smic_data = smic_data.dropna(axis=0)
#smic_data = smic_data[(smic_data["gtp_log_prob_smic"]!=np.inf) & (smic_data["new_gtp_log_prob_smic"]!=np.inf) & (smic_data["gtp_avg_log_prob_smic"]!=np.inf) & (smic_data["new_gtp_avg_log_prob_smic"]!=np.inf)]

#print(smic_data.columns)
smic_data['lm_change'] = (smic_data['smc_lmscore'] - smic_data['smic_lmscore'])/smic_data['smc_lmscore']
smic_data['smic_better'] = smic_data['smc_prob'] < smic_data['smic_prob']




stats = smic_data[['sourceid', 'judgeid', 'ruleid', 'lm_change', 'smic_better']]
stats_ruleid = stats.groupby(['sourceid', 'judgeid', 'ruleid']).agg({'lm_change':['mean'], 'smic_better':['sum', 'count']})
stats_no_ruleid = stats.groupby(['sourceid', 'judgeid']).agg({'lm_change':['mean'], 'smic_better':['sum', 'count']})
stats_ruleid = stats_ruleid.reset_index()
stats_no_ruleid = stats_no_ruleid.reset_index()



print('\n##########################')
print('Overall Statistics')
print('###########################')
noof_accepted_smic = stats_ruleid['smic_better']['sum'].sum()


noof_generated_smic = stats_ruleid['smic_better']['count'].sum()

total_smcs = len(stats_no_ruleid)

print('Number of Correct Summaries %d' % total_smcs)

overall_stats=[]
index_names= []
overall_stats.append({"AMrush": noof_accepted_smic})
index_names.append('Accepted')
overall_stats.append({"AMrush": noof_generated_smic})
index_names.append('Generated')

per_accepted = round(np.float(noof_accepted_smic)/np.float(noof_generated_smic)*100., 5)

overall_stats.append({"AMrush": per_accepted})
index_names.append('Percentage Accepted')


smc_atleast1_smic = len(stats_no_ruleid[stats_no_ruleid['smic_better']['sum']>0])

overall_stats.append({"AMrush": smc_atleast1_smic})
index_names.append('Atleast 1')


per_accepted_atleast1 = round(np.float(smc_atleast1_smic)/np.float(total_smcs)*100., 5)

overall_stats.append({"AMrush": per_accepted_atleast1})
index_names.append('Percentage Atleast 1')

avg_lmchange = smic_data[smic_data['smic_better']!=False]['lm_change'].mean() - smic_data[smic_data['smic_better']!=True]['lm_change'].mean()


overall_stats.append({"AMrush": avg_lmchange})
index_names.append('LM Change')



smic_data_sorted = smic_data.sort_values(['sourceid', 'judgeid', 'lm_change'],ascending=[True, True, False], inplace=False)

bestsmic = smic_data_sorted.groupby(['sourceid', 'judgeid']).first()
bestsmic = bestsmic.reset_index()

per_best_smic = round(np.float(bestsmic['smic_better'].sum())/np.float(total_smcs)*100., 5)

overall_stats.append({"AMrush": per_best_smic})
index_names.append('Perc Best LM')


#correlation = pd.DataFrame()
amrush_change = list((smic_data['smc_prob'] - smic_data['smic_prob'])/smic_data['smc_prob'])

lm_change = list(smic_data['lm_change'])




(pearson_coeff_amrush, pvalue_amrush) = pearsonr(lm_change, amrush_change)



overall_stats.append({"AMrush": pearson_coeff_amrush})
index_names.append('Correlation')


overall_stats = pd.DataFrame(data=overall_stats, index=index_names)
pd.set_option('float_format', '{:.4f}'.format)
print(overall_stats)




print('\n##########################')
print('Rule wise Statistics')
print('###########################')



smic_accepted_rulewise = stats.groupby(['ruleid']).agg({'smic_better':['sum', 'count']})
index_names = range(1, len(smic_accepted_rulewise)+1)
column_names = ["AMrush"]



stats_per_accept = []
per_accepted_rulewise = stats_per_accept.append(list(smic_accepted_rulewise['smic_better']['sum']/smic_accepted_rulewise['smic_better']['count']*100.))

stats_per_accept = np.transpose(np.array(stats_per_accept))

stats_per_accept = pd.DataFrame(stats_per_accept, index=index_names, columns=column_names)

print('Percentage of Accepted SMICs')
print(stats_per_accept)

#####################################################

print('\nPercentage Contribution')
stats_per_cont = []
per_cont = stats_per_cont.append(list(smic_accepted_rulewise['smic_better']['sum']/noof_accepted_smic*100.))

stats_per_cont = np.transpose(np.array(stats_per_cont))
stats_per_cont = pd.DataFrame(stats_per_cont, index=index_names, columns=column_names)
print(stats_per_cont)

#######################################################

stats_ruleid['atleast1_smic'] = stats_ruleid['smic_better']['sum']>0


smc_atleast1_smic_rulewise = stats_ruleid.groupby(['ruleid']).agg({'atleast1_smic':['sum', 'count']})

stats_per_atleast1 = []
per_atleast1 = stats_per_atleast1.append(list(smc_atleast1_smic_rulewise['atleast1_smic']['sum']/smc_atleast1_smic_rulewise['atleast1_smic']['count']*100.))



print('\nPercentage of SMC with at least 1 accepted SMIC')

stats_per_atleast1 = np.transpose(np.array(stats_per_atleast1))
stats_per_atleast1 = pd.DataFrame(stats_per_atleast1, index=index_names, columns=column_names)
print(stats_per_atleast1)


############################################################



print('\n##########################')
print('Bucketing')
print('###########################')
bucket_stat = pd.DataFrame()
bucket_stat['accepted_smic'] = stats_no_ruleid['smic_better']['sum']
bucket_stat['generated_smic'] = stats_no_ruleid['smic_better']['count']
bucket_stat.loc[bucket_stat['accepted_smic'] >= 20, ['accepted_smic']] = 20
bucket_stat = bucket_stat.groupby(['accepted_smic']).agg({'accepted_smic':['sum'], 'generated_smic':['sum', 'count']})

index_names_buckets = bucket_stat.index


stats_per_smc = []
stats_per_smc.append(list(bucket_stat['generated_smic']['count']/total_smcs*100.))


print('\nPercentage of SMCs with SMIC count bucketing')
stats_per_smc = np.transpose(np.array(stats_per_smc))
stats_per_smc = pd.DataFrame(stats_per_smc, index=index_names_buckets, columns=column_names)
print(stats_per_smc)

stats_per_accepted = []
stats_per_accepted.append(list(bucket_stat['accepted_smic']['sum']/bucket_stat['generated_smic']['sum']*100.))


print('\nPercentage of Accepted SMICs with SMIC count bucketing')
stats_per_accepted = np.transpose(np.array(stats_per_accepted))
stats_per_accepted = pd.DataFrame(stats_per_accepted, index=index_names_buckets, columns=column_names)
print(stats_per_accepted)
