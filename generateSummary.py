import config
import pandas as pd
import numpy as np
import json
from scipy.stats import pearsonr

smc_data = pd.read_csv('data/tutanova_smc_with_classes.csv', dtype={'sourceid': object, 'judgeid': object, 'smc_lmscore': np.float64, 'gtp_log_prob': np.float64, 'gtp_avg_log_prob': np.float64, 'new_gtp_log_prob': np.float64, 'new_gtp_avg_log_prob': np.float64, 'new_rev_gtp_log_prob': np.float64, 'new_rev_gtp_avg_log_prob': np.float64, 'nov1_classes':np.int32, 'nov2_classes':np.int32 }, usecols=['sourceid', 'judgeid', 'smc_lmscore', 'gtp_log_prob', 'gtp_avg_log_prob', 'new_gtp_log_prob', 'new_gtp_avg_log_prob', 'new_rev_gtp_log_prob', 'new_rev_gtp_avg_log_prob', 'nov1_classes', 'nov2_classes'])
print(smc_data)
smc_data = smc_data[smc_data['nov1_classes']==1]

#smc_data = pd.read_csv(config.smc_final, dtype={'sourceid': object, 'judgeid': object, 'system': object, 'source': object, 'gold': object})
#Index(['sourceid', 'system', 'source', 'category', 'judgeid', 'meaning',
#       'grammar', 'sents', 'rank', 'gold', 'rouge1_f', 'rouge1_p', 'rouge1_r',
#       'rouge2_f', 'rouge2_p', 'rouge2_r', 'rougel_f', 'rougel_p', 'rougel_r',
#       'smc_lmscore', 'gold_lmscore', 'oov_words', 'gtp_log_prob',
#       'gtp_avg_log_prob'],
#      dtype='object')
smc_data.rename(columns={'gtp_log_prob': 'gtp_log_prob_smc', 'gtp_avg_log_prob': 'gtp_avg_log_prob_smc', 'new_gtp_log_prob': 'new_gtp_log_prob_smc', 'new_gtp_avg_log_prob': 'new_gtp_avg_log_prob_smc', 'new_rev_gtp_log_prob': 'new_rev_gtp_log_prob_smc', 'new_rev_gtp_avg_log_prob': 'new_rev_gtp_avg_log_prob_smc'}, inplace=True)
noof_real = len(smc_data)
print("Number of Real Summaries %d"%(noof_real))
noof_source = len(smc_data['sourceid'].unique())
print("Number of Sources %d"%(noof_source))

smic_data = pd.read_csv(config.smic_final_new, dtype={"judgeid": object, 'rouge1_f': np.float64, u'rouge1_p': np.float64, 'rouge1_r': np.float64, 'rouge2_f': np.float64, 'rouge2_p': np.float64, 'rouge2_r': np.float64, 'rougel_f': np.float64, 'rougel_p': np.float64, 'rougel_r': np.float64, 'ruleid': np.int32, 'smic': object, 'sourceid': object, 'smic_lmscore': np.float64, 'smic_id': np.int32, 'gtp_log_prob': np.float64, 'gtp_avg_log_prob': np.float64, 'oov_words': np.bool_, 'new_gtp_log_prob': np.float64, 'new_gtp_avg_log_prob': np.float64, 'new_gtp_oov_words': np.bool_, 'new_rev_gtp_log_prob': np.float64, 'new_rev_gtp_avg_log_prob': np.float64, 'new_rev_gtp_oov_words': np.bool_})

smic_data = smic_data[smic_data['gtp_log_prob'].notnull()]
smic_data = smic_data[smic_data['new_gtp_log_prob'].notnull()]
smic_data = smic_data[smic_data['new_rev_gtp_log_prob'].notnull()]

smic_data.rename(columns={'gtp_log_prob': 'gtp_log_prob_smic', 'gtp_avg_log_prob': 'gtp_avg_log_prob_smic', 'new_gtp_log_prob': 'new_gtp_log_prob_smic', 'new_gtp_avg_log_prob': 'new_gtp_avg_log_prob_smic', 'new_rev_gtp_log_prob': 'new_rev_gtp_log_prob_smic', 'new_rev_gtp_avg_log_prob': 'new_rev_gtp_avg_log_prob_smic'}, inplace=True)

before_join_len = (len(smic_data))
smic_data = pd.merge(smic_data, smc_data, left_on=['sourceid', 'judgeid'], right_on=['sourceid', 'judgeid'], how='left')

if(before_join_len!=len(smic_data)):
    raise Exception('Something wrong with join')
print("Total data found %d" % before_join_len)


smic_data = smic_data.replace([np.inf, -np.inf], np.nan)
smic_data = smic_data.dropna(axis=0)
#smic_data = smic_data[(smic_data["gtp_log_prob_smic"]!=np.inf) & (smic_data["new_gtp_log_prob_smic"]!=np.inf) & (smic_data["gtp_avg_log_prob_smic"]!=np.inf) & (smic_data["new_gtp_avg_log_prob_smic"]!=np.inf)]

#print(smic_data.columns)
smic_data['lm_change'] = (smic_data['smc_lmscore'] - smic_data['smic_lmscore'])/smic_data['smc_lmscore']
smic_data['smic_better'] = smic_data['gtp_log_prob_smc'] < smic_data['gtp_log_prob_smic']
smic_data['new_smic_better'] = smic_data['new_gtp_log_prob_smc'] < smic_data['new_gtp_log_prob_smic']

list_new_gtp_smic_better = np.array(list(smic_data['new_smic_better'])).astype(int)
list_gtp_smic_better = np.array(list(smic_data['smic_better'])).astype(int)

print(list_new_gtp_smic_better)

diff = list_new_gtp_smic_better - list_gtp_smic_better

print(np.where(diff>0)[0])
gtp_was_right = len(np.where(diff>0)[0])
padded_gtp_was_right = len(np.where(diff<0)[0])

disagree_count = np.count_nonzero(diff)
agree_count = len(diff) - disagree_count

print("Between GTP and Padded GTP Agreement: %d, Disgreement:%d out of %d"%(agree_count, disagree_count, len(diff)))

print("Original GTP was better in %d cases and Padded GTP was better in %d cases"%(gtp_was_right, padded_gtp_was_right))


#smic_data['new_smic_better'] = smic_data['new_gtp_log_prob_smic'] - smic_data['new_gtp_log_prob_smc'] < smic_data['new_rev_gtp_log_prob_smic'] - smic_data['new_rev_gtp_log_prob_smc']


smic_data['smic_better_avg'] = smic_data['gtp_avg_log_prob_smc'] < smic_data['gtp_avg_log_prob_smic']
smic_data['new_smic_better_avg'] = smic_data['new_gtp_avg_log_prob_smc'] < smic_data['new_gtp_avg_log_prob_smic']


stats = smic_data[['sourceid', 'judgeid', 'ruleid', 'lm_change', 'smic_better', 'smic_better_avg', 'new_smic_better', 'new_smic_better_avg']]
stats_ruleid = stats.groupby(['sourceid', 'judgeid', 'ruleid']).agg({'lm_change':['mean'], 'smic_better':['sum', 'count'], 'smic_better_avg':['sum', 'count'], 'new_smic_better':['sum', 'count'],'new_smic_better_avg':['sum', 'count']})
stats_no_ruleid = stats.groupby(['sourceid', 'judgeid']).agg({'lm_change':['mean'], 'smic_better':['sum', 'count'], 'smic_better_avg':['sum', 'count'], 'new_smic_better':['sum', 'count'], 'new_smic_better_avg':['sum', 'count']})
stats_ruleid = stats_ruleid.reset_index()
stats_no_ruleid = stats_no_ruleid.reset_index()


non_differentiable_smic = smic_data[smic_data['smic_better_avg']==True]
non_differentiable_smic.to_csv(config.non_differentiable_smic, index=False)



print('\n##########################')
print('Overall Statistics')
print('###########################')
noof_accepted_smic = stats_ruleid['smic_better']['sum'].sum()
noof_accepted_smic_avg = stats_ruleid['smic_better_avg']['sum'].sum()
noof_accepted_new_smic = stats_ruleid['new_smic_better']['sum'].sum()
noof_accepted_new_smic_avg = stats_ruleid['new_smic_better_avg']['sum'].sum()



noof_generated_smic = stats_ruleid['smic_better']['count'].sum()
noof_generated_smic_avg = stats_ruleid['smic_better_avg']['count'].sum()
noof_generated_new_smic = stats_ruleid['new_smic_better']['count'].sum()
noof_generated_new_smic_avg = stats_ruleid['new_smic_better_avg']['count'].sum()

total_smcs = len(stats_no_ruleid)

print('Number of Correct Summaries %d' % total_smcs)

overall_stats=[]
index_names= []
overall_stats.append({"GTP": noof_accepted_smic, "New GTP": noof_accepted_new_smic, "GTP (avg)": noof_accepted_smic_avg, "New GTP (avg)": noof_accepted_new_smic_avg})
index_names.append('Accepted')
overall_stats.append({"GTP": noof_generated_smic, "New GTP": noof_generated_new_smic, "GTP (avg)": noof_generated_smic_avg, "New GTP (avg)": noof_generated_new_smic_avg})
index_names.append('Generated')

per_accepted = round(np.float(noof_accepted_smic)/np.float(noof_generated_smic)*100., 5)
per_accepted_avg = round(np.float(noof_accepted_smic_avg)/np.float(noof_generated_smic_avg)*100., 5)
per_accepted_new = round(np.float(noof_accepted_new_smic)/np.float(noof_generated_new_smic)*100., 5)
per_accepted_new_avg = round(np.float(noof_accepted_new_smic_avg)/np.float(noof_generated_new_smic_avg)*100., 5)

overall_stats.append({"GTP": per_accepted, "New GTP": per_accepted_new, "GTP (avg)": per_accepted_avg, "New GTP (avg)": per_accepted_new_avg})
index_names.append('Percentage Accepted')


smc_atleast1_smic = len(stats_no_ruleid[stats_no_ruleid['smic_better']['sum']>0])
smc_atleast1_smic_avg = len(stats_no_ruleid[stats_no_ruleid['smic_better_avg']['sum']>0])
smc_atleast1_new_smic = len(stats_no_ruleid[stats_no_ruleid['new_smic_better']['sum']>0])
smc_atleast1_new_smic_avg = len(stats_no_ruleid[stats_no_ruleid['new_smic_better_avg']['sum']>0])

overall_stats.append({"GTP": smc_atleast1_smic, "New GTP": smc_atleast1_new_smic, "GTP (avg)": smc_atleast1_smic_avg, "New GTP (avg)": smc_atleast1_new_smic_avg})
index_names.append('Atleast 1')


per_accepted_atleast1 = round(np.float(smc_atleast1_smic)/np.float(total_smcs)*100., 5)
per_accepted_atleast1_avg = round(np.float(smc_atleast1_smic_avg)/np.float(total_smcs)*100., 5)
per_accepted_atleast1_new = round(np.float(smc_atleast1_new_smic)/np.float(total_smcs)*100., 5)
per_accepted_atleast1_new_avg = round(np.float(smc_atleast1_new_smic_avg)/np.float(total_smcs)*100., 5)

overall_stats.append({"GTP": per_accepted_atleast1, "New GTP": per_accepted_atleast1_new, "GTP (avg)": per_accepted_atleast1_avg, "New GTP (avg)": per_accepted_atleast1_new_avg})
index_names.append('Percentage Atleast 1')

avg_lmchange = smic_data[smic_data['smic_better']!=False]['lm_change'].mean() - smic_data[smic_data['smic_better']!=True]['lm_change'].mean()
avg_lmchange_avg = smic_data[smic_data['smic_better_avg']!=False]['lm_change'].mean() - smic_data[smic_data['smic_better_avg']!=True]['lm_change'].mean()
avg_lmchange_new = smic_data[smic_data['new_smic_better']!=False]['lm_change'].mean() - smic_data[smic_data['new_smic_better']!=True]['lm_change'].mean()
avg_lmchange_new_avg = smic_data[smic_data['new_smic_better_avg']!=False]['lm_change'].mean() - smic_data[smic_data['new_smic_better_avg']!=True]['lm_change'].mean()

overall_stats.append({"GTP": avg_lmchange, "New GTP": avg_lmchange_new, "GTP (avg)": avg_lmchange_avg, "New GTP (avg)": avg_lmchange_new_avg})
index_names.append('LM Change')



smic_data_sorted = smic_data.sort_values(['sourceid', 'judgeid', 'lm_change'],ascending=[True, True, False], inplace=False)

bestsmic = smic_data_sorted.groupby(['sourceid', 'judgeid']).first()
bestsmic = bestsmic.reset_index()

per_best_smic = round(np.float(bestsmic['smic_better'].sum())/np.float(total_smcs)*100., 5)
per_best_smic_avg = round(np.float(bestsmic['smic_better_avg'].sum())/np.float(total_smcs)*100., 5)
per_best_new_smic = round(np.float(bestsmic['new_smic_better'].sum())/np.float(total_smcs)*100., 5)
per_best_new_smic_avg = round(np.float(bestsmic['new_smic_better_avg'].sum())/np.float(total_smcs)*100., 5)

overall_stats.append({"GTP": per_best_smic, "New GTP": per_best_new_smic, "GTP (avg)": per_best_smic_avg, "New GTP (avg)": per_best_new_smic_avg})
index_names.append('Perc Best LM')

#correlation = pd.DataFrame()
gtp_change = list((smic_data['gtp_log_prob_smc'] - smic_data['gtp_log_prob_smic'])/smic_data['gtp_log_prob_smc'])
new_gtp_change = list((smic_data['new_gtp_log_prob_smc'] - smic_data['new_gtp_log_prob_smic'])/smic_data['new_gtp_log_prob_smc'])
lm_change = list(smic_data['lm_change'])




(pearson_coeff_gtp, pvalue_gtp) = pearsonr(lm_change, gtp_change)
(pearson_coeff_new_gtp , pvalue_new_gtp) = pearsonr(lm_change, new_gtp_change)



overall_stats.append({"GTP": pearson_coeff_gtp, "New GTP": pvalue_gtp, "GTP (avg)": pearson_coeff_new_gtp, "New GTP (avg)": pvalue_new_gtp})
index_names.append('Correlation')


overall_stats = pd.DataFrame(data=overall_stats, index=index_names)
pd.set_option('float_format', '{:.4f}'.format)
print(overall_stats)


'''

print('\n##########################')
print('Rule wise Statistics')
print('###########################')



smic_accepted_rulewise = stats.groupby(['ruleid']).agg({'smic_better':['sum', 'count'], 'smic_better_avg':['sum', 'count'], 'new_smic_better':['sum', 'count'], 'new_smic_better_avg':['sum', 'count']})
index_names = range(1, len(smic_accepted_rulewise)+1)
column_names = ["GTP", "GTP (avg)", "New GTP", "New GTP (avg)"]



stats_per_accept = []
per_accepted_rulewise = stats_per_accept.append(list(smic_accepted_rulewise['smic_better']['sum']/smic_accepted_rulewise['smic_better']['count']*100.))
per_accepted_rulewise_avg = stats_per_accept.append(list(smic_accepted_rulewise['smic_better_avg']['sum']/smic_accepted_rulewise['smic_better_avg']['count']*100.))
per_accepted_new_rulewise = stats_per_accept.append(list(smic_accepted_rulewise['new_smic_better']['sum']/smic_accepted_rulewise['new_smic_better']['count']*100.))
per_accepted_new_rulewise_avg = stats_per_accept.append(list(smic_accepted_rulewise['new_smic_better_avg']['sum']/smic_accepted_rulewise['new_smic_better_avg']['count']*100.))

stats_per_accept = np.transpose(np.array(stats_per_accept))

stats_per_accept = pd.DataFrame(stats_per_accept, index=index_names, columns=column_names)

print('Percentage of Accepted SMICs')
print(stats_per_accept)

#####################################################

print('\nPercentage Contribution')
stats_per_cont = []
per_cont = stats_per_cont.append(list(smic_accepted_rulewise['smic_better']['sum']/noof_accepted_smic*100.))
per_cont_avg = stats_per_cont.append(list(smic_accepted_rulewise['smic_better_avg']['sum']/noof_accepted_smic*100.))
per_cont_new = stats_per_cont.append(list(smic_accepted_rulewise['new_smic_better']['sum']/noof_accepted_smic*100.))
per_cont_new_avg = stats_per_cont.append(list(smic_accepted_rulewise['new_smic_better_avg']['sum']/noof_accepted_smic*100.))

stats_per_cont = np.transpose(np.array(stats_per_cont))
stats_per_cont = pd.DataFrame(stats_per_cont, index=index_names, columns=column_names)
print(stats_per_cont)

#######################################################

stats_ruleid['atleast1_smic'] = stats_ruleid['smic_better']['sum']>0
stats_ruleid['atleast1_smic_avg'] = stats_ruleid['smic_better_avg']['sum']>0
stats_ruleid['atleast1_new_smic'] = stats_ruleid['new_smic_better']['sum']>0
stats_ruleid['atleast1_new_smic_avg'] = stats_ruleid['new_smic_better_avg']['sum']>0

smc_atleast1_smic_rulewise = stats_ruleid.groupby(['ruleid']).agg({'atleast1_smic':['sum', 'count'], 'atleast1_smic_avg':['sum', 'count'], 'atleast1_new_smic':['sum', 'count'], 'atleast1_new_smic_avg':['sum', 'count']})

stats_per_atleast1 = []
per_atleast1 = stats_per_atleast1.append(list(smc_atleast1_smic_rulewise['atleast1_smic']['sum']/smc_atleast1_smic_rulewise['atleast1_smic']['count']*100.))
per_atleast1_avg = stats_per_atleast1.append(list(smc_atleast1_smic_rulewise['atleast1_smic_avg']['sum']/smc_atleast1_smic_rulewise['atleast1_smic_avg']['count']*100.))
per_atleast1_new = stats_per_atleast1.append(list(smc_atleast1_smic_rulewise['atleast1_new_smic']['sum']/smc_atleast1_smic_rulewise['atleast1_new_smic']['count']*100.))
per_atleast1_new_avg = stats_per_atleast1.append(list(smc_atleast1_smic_rulewise['atleast1_new_smic_avg']['sum']/smc_atleast1_smic_rulewise['atleast1_new_smic_avg']['count']*100.))


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

bucket_stat_avg = pd.DataFrame()
bucket_stat_avg['accepted_smic'] = stats_no_ruleid['smic_better_avg']['sum']
bucket_stat_avg['generated_smic'] = stats_no_ruleid['smic_better_avg']['count']
bucket_stat_avg.loc[bucket_stat_avg['accepted_smic'] >= 20, ['accepted_smic']] = 20
bucket_stat_avg = bucket_stat_avg.groupby(['accepted_smic']).agg({'accepted_smic':['sum'], 'generated_smic':['sum', 'count']})

bucket_stat_new = pd.DataFrame()
bucket_stat_new['accepted_smic'] = stats_no_ruleid['new_smic_better']['sum']
bucket_stat_new['generated_smic'] = stats_no_ruleid['new_smic_better']['count']
bucket_stat_new.loc[bucket_stat_new['accepted_smic'] >= 20, ['accepted_smic']] = 20
bucket_stat_new = bucket_stat_new.groupby(['accepted_smic']).agg({'accepted_smic':['sum'], 'generated_smic':['sum', 'count']})

bucket_stat_new_avg = pd.DataFrame()
bucket_stat_new_avg['accepted_smic'] = stats_no_ruleid['new_smic_better_avg']['sum']
bucket_stat_new_avg['generated_smic'] = stats_no_ruleid['new_smic_better_avg']['count']
bucket_stat_new_avg.loc[bucket_stat_new_avg['accepted_smic'] >= 20, ['accepted_smic']] = 20
bucket_stat_new_avg = bucket_stat_new_avg.groupby(['accepted_smic']).agg({'accepted_smic':['sum'], 'generated_smic':['sum', 'count']})

index_names_buckets = bucket_stat.index


stats_per_smc = []
stats_per_smc.append(list(bucket_stat['generated_smic']['count']/total_smcs*100.))
stats_per_smc.append(list(bucket_stat_avg['generated_smic']['count']/total_smcs*100.))
stats_per_smc.append(list(bucket_stat_new['generated_smic']['count']/total_smcs*100.))
stats_per_smc.append(list(bucket_stat_new_avg['generated_smic']['count']/total_smcs*100.))


print('\nPercentage of SMCs with SMIC count bucketing')
stats_per_smc = np.transpose(np.array(stats_per_smc))
stats_per_smc = pd.DataFrame(stats_per_smc, index=index_names_buckets, columns=column_names)
print(stats_per_smc)

stats_per_accepted = []
stats_per_accepted.append(list(bucket_stat['accepted_smic']['sum']/bucket_stat['generated_smic']['sum']*100.))
stats_per_accepted.append(list(bucket_stat_avg['accepted_smic']['sum']/bucket_stat_avg['generated_smic']['sum']*100.))
stats_per_accepted.append(list(bucket_stat_new['accepted_smic']['sum']/bucket_stat_new['generated_smic']['sum']*100.))
stats_per_accepted.append(list(bucket_stat_new_avg['accepted_smic']['sum']/bucket_stat_new_avg['generated_smic']['sum']*100.))

print('\nPercentage of Accepted SMICs with SMIC count bucketing')
stats_per_accepted = np.transpose(np.array(stats_per_accepted))
stats_per_accepted = pd.DataFrame(stats_per_accepted, index=index_names_buckets, columns=column_names)
print(stats_per_accepted)
'''
