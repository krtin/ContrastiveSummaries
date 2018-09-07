import config
import pandas as pd
import numpy as np
import pickle as pkl
import os
import scipy.stats
import time
from multiprocessing import Pool, cpu_count, Value, Array, Manager
import psutil
import os

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 5, length = 100, fill = '█', accepted = 0):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    #f = open('results/stats.txt', 'w')
    print('\r%s |%s| %s%% %s (%s) valid %d' % (prefix, bar, percent, suffix, str(iteration), accepted), end = '\r')
    #f.write(prefix +' | '+ bar +'| '+ percent +'% '+ suffix + '(' + str(iteration) + ')')
    #f.close()
    # Print New Line on Complete
    if iteration == total:
        print()

def generateSMCStats():
    smc_data = pd.read_csv(config.preprocessed_with_lm, usecols=['sourceid', 'rouge1_f', 'rouge1_p', 'rouge1_r', 'rouge2_f', 'rouge2_p', 'rouge2_r', 'rougel_f', 'rougel_p', 'rougel_r', 'smc_lmscore'])
    sourceids = list(smc_data['sourceid'].unique())

    smc_stats = []
    for sourceid in sourceids:
        #get SMCs for current source
        currentdata = smc_data[smc_data['sourceid']==sourceid]
        #delete column for sourceids
        del currentdata['sourceid']
        #find max values
        max_values = currentdata.max()
        #find change
        currentdata = max_values - currentdata
        #make zero values none
        currentdata[currentdata==0] = None

        smc_stats.extend(currentdata.to_dict(orient='records'))


    #convert back to data frame
    smc_stats = pd.DataFrame(smc_stats)
    smc_stats_mean = list(smc_stats.mean())
    smc_stats_var = list(smc_stats.var())
    #write stats to file so that it can be loaded later
    pkl.dump([smc_stats_mean, smc_stats_var],open(config.smc_variation_stats,'wb'))

starttime = time.time()

#check if SMC stats need to be generated
if(os.path.exists(config.smc_variation_stats) is False):
    print("Generating SMC mean and variance file")
    generateSMCStats()

[smc_stats_mean, smc_stats_var] = pkl.load(open(config.smc_variation_stats,'rb'))
#print(smc_stats_mean, smc_stats_var)


print("Starting to Filter corpus based upon pvalue")
#load SMIC corpus
smic_data = pd.read_csv(config.smiccorpus_with_lm, dtype={"judgeid": object, 'rouge1_f': np.float64, u'rouge1_p': np.float64, 'rouge1_r': np.float64, 'rouge2_f': np.float64, 'rouge2_p': np.float64, 'rouge2_r': np.float64, 'rougel_f': np.float64, 'rougel_p': np.float64, 'rougel_r': np.float64, 'ruleid': np.int32, 'smic': object, 'sourceid': object, 'smic_lmscore': np.float64, 'smic_id': np.int32})


smc_data = pd.read_csv(config.preprocessed_with_lm, dtype={"judgeid": object, 'rouge1_f': np.float64, u'rouge1_p': np.float64, 'rouge1_r': np.float64, 'rouge2_f': np.float64, 'rouge2_p': np.float64, 'rouge2_r': np.float64, 'rougel_f': np.float64, 'rougel_p': np.float64, 'rougel_r': np.float64, 'sourceid': object, 'smc_lmscore': np.float64, 'source': object}, usecols=['judgeid', 'rouge1_f', u'rouge1_p', 'rouge1_r', 'rouge2_f', 'rouge2_p', 'rouge2_r', 'rougel_f', 'rougel_p', 'rougel_r', 'sourceid', 'smc_lmscore', 'source'])
smc_rows = []
prevsourceid = -1
accepted_smic = Value('i', 0)
counter = Value('i', 0)

totallen = len(smic_data)

#share smic_data across processes
mgr = Manager()
ns = mgr.Namespace()
ns.smic_data = smic_data

def filterSMIC(i, row):
    global smc_data, ns, totallen, accepted_smic, counter

    #get rows only when sourceid changes
    #if(prevsourceid!=row['sourceid']):
    #for multi processing always get smc_rows
    smc_rows = smc_data[smc_data['sourceid']==str(row['sourceid'])]

    #get single row by matching judge id
    smc_row = smc_rows[smc_rows['judgeid']==row['judgeid']]

    if(len(smc_row)!=1):
        raise Exception('Multiple Records with same sourceid and judgeid found or record not found')

    smc_row = smc_row.to_dict(orient='records')[0]

    smic_scores = [row['rouge1_f'], row['rouge1_p'], row['rouge1_r'], row['rouge2_f'], row['rouge2_p'], row['rouge2_r'], row['rougel_f'], row['rougel_p'], row['rougel_r'], row['smic_lmscore']]

    smc_scores = [smc_row['rouge1_f'], smc_row['rouge1_p'], smc_row['rouge1_r'], smc_row['rouge2_f'], smc_row['rouge2_p'], smc_row['rouge2_r'], smc_row['rougel_f'], smc_row['rougel_p'], smc_row['rougel_r'], smc_row['smc_lmscore']]

    score_change = abs(np.array(smc_scores) - np.array(smic_scores))
    zvalue = abs(np.divide(score_change - np.array(smc_stats_mean), np.array(smc_stats_var)))
    pvalues = scipy.stats.norm.cdf(zvalue)

    acceptable = np.greater_equal(pvalues, np.full(len(pvalues), config.pvalue))
    acceptable = acceptable.astype(int)
    rouge_fvalue_test = acceptable[0]+acceptable[3]+acceptable[6]



    flag=0
    if(rouge_fvalue_test==3):
        with accepted_smic.get_lock():
            accepted_smic.value +=1
            flag=1


    with counter.get_lock():
        counter.value +=1
    printProgressBar(counter.value, totallen, prefix = 'Progress:', suffix = 'Complete', length = 50, accepted = accepted_smic.value)


    #else:

        #delete row from data
        #with ns.get_lock():
        #ns.smic_data.drop(i, inplace=True)

    #dont need this for multiprocessing
    #prevsourceid = row['sourceid']

    if(flag):
        return row

def limit_cpu():
    "is called at every process start"
    p = psutil.Process(os.getpid())
    # set to lowest priority, this is windows only, on Unix use ps.nice(19)
    p.nice(5)

pool = Pool(None, limit_cpu)

#smic_data = smic_data[0:500]
#multi-processing
selected_smic = []
for p in pool.starmap(filterSMIC, smic_data.iterrows()):
    if(p is not None):
        selected_smic.append(p)
    pass

smic_data = pd.DataFrame(selected_smic)
smic_data = smic_data.sort_values('smic_id')
len1 = (len(smic_data))
smic_data = pd.merge(smic_data, smc_data[['judgeid', 'sourceid', 'source']], left_on=['sourceid', 'judgeid'], right_on=['sourceid', 'judgeid'], how='left')
len2 = (len(smic_data))

if(len1!=len2):
    raise Exception('Unequal length after join')

with open(config.sourcefile_filtered_gtp, 'w') as f:
    f.write('\n'.join(list(smic_data["source"])))

del smic_data['source']

smic_data.to_csv(config.smiccorpus_filtered, index=False)
with open(config.smicfile_filtered_gtp, 'w') as f:
    f.write('\n'.join(list(smic_data['smic'])))



with open(config.smicidfile_filtered_gtp, 'w') as f:
    f.write('\n'.join(list(smic_data['smic_id'].astype(str))))

endtime = time.time()

print("Filtering took %s and generated %d SMICs out of %d" % (time.strftime("%H:%M:%S", time.gmtime(endtime - starttime)), len(smic_data), totallen))
