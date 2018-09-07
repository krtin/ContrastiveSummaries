import numpy as np
import pandas as pd
from rouge import Rouge
from smic_generator_stepwise import smic_generator
import pickle as pkl
import config
import time
from multiprocessing import Pool, cpu_count, Value, Array, Manager
import psutil
import os


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 5, length = 100, fill = '█', time_string = ''):
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
    print('\r%s |%s| %s%% %s (%s) est %s' % (prefix, bar, percent, suffix, str(iteration), time_string), end = '\r')
    #f.write(prefix +' | '+ bar +'| '+ percent +'% '+ suffix + '(' + str(iteration) + ')')
    #f.close()
    # Print New Line on Complete
    if iteration == total:
        print()

def getRougeScore(gold, sys):
    rouge = Rouge()
    scores = rouge.get_scores(gold, sys)
    return scores


data = pd.read_csv(config.preprocessed_with_lm, dtype={"judgeid": object, 'rouge1_f': np.float64, u'rouge1_p': np.float64, 'rouge1_r': np.float64, 'rouge2_f': np.float64, 'rouge2_p': np.float64, 'rouge2_r': np.float64, 'rougel_f': np.float64, 'rougel_p': np.float64, 'rougel_r': np.float64, 'sourceid': object, 'smc_lmscore': np.float64})
#print(data.columns)

totallen = len(data)

smic_gen = smic_generator()
manager = Manager()
totalsmics = manager.list([])
#totalsmics = []
starttime = time.time()
est_time = 0
counter = Value('i', 0)
#for i, row in data.iterrows():
def genSMIC(j, row):
    global counter, est_time, starttime, totalsmics, smic_gen, totallen

    i = counter.value
    printProgressBar(i+1, totallen, prefix = 'Progress:', suffix = 'Complete', length = 50, time_string = time.strftime("%H:%M:%S", time.gmtime(est_time)))
    #print(row['sourceid'], row['judgeid'])
    #if((row['sourceid']=='2305' and row['judgeid']=='213') or config.debug_data==0):
    #if((row['sourceid']=='2216' and row['judgeid']=='100') or config.debug_data==0):
    if(1):
        sourceid = row['sourceid']
        judgeid = row['judgeid']

        #print(sourceid)
        source = row['source']
        gold = row['gold']
        smc = row['system']
        #print(source)
        #print(gold)
        #print(smc)

        #create smic generator object
        score = 0
        smics = smic_gen.generate(source, gold, smc, score, sourceid, judgeid, config.debug_mode)
        #print(smics)

        #if(config.debug_data!=0):
        #    break


        #incase no smic was generated
        if(len(smics)!=0):
            #with totalsmics.get_lock():
            totalsmics.extend(smics)
            #print(smics)
    #print((totalsmics))
    progtime = time.time()
    with counter.get_lock():
        counter.value += 1
    i = counter.value
    est_time = (progtime - starttime)/(i+1)*(totallen-i-1)

    return smics

def limit_cpu():
    "is called at every process start"
    p = psutil.Process(os.getpid())
    # set to lowest priority, this is windows only, on Unix use ps.nice(19)
    p.nice(5)

pool = Pool(None, limit_cpu)
if(config.debug_data!=0):
    data = data[801:802]
#print(data)
#print(len(data))
for p in pool.starmap(genSMIC, data.iterrows()):
    #print(len(p))
    #print(p)
    pass

noof_smics = len(totalsmics)
totalsmics = list(totalsmics)
#print(totalsmics[0]['smic'])
#print(totalsmics[0]['positions'])
totalsmics = pd.DataFrame(totalsmics)
#just to ensure there are no duplicates
totalsmics['smic'] = totalsmics['smic'].str.lower()
totalsmics = totalsmics.drop_duplicates('smic')
length_afterdrop = len(totalsmics)
#print(totalsmics)


totalsmics.to_csv('data/stepcorpus.csv', index=False)

endtime = time.time()
print("SMIC generation complete, it took %s to generate %d SMICs drop %d" % (time.strftime("%H:%M:%S", time.gmtime(endtime - starttime)), noof_smics, length_afterdrop))
