import numpy as np
import pandas as pd
from rouge import Rouge
from smic_generator_disc import smic_generator
import pickle as pkl
import config
import time
from multiprocessing import Pool, cpu_count, Value, Array, Manager
import psutil
import os

os.nice(5)
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

gigaword_title = "data/test.title.reduced.txt"
gigaword_article = "data/test.article.reduced.txt"

debug_data = 0
print("Reading corpus")

with open(gigaword_article ,'r') as f:
    articles = f.read().split('\n')

with open(gigaword_title ,'r') as f:
    titles = f.read().split('\n')

print("Total %d articles found and %d Titles found"%(len(articles), len(titles)))

if(len(articles)!=len(titles)):
    print("Length of articles and titles should be same")

idx = range(len(articles))

data = pd.DataFrame([], columns=['idx', 'source', 'smc'])

data['idx'] = idx
data['source'] = articles
data['smc'] = titles

data.to_csv('data/gigaword_corpus.csv', index=False)

print(data)

if(debug_data!=0):
    data = data[0:5]
#print(data)
#print(len(data))
batchsize = 10000
totallen = len(data)
totaldropped=0
avg_batch_time = 0.
starting_batch = 0
batch_start_time = 0
totalsmics=[]
data_store_path = "data/discriminator_corpus_train.pkl"

#print("starting batch", starting_batch)
starting_batch = 0
#print("starting batch", starting_batch)
#print(data.columns)

smic_gen = smic_generator()
manager = Manager()
totalsmics_old = manager.list([])
#totalsmics = []
starttime = time.time()
est_time = 0
counter = Value('i', 0)
#for i, row in data.iterrows():
def genSMIC(j, row):
    global counter, est_time, batch_start_time, smic_gen, totallen

    i = counter.value
    printProgressBar(i+1, totallen, prefix = 'Progress:', suffix = 'Complete', length = 50, time_string = time.strftime("%H:%M:%S", time.gmtime(est_time)))

    if(1):
        sourceid = row['idx']
        source = row['source']
        smc = row['smc']

        score = 0
        smics = smic_gen.generate(source, None, smc, score, sourceid, None, config.debug_mode)


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

#pool = Pool(None, limit_cpu)
print("Number of CPUs", cpu_count())
def limit_cpu():
    "is called at every process start"
    p = psutil.Process(os.getpid())
    # set to lowest priority, this is windows only, on Unix use ps.nice(19)
    p.nice(5)

pool = Pool(None, limit_cpu)

if(config.debug_data!=0):
    data = data[801:805]
#print(data)
#print(len(data))
totalsmics = []
for smics in pool.starmap(genSMIC, data.iterrows()):
    #print(len(p))
    #print(p)
    if(len(smics)!=0):
        totalsmics.extend(smics)
    pass

noof_smics = len(totalsmics)
totalsmics = list(totalsmics)

totalsmics = pd.DataFrame(totalsmics)
#just to ensure there are no duplicates
totalsmics['smic'] = totalsmics['smic'].str.lower()
totalsmics = totalsmics.drop_duplicates('smic')
length_afterdrop = len(totalsmics)
del totalsmics["judgeid"]


noof_smics = len(totalsmics)
totalsmics['smic_id'] = range(noof_smics)
print(totalsmics)

totalsmics.to_csv('data/gigaword_smic_corpus.csv', index=False)

with open('data/gigaword_smic.txt', 'w') as f:
    f.write('\n'.join(list(totalsmics['smic'])))

with open('data/gigaword_smic_id.txt', 'w') as f:
    f.write('\n'.join(list(totalsmics['smic_id'].astype(str))))


endtime = time.time()
print("SMIC generation complete, it took %s to generate %d SMICs drop %d" % (time.strftime("%H:%M:%S", time.gmtime(endtime - starttime)), noof_smics, totaldropped))
