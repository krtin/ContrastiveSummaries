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

discrim_corpus = "../summarization_models/discrimnator/data/train.csv"
debug_data = 0
print("Reading corpus %s"% discrim_corpus)
data = pd.read_csv(discrim_corpus, dtype={"idx": np.int64, 'source': object, 'smc': object, })

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
print("Will save at %s" % data_store_path)
if(os.path.exists(data_store_path)):
    [totalsmics, starting_batch, totaldropped] = pkl.load(open(data_store_path, 'rb'))
#print("starting batch", starting_batch)
starting_batch = int(starting_batch/10)
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
    #printProgressBar(i+1, totallen, prefix = 'Progress:', suffix = 'Complete', length = 50, time_string = time.strftime("%H:%M:%S", time.gmtime(est_time)))
    #print(row['sourceid'], row['judgeid'])
    #if((row['sourceid']=='2305' and row['judgeid']=='213') or config.debug_data==0):
    #if((row['sourceid']=='2216' and row['judgeid']=='100') or config.debug_data==0):
    if(1):
        sourceid = row['idx']
        source = row['source']
        smc = row['smc']
        #print(source)
        #print(idx)
        #print(smc)



        #create smic generator object
        score = 0
        smics = smic_gen.generate(source, None, smc, score, sourceid, None, config.debug_mode)

        #print(len(smics))

        #if(config.debug_data!=0):
        #    break


        #incase no smic was generated
        #if(len(smics)!=0):
            #with totalsmics.get_lock():
        #    totalsmics.extend(smics)
            #print(smics)
    #print((totalsmics))

    with counter.get_lock():
        counter.value += 1
    batch_count = counter.value
    progtime = time.time()
    est_time = (progtime - batch_start_time)/(batch_count)*(batchsize-batch_count)
    printProgressBar(batch_count, batchsize, prefix = 'Progress:', suffix = 'Complete', length = 50, time_string = time.strftime("%H:%M:%S", time.gmtime(est_time)))
    if(counter.value == batchsize):
        counter.value = 0

    return smics

def limit_cpu():
    "is called at every process start"
    p = psutil.Process(os.getpid())
    # set to lowest priority, this is windows only, on Unix use ps.nice(19)
    p.nice(5)

#pool = Pool(None, limit_cpu)
print("Number of CPUs", cpu_count())
pool = Pool(max(cpu_count()//2, 1))

for batch in range(starting_batch, int(np.ceil(totallen/batchsize))):
    #if(batch%10 ==0):
    #    break
    print("Starting Batch %d"%(batch+1))
    batch_smics = []
    batch_start_time = time.time()
    batch_count = 0
    for smics in pool.starmap(genSMIC, data[batch*batchsize: min((batch+1)*batchsize, totallen)].iterrows()):
        #print(len(p))
        #print(p)

        if(len(smics)!=0):
            batch_smics.extend(smics)


        pass

    batch_smics = pd.DataFrame(batch_smics)
    del batch_smics["judgeid"]
    beforedrop_len = len(batch_smics)
    batch_smics['smic'] = batch_smics['smic'].str.lower()
    batch_smics = batch_smics.drop_duplicates('smic')
    totaldropped += beforedrop_len - len(batch_smics)

    batch_end_time = time.time()
    avg_batch_time = float(avg_batch_time*batch + (batch_end_time - batch_start_time))/float((batch+1))
    expectedtime = float(int(np.ceil(totallen/batchsize))-batch-1-starting_batch)*avg_batch_time

    if(len(totalsmics)==0):
        totalsmics = batch_smics
    else:
        totalsmics = pd.concat([totalsmics, batch_smics])

    pkl.dump([totalsmics, batch+1, totaldropped], open(data_store_path, 'wb'))

    print("Batch %d saved out of %d Batches, expected time to finish %s. Dropped %d. SMICs %d"%(batch+1, int(np.ceil(totallen/batchsize)), time.strftime("%H:%M:%S", time.gmtime(expectedtime)), totaldropped, len(totalsmics)))




noof_smics = len(totalsmics)
#print(totalsmics)

endtime = time.time()
print("SMIC generation complete, it took %s to generate %d SMICs drop %d" % (time.strftime("%H:%M:%S", time.gmtime(endtime - starttime)), noof_smics, totaldropped))
#data = pd.read_csv(config.smiccorpus)
#print(data)
#print(data['smic'])
