import pandas as pd
from tensorflow.core.example import example_pb2
import struct

from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')

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

    return sents.lower()

def get_bin_data(article, abstract):
    if(type(article)==float or type(abstract)==float):
        print(article)
        return None, None
    article = article.encode('utf-8')
    abstract = abstract.encode('utf-8')
    tf_example = example_pb2.Example()
    tf_example.features.feature['article'].bytes_list.value.extend([article])
    tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)

    length = struct.pack('q', str_len)
    data = struct.pack('%ds' % str_len, tf_example_str)

    return length, data

create_txt_files = 0
create_bin_files = 1

if(create_txt_files):
    smc_file = 'data/gigaword_corpus.csv'
    smic_file = 'data/gigaword_smic_corpus.csv'

    smc_data = pd.read_csv(smc_file)
    smic_data = pd.read_csv(smic_file)

    print(smc_data.columns)
    print(smic_data.columns)
    del smic_data['Unnamed: 0']

    beforelen = len(smic_data)
    smic_data = pd.merge(smic_data, smc_data, left_on='sourceid', right_on='idx', how='inner')


    if(len(smic_data)!=beforelen):
        raise Exception('some problem in merging')

    with open('data/gigaword_smic.txt', 'w') as f:
        f.write('\n'.join(list(smic_data['smic'])))

    with open('data/gigaword_smic_id.txt', 'w') as f:
        f.write('\n'.join(list(smic_data['smic_id'].astype(str))))

    with open('data/gigaword_smic_source.txt', 'w') as f:
        f.write('\n'.join(list(smic_data['source'])))

if(create_bin_files):
    smc_bin = 'data/gigaword/smc.bin'
    smic_bin = 'data/gigaword/smic.bin'

    print('Reading Data')

    with open('data/gigaword_smic.txt', 'r') as f:
        smic_data = f.read().split('\n')

    with open('data/gigaword_smic_source.txt', 'r') as f:
        smic_data_source = f.read().split('\n')

    with open('data/test.title.reduced.txt', 'r') as f:
        smc_data = f.read().split('\n')

    with open('data/test.article.reduced.txt', 'r') as f:
        smc_data_source = f.read().split('\n')

    print('Writing SMIC bin files')

    total_data=0
    writer = open(smc_bin, 'wb')
    for smc, source in zip(smc_data, smc_data_source):
        smc = tokenize(smc)
        source = tokenize(source)
        length, data = get_bin_data(source, smc)
        if(length==None):
            raise Exception(':O')
        writer.write(length)
        writer.write(data)
        total_data+=1
        print('\r Written: %d'%(total_data), end='')

    writer.close()
    print()
    total_data=0
    writer = open(smic_bin, 'wb')
    for smic, source in zip(smic_data, smic_data_source):
        source = tokenize(source)
        length, data = get_bin_data(source, smic)
        if(length==None):
            raise Exception(':O')
        writer.write(length)
        writer.write(data)
        total_data+=1
        print('\r Written: %d'%(total_data), end='')
    print()
    writer.close()
