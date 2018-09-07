import os
import numpy as np

smic_dir = "../summarization_models/namas_trained/rougetrick/smics"
out_source_smic_file = "../summarization_models/namas_trained/rougetrick/smic_article.txt"
out_summary_smic_file = "../summarization_models/namas_trained/rougetrick/smic_abstract.txt"

if(os.path.exists(smic_dir) is False):
    os.mkdir(smic_dir)

noof_splits = 3

with open(out_source_smic_file, 'r') as f:
    sources = f.read().split('\n')

if(sources[-1]==''):
    sources.pop()

noof_sources = len(sources)
print(noof_sources)


with open(out_summary_smic_file, 'r') as f:
    smics = f.read().split('\n')

noof_smics = len(smics)
print(noof_smics)

if(noof_smics!=noof_sources):
    raise Exception("Number of sources and summaries not equal")

split_qty = []
batch_size = int(noof_smics/noof_splits)
for i in range(noof_splits-1):
    split_qty.append(batch_size)

split_qty.append(noof_smics - batch_size*(noof_splits-1))

counter = 0
batch_counter = 0
batch_data_source = []
batch_data_smic = []

for source, smic in zip(sources, smics):
    batch_data_source.append(source)
    batch_data_smic.append(smic)

    counter += 1

    if(counter==split_qty[batch_counter]):
        #write to file
        with open( os.path.join(smic_dir, "smic_abstract_"+str(batch_counter+1)+".txt"), 'w') as f:
            f.write('\n'.join(batch_data_smic))

        with open( os.path.join(smic_dir, "smic_article_"+str(batch_counter+1)+".txt"), 'w') as f:
            f.write('\n'.join(batch_data_source))

        #reset data
        batch_data_source = []
        batch_data_smic = []

        #update batch counter
        batch_counter += 1

        #reset counter
        counter = 0
