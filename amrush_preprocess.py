import config
from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')

file_source_smic = config.sourcefile_filtered_gtp
#can be article or abstract
file_summary_smic = config.smicfile_filtered_gtp

file_source_smc = config.sourcefile
#can be article or abstract
file_summary_smc = config.smcfile

out_source_smic_file = "../summarization_models/namas_trained/rougetrick/smic_article.txt"
out_summary_smic_file = "../summarization_models/namas_trained/rougetrick/smic_abstract.txt"
out_source_smc_file = "../summarization_models/namas_trained/rougetrick/smc_article.txt"
out_summary_smc_file = "../summarization_models/namas_trained/rougetrick/smc_abstract.txt"

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

print('Writing Sources for SMIC')
count_source = 0
with open(out_source_smic_file, 'w') as wf:
    with open(file_source_smic, 'r') as f:
        for i, source in enumerate(f):

            source = tokenize(source)

            wf.write(source+'\n')
            print('\r%d'%(count_source), end='')
            count_source += 1

print('Writing SMICS')
count_smic = 0
with open(out_summary_smic_file, 'w') as wf:
    with open(file_summary_smic, 'r') as f:
        for i, smic in enumerate(f):
            #already tokenized so just copy
            wf.write(smic)
            print('\r%d'%(count_smic), end='')
            count_smic += 1

if(count_source!=count_smic):
    raise Exception("Number of Source %d dont match with number of smics %d"%(count_source, count_smic))

print('Writing Sources for SMC')
count_source = 0
with open(out_source_smc_file, 'w') as wf:
    with open(file_source_smc, 'r') as f:
        for i, source in enumerate(f):

            source = tokenize(source)

            wf.write(source+'\n')
            print('\r%d'%(count_source), end='')
            count_source += 1

print('Writing SMCS')
count_smc = 0
with open(out_summary_smc_file, 'w') as wf:
    with open(file_summary_smc, 'r') as f:
        for i, smc in enumerate(f):
            smc = tokenize(smc)

            wf.write(smc+'\n')
            print('\r%d'%(count_smc), end='')
            count_smc += 1

if(count_source!=count_smc):
    raise Exception("Number of Source %d dont match with number of smcs %d"%(count_source, count_smc))
