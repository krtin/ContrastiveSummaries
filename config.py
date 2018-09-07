#Raw data corpus in specific format
#['sourceid', 'category', 'source', 'summary', 'judgeid', 'meaning', 'grammar', 'sents', 'rank']
#should contain columns rank, id and summary, source
rawCorpus = 'data/rougetrick_toutanova.csv'

#processed corpus
#Index([u'sourceid', u'system', u'source', u'category', u'judgeid', u'meaning',
#       u'grammar', u'sents', u'rank', u'gold', u'rouge1_f', u'rouge1_p',
#       u'rouge1_r', u'rouge2_f', u'rouge2_p', u'rouge2_r', u'rougel_f',
#       u'rougel_p', u'rougel_r'],
#      dtype='object')

processedCorpus = 'data/rougetrick_processed.csv'
force_process = False

#specify file path where smc sentences will be written, this will be used to get the LM scores
smcfile = 'data/smc_sents.txt'

#specify file path where gold sentences will be written, this will be used to get the LM scores
goldfile = 'data/gold_sents.txt'

#file to save preprocessed file after adding lm scores
preprocessed_with_lm = 'data/rougetrick_with_lmscores.csv'

#set debug mode off with value 0 and any other integer to run for that specific ruleid
debug_mode = 0

#set debug mode for data 0 for no debug and any other integer to select that row
debug_data = 0

#file where all the generated smics will be stored
smiccorpus = 'data/rougetrick_smic.csv'

#specify file path where smics will be stored to get LM scores
smicfile = 'data/smic_sents.txt'

#file to store source sentences corresponding to smc
sourcefile = 'data/source_smc_sents.txt'

#file for storing smic with their lm scores
smiccorpus_with_lm = 'data/rougetrick_smic_with_lmscores.csv'

#file for storing variation in rouge and lm scores for SMCs
smc_variation_stats = 'data/smc_variation_stats.pkl'

#file after filtering smics
smiccorpus_filtered = 'data/rougetrick_smic_filtered.csv'

#smic sents filtered
smicfile_filtered_gtp = 'data/gettothepoint/smic_sents_filtered.txt'

#smicid sents filtered
smicidfile_filtered_gtp = 'data/gettothepoint/smicid_sents_filtered.txt'

#smicid sents filtered
smicidfile_filtered_new_gtp = 'data/gettothepoint/smicid_sents_filtered.txt'

#source sents filtered
sourcefile_filtered_gtp = 'data/gettothepoint/source_sents_filtered.txt'

#pvalue for filtering corpus
pvalue = 0.99

#probability file for get to the point smc
gtp_smc_probfile = '/home/ml/kkumar15/nlpresearch/summarization_models/GetToThePoint_AbigailSee/gettothepoint_krtin/pretrained_model_tf1.2.1/decode_smc_400maxenc_1beam_35mindec_100maxdec_ckpt-238410/prob/probs.pkl'

#probability file for get to the point smic
gtp_smic_probfile = '/home/ml/kkumar15/nlpresearch/summarization_models/GetToThePoint_AbigailSee/gettothepoint_krtin/pretrained_model_tf1.2.1/decode_smic_400maxenc_1beam_35mindec_100maxdec_ckpt-238410/prob/probs.pkl'

#probability dir for get to the point smic
gtp_smic_probdir = '/home/ml/kkumar15/nlpresearch/summarization_models/GetToThePoint_AbigailSee/gettothepoint_krtin/pretrained_model_tf1.2.1/decode_smic_400maxenc_1beam_35mindec_100maxdec_ckpt-238410/prob/'

#probability file for get to the point smc
new_gtp_smc_probfile = '/home/ml/kkumar15/nlpresearch/summarization_models/new_summarization/trained_1isto1/decode_padded_smc_400maxenc_1beam_35mindec_100maxdec_ckpt-408277/prob/probs.pkl'

#probability file for get to the point smic
new_gtp_smic_probfile = '/home/ml/kkumar15/nlpresearch/summarization_models/new_summarization/trained_1isto1/decode_padded_smic_400maxenc_1beam_35mindec_100maxdec_ckpt-408277/prob/probs.pkl'

#probability dir for get to the point smic
new_gtp_smic_probdir = '/home/ml/kkumar15/nlpresearch/summarization_models/new_summarization/trained_1isto1/decode_padded_smic_400maxenc_1beam_35mindec_100maxdec_ckpt-408277/prob/'

#probability file for get to the point smc
new_gtp_smc_reverse_probfile = '/home/ml/kkumar15/nlpresearch/summarization_models/new_summarization/trained_1isto1/decode_padded_smc_1reversed_ckpt-408277/prob/probs.pkl'

#probability file for get to the point smic
new_gtp_smic_reverse_probfile = '/home/ml/kkumar15/nlpresearch/summarization_models/new_summarization/trained_1isto1/decode_padded_smic_1reversed_ckpt-408277/prob/probs.pkl'

#probability dir for get to the point smic
new_gtp_smic_reverse_probdir = '/home/ml/kkumar15/nlpresearch/summarization_models/new_summarization/trained_1isto1/decode_padded_smic_1reversed_ckpt-408277/prob/'


smc_final = 'data/smc_final.csv'

smic_final = 'data/smic_final.csv'

smc_final_new = 'data/smc_final_new.csv'

smic_final_new = 'data/smic_final_new.csv'

non_differentiable_smic = 'data/non_differentiable_smic.csv'

non_differentiable_smic_manual_check = 'data/verified_smics.csv'
