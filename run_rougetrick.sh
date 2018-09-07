
#will process the raw corpus by picking the best summary as gold standard and other reference summaries as SMCs for each source
#further will also generate lm scores for gold and SMCs
#will store the final result in config.preprocessed_with_lm
python preprocess.py (2 min)

#this will take config.preprocessed_with_lm as input and generate the SMICs using rules in rules.json
#the smic corpus will be stored in config.smiccorpus, this will only have SMICs with ids to corresponding SMC in config.preprocessed_with_lm
#corenlp server must be running in the background
#java -cp .:lib/simplenlg-v4.4.3.jar simplenlgserver
python3 createRougetrickCorpus.py (10 hr 31 min)

#this will use config.smiccorpus to generate config.smiccorpus_with_lm
python processSMICs.py (23 min)

#this will take config.smiccorpus and config.preprocessed_with_lm and generate config.smiccorpus_filtered and config.smicfile_filtered
python3 filterCorpus.py (7 min)

#run preprocessing step for get to the point summarization
python gettothepoint_preprocess.py

#generate probabilies using gettothepoint for smc
python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_smc_sents/smc_sents_* --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True

#for cnndailymail corpus
python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/cnndailymail/chunked/smc/smc_*.bin --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --sel_gpu=0 --coverage=1

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_padded_smc_sents/smc_sents_*.bin --vocab_path=../../new_summarization/finished_files/vocab --exp_name=../../new_summarization/trained_1isto1 --single_pass=True --padded_abstract=1 --decode_reverse=1 --sel_gpu=0


python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_smic_sents_filtered/smic_sents_filtered_* --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True



garden-path



--jc-gpu1 (2)

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_smic_sents_filtered/smic_sents_filtered_* --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --decode_start=1 --decode_end=110000

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_smic_sents_filtered/smic_sents_filtered_* --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --decode_start=110001 --decode_end=220000




--agent-11 (2)

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_smic_sents_filtered/smic_sents_filtered_* --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --decode_start=220001 --decode_end=330000

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_smic_sents_filtered/smic_sents_filtered_* --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --decode_start=330001 --decode_end=440000


--agent-server-1 (2)

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_smic_sents_filtered/smic_sents_filtered_* --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --decode_start=440001 --decode_end=550000

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_smic_sents_filtered/smic_sents_filtered_* --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --decode_start=550001 --decode_end=660000

--jc-gpu3 (1)

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_smic_sents_filtered/smic_sents_filtered_* --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --decode_start=660001 --decode_end=770000

--jc-gpu2 (1)

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_smic_sents_filtered/smic_sents_filtered_* --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --decode_start=770001 --decode_end=880000

--jp-gpu3 (2)

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_smic_sents_filtered/smic_sents_filtered_* --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --decode_start=880001 --decode_end=990000

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_smic_sents_filtered/smic_sents_filtered_* --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --decode_start=990001 --decode_end=1100000

######## Moded summarizer #####

--dp-gpu1 (3)

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_padded_smic_sents_filtered/smic_sents_filtered_* --padded_abstract=1 --vocab_path=../../new_summarization/finished_files/vocab --exp_name=../../new_summarization/trained_1isto1 --single_pass=True --decode_start=1 --decode_end=55000 --sel_gpu=0 --decode_reverse=1

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_padded_smic_sents_filtered/smic_sents_filtered_* --padded_abstract=1 --vocab_path=../../new_summarization/finished_files/vocab --exp_name=../../new_summarization/trained_1isto1 --single_pass=True --decode_start=55001 --decode_end=110000 --sel_gpu=0 --decode_reverse=1

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_padded_smic_sents_filtered/smic_sents_filtered_* --padded_abstract=1 --vocab_path=../../new_summarization/finished_files/vocab --exp_name=../../new_summarization/trained_1isto1 --single_pass=True --decode_start=110001 --decode_end=165000 --sel_gpu=1 --decode_reverse=1

--dp-gpu2 (3)
python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_padded_smic_sents_filtered/smic_sents_filtered_* --padded_abstract=1 --vocab_path=../../new_summarization/finished_files/vocab --exp_name=../../new_summarization/trained_1isto1 --single_pass=True --decode_start=165001 --decode_end=220000 --sel_gpu=0 --decode_reverse=1

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_padded_smic_sents_filtered/smic_sents_filtered_* --padded_abstract=1 --vocab_path=../../new_summarization/finished_files/vocab --exp_name=../../new_summarization/trained_1isto1 --single_pass=True --decode_start=220001 --decode_end=275000 --sel_gpu=1 --decode_reverse=1

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_padded_smic_sents_filtered/smic_sents_filtered_* --padded_abstract=1 --vocab_path=../../new_summarization/finished_files/vocab --exp_name=../../new_summarization/trained_1isto1 --single_pass=True --decode_start=275001 --decode_end=330000 --sel_gpu=1 --decode_reverse=1

--dp-gpu3 (2)

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_padded_smic_sents_filtered/smic_sents_filtered_* --padded_abstract=1 --vocab_path=../../new_summarization/finished_files/vocab --exp_name=../../new_summarization/trained_1isto1 --single_pass=True --decode_start=330001 --decode_end=385000 --sel_gpu=0 --decode_reverse=1

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_padded_smic_sents_filtered/smic_sents_filtered_* --padded_abstract=1 --vocab_path=../../new_summarization/finished_files/vocab --exp_name=../../new_summarization/trained_1isto1 --single_pass=True --decode_start=385001 --decode_end=440000 --sel_gpu=1 --decode_reverse=1

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_padded_smic_sents_filtered/smic_sents_filtered_* --padded_abstract=1 --vocab_path=../../new_summarization/finished_files/vocab --exp_name=../../new_summarization/trained_1isto1 --single_pass=True --decode_start=440001 --decode_end=495000 --sel_gpu=0 --decode_reverse=1

--dp-gpu5 (3)

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_padded_smic_sents_filtered/smic_sents_filtered_* --padded_abstract=1 --vocab_path=../../new_summarization/finished_files/vocab --exp_name=../../new_summarization/trained_1isto1 --single_pass=True --decode_start=495001 --decode_end=550000 --sel_gpu=0 --decode_reverse=1

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_padded_smic_sents_filtered/smic_sents_filtered_* --padded_abstract=1 --vocab_path=../../new_summarization/finished_files/vocab --exp_name=../../new_summarization/trained_1isto1 --single_pass=True --decode_start=550001 --decode_end=605000 --sel_gpu=1 --decode_reverse=1

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_padded_smic_sents_filtered/smic_sents_filtered_* --padded_abstract=1 --vocab_path=../../new_summarization/finished_files/vocab --exp_name=../../new_summarization/trained_1isto1 --single_pass=True --decode_start=605001 --decode_end=660000 --sel_gpu=0 --decode_reverse=1

--jp-gpu3 (3)

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_padded_smic_sents_filtered/smic_sents_filtered_* --padded_abstract=1 --vocab_path=../../new_summarization/finished_files/vocab --exp_name=../../new_summarization/trained_1isto1 --single_pass=True --decode_start=660001 --decode_end=715000 --sel_gpu=0 --decode_reverse=1

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_padded_smic_sents_filtered/smic_sents_filtered_* --padded_abstract=1 --vocab_path=../../new_summarization/finished_files/vocab --exp_name=../../new_summarization/trained_1isto1 --single_pass=True --decode_start=715001 --decode_end=770000 --sel_gpu=0 --decode_reverse=1

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_padded_smic_sents_filtered/smic_sents_filtered_* --padded_abstract=1 --vocab_path=../../new_summarization/finished_files/vocab --exp_name=../../new_summarization/trained_1isto1 --single_pass=True --decode_start=770001 --decode_end=825000 --sel_gpu=0 --decode_reverse=1

--jc-gpu2 (3)

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_padded_smic_sents_filtered/smic_sents_filtered_* --padded_abstract=1 --vocab_path=../../new_summarization/finished_files/vocab --exp_name=../../new_summarization/trained_1isto1 --single_pass=True --decode_start=825001 --decode_end=880000 --sel_gpu=0 --decode_reverse=1

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_padded_smic_sents_filtered/smic_sents_filtered_* --padded_abstract=1 --vocab_path=../../new_summarization/finished_files/vocab --exp_name=../../new_summarization/trained_1isto1 --single_pass=True --decode_start=880001 --decode_end=935000 --sel_gpu=1 --decode_reverse=1

--agent-11 (3)

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_padded_smic_sents_filtered/smic_sents_filtered_* --padded_abstract=1 --vocab_path=../../new_summarization/finished_files/vocab --exp_name=../../new_summarization/trained_1isto1 --single_pass=True --decode_start=935001 --decode_end=990000 --sel_gpu=1 --decode_reverse=1

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_padded_smic_sents_filtered/smic_sents_filtered_* --padded_abstract=1 --vocab_path=../../new_summarization/finished_files/vocab --exp_name=../../new_summarization/trained_1isto1 --single_pass=True --decode_start=990001 --decode_end=1045000 --sel_gpu=1 --decode_reverse=1

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gettothepoint/chunked_padded_smic_sents_filtered/smic_sents_filtered_* --padded_abstract=1 --vocab_path=../../new_summarization/finished_files/vocab --exp_name=../../new_summarization/trained_1isto1 --single_pass=True --decode_start=1045001 --decode_end=1100000 --sel_gpu=1 --decode_reverse=1

python addGettoPointScore.py



################## cnndailymail

--dp-gpu3

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/cnndailymail/chunked/smic/smic_*.bin --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --sel_gpu=0 --coverage=1 --decode_start=1 --decode_end=50000

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/cnndailymail/chunked/smic/smic_*.bin --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --sel_gpu=1 --coverage=1 --decode_start=50001 --decode_end=100000

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/cnndailymail/chunked/smic/smic_*.bin --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --sel_gpu=1 --coverage=1 --decode_start=100001 --decode_end=150000


--jp-gpu1

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/cnndailymail/chunked/smic/smic_*.bin --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --sel_gpu=0 --coverage=1 --decode_start=150001 --decode_end=200000

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/cnndailymail/chunked/smic/smic_*.bin --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --sel_gpu=0 --coverage=1 --decode_start=200001 --decode_end=250000

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/cnndailymail/chunked/smic/smic_*.bin --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --sel_gpu=1 --coverage=1 --decode_start=250001 --decode_end=300000

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/cnndailymail/chunked/smic/smic_*.bin --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --sel_gpu=1 --coverage=1 --decode_start=300001 --decode_end=350000

--agent7-ml

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/cnndailymail/chunked/smic/smic_*.bin --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --sel_gpu=0 --coverage=1 --decode_start=350001 --decode_end=375000

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/cnndailymail/chunked/smic/smic_*.bin --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --sel_gpu=0 --coverage=1 --decode_start=375001 --decode_end=400000

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/cnndailymail/chunked/smic/smic_*.bin --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --sel_gpu=0 --coverage=1 --decode_start=400001 --decode_end=425000

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/cnndailymail/chunked/smic/smic_*.bin --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --sel_gpu=0 --coverage=1 --decode_start=425001 --decode_end=450000

--agent-8

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/cnndailymail/chunked/smic/smic_*.bin --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --sel_gpu=0 --coverage=1 --decode_start=450001 --decode_end=475000

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/cnndailymail/chunked/smic/smic_*.bin --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --sel_gpu=0 --coverage=1 --decode_start=475001 --decode_end=500000

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/cnndailymail/chunked/smic/smic_*.bin --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --sel_gpu=0 --coverage=1 --decode_start=500001 --decode_end=525000

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/cnndailymail/chunked/smic/smic_*.bin --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --sel_gpu=0 --coverage=1 --decode_start=525001 --decode_end=550000



--dp-gpu3

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/cnndailymail/chunked/smic/smic_*.bin --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --sel_gpu=3 --coverage=1 --decode_start=550001 --decode_end=600000

################# amrush toutanova

th summary/run.lua -modelFilename abs-model.th -inputfArt rougetrick/smc_article.txt -inputfAbs rougetrick/smc_abstract.txt -blockRepeatWords -outputfile probs/smc_toutanova.txt

-jc-gpu1
th summary/run.lua -modelFilename abs-model.th -inputfArt rougetrick/smics/smic_article_1.txt -inputfAbs rougetrick/smics/smic_abstract_1.txt -blockRepeatWords -outputfile probs/smic_toutanova_1.txt

th summary/run.lua -modelFilename abs-model.th -inputfArt rougetrick/smics/smic_article_2.txt -inputfAbs rougetrick/smics/smic_abstract_2.txt -blockRepeatWords -outputfile probs/smic_toutanova_2.txt

th summary/run.lua -modelFilename abs-model.th -inputfArt rougetrick/smics/smic_article_3.txt -inputfAbs rougetrick/smics/smic_abstract_3.txt -blockRepeatWords -outputfile probs/smic_toutanova_3.txt


th summary/run.lua -modelFilename abs-model.th -inputfArt test.example.article.txt -inputfAbs test.example.abstract.txt -blockRepeatWords -outputfile probs/test.txt


################# GIGAWORD


--jc-gpu(1)

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gigaword/chunked/smic/smic*.bin --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --decode_start=1 --decode_end=100000 --sel_gpu=0

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gigaword/chunked/smic/smic*.bin --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --decode_start=100001 --decode_end=200000 --sel_gpu=1

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gigaword/chunked/smic/smic*.bin --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --decode_start=200001 --decode_end=300000 --sel_gpu=1

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gigaword/chunked/smic/smic*.bin --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --decode_start=300001 --decode_end=400000 --sel_gpu=1

python run_summarization.py --mode=decode --data_path=/home/ml/kkumar15/nlpresearch/rougetrick_clean/data/gigaword/chunked/smic/smic*.bin --vocab_path=../pointer-generator/finished_files/vocab --exp_name=pretrained_model_tf1.2.1 --single_pass=True --decode_start=400001 --decode_end=510000 --sel_gpu=0
