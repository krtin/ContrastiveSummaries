#!/bin/bash
# tokenize corresponding files

#perl ${CODE_DIR}tokenizer.perl -l 'en' < ${CODE_DIR}toutanova_lm.txt > ${CODE_DIR}toutanova_lm.txt.tok
#perl ${CODE_DIR}tokenizer.perl -l 'en' < ${CODE_DIR}nyt_toutanova_lm.txt > ${CODE_DIR}nyt_toutanova_lm.txt.tok
perl ${CODE_DIR}tokenizer.perl -l 'en' < ${CODE_DIR}smic.txt > ${CODE_DIR}smic.txt.tok
perl ${CODE_DIR}tokenizer.perl -l 'en' < ${CODE_DIR}smc.txt > ${CODE_DIR}smc.txt.tok

# extract dictionaries
#python ${CODE_DIR}build_dictionary.py ${CODE_DIR}toutanova_lm.txt.tok
#python ${CODE_DIR}build_dictionary.py ${CODE_DIR}nyt_toutanova_lm.txt.tok
#python ${CODE_DIR}build_dictionary.py ${CODE_DIR}europarl-v7.fr-en.en.tok

# shuffle traning data
#python ${CODE_DIR}shuffle.py ${CODE_DIR}toutanova_lm.txt.tok ${CODE_DIR}toutanova_lm.txt.tok
#python ${CODE_DIR}shuffle.py ${CODE_DIR}nyt_toutanova_lm.txt.tok ${CODE_DIR}nyt_toutanova_lm.txt.tok
