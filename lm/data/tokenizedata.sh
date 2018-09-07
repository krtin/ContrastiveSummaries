#!/bin/bash
# tokenize corresponding files

if [ "$#" -ne 2 ]; then
    echo ""
    echo "Usage: $0 inputfile outputfile"
    echo ""
    exit 1
fi


perl ${CODE_DIR}lm/data/tokenizer.perl -l 'en' < ${CODE_DIR}$1 > ${CODE_DIR}$2
