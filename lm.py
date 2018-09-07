
from subprocess import check_output
import config

run_lm = "lm/session0/predict.sh"
run_token  = "lm/data/tokenizedata.sh"

def generateLMScore(inputfile, outputfile):
    #tokenize inputfile
    cmd_token = run_token + ' ' + inputfile + ' ' + inputfile+'.tok'
    #print(cmd_token)
    #out = check_output(['/bin/bash', '-c', cmd_token])
    #print(out)

    cmd_lm = run_lm + ' ' + inputfile+'.tok' + ' ' + outputfile
    print(cmd_lm)
    #out = check_output(['/bin/bash', '-c', cmd_lm])
    #print(out)


generateLMScore(config.smicfile, config.smicfile+'_lm.pkl')
