import os
from sys import argv
import subprocess
import signal
from subprocess import check_output

text = argv[1]

#print("Input:", text)
command = './tokenizer.perl -threads 1 -l en <<< "\\"' + text + '\\""'
#output = os.system(command)
#print(command)

out = check_output(['/bin/bash', '-c', command])
print(out.strip('\n'))
#proc = subprocess.Popen(['/bin/bash', '-c', command])
#output = output.stdout.strip('\n')
#proc.wait()
#print (proc.returncode)
#output = proc.stdout
#os.killpg(os.getpgid(proc.pid), signal.SIGTERM)



#print(output)
