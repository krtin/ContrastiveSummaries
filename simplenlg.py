import time
import requests

def changePOS(verb, pos):
    data = requests.get('http://127.0.0.1:8181?verb='+verb+'&pos='+pos)
    out = data.text

    out = out.strip('\n').split('!!!')
    out = [item for item in out if item != '!!ERROR!!']
    #to make sure words are unique in case of multiple words
    out = set(out)
    #print(out)
    return out

def changePlurality(noun, pos):
    data = requests.get('http://127.0.0.1:8181?noun='+noun+'&pos='+pos)
    out = data.text
    out = out.strip('\n')
    if(out=='!!ERROR!!' or out=='null'):
        out = ''

    #print(out)
    return out


#starttime = time.time()
#changePlurality('students', 'NN')
#changePOS("'s", 'VBD')
#endtime = time.time()
#print(endtime - starttime)
#print(time.strftime("%H:%M:%S", time.gmtime(endtime - starttime)))
