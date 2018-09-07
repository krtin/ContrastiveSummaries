import cPickle as pkl
import gzip
import os
from subprocess import check_output

class TextIterator:
    def __init__(self, source,
                 source_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1):
        self.inputistext = False
        if source.endswith('.gz'):
            self.source = gzip.open(source, 'r')
        elif(source.endswith('.tok')):
            self.source = open(source, 'r')
        else:
            #it is not a file path, then the input sentence
            command = 'lm/data/tokenizer.perl -threads 1 -l en <<< "' + source + '"'
            out = check_output(['/bin/bash', '-c', command])
            self.source = out.strip('\n')
            self.inputistext = True
        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            if(self.inputistext is False):
                self.reset()
            raise StopIteration

        source = []

        try:

            # actual work here
            while True:
                if self.inputistext is False:
                    ss = self.source.readline()
                else:
                    ss = self.source
                    #print('I have set the source to: ', ss)
                if ss == "":
                    raise IOError
                ss = ss.strip().split()
                ss = [self.source_dict[w] if w in self.source_dict else 1
                      for w in ss]
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source else 1 for w in ss]

                if self.inputistext:
                    self.end_of_data = True
                    if (len(source) < self.batch_size):
                        source.append(ss)
                    #print("Number of sentences: ",len(source))
                    #print(source)

                    break
                #disable max len for now
                if len(ss) > self.maxlen:
                    continue

                source.append(ss)

                if (len(source) >= self.batch_size):
                    break

        except IOError:
            self.end_of_data = True

        if (len(source) <= 0 and self.inputistext is False):
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source
