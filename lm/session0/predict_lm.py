import os

from lm import (build_sampler, gen_sample, load_params,
                 init_params, init_tparams, build_model, pred_probs, prepare_data)
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import shared
import pickle as pkl
from data_iterator import TextIterator
import numpy
import theano
from sys import argv

profile = False

def main(saveto, test_dataset, out_prob_file, dictionary, n_words, valid_batch_size, maxlen):

    valid = TextIterator(test_dataset,
                         dictionary,
                         n_words_source=n_words,
                         batch_size=valid_batch_size,
                         maxlen=maxlen)

    trng = RandomStreams(1234)
    use_noise = shared(numpy.float32(0.))

    # load dictionary
    with open(dictionary, 'rb') as f:
        worddicts = pkl.load(f)

    # invert dictionary
    worddicts_r = dict()
    for kk, vv in worddicts.iteritems():
        worddicts_r[vv] = kk

    #model options
    with open('%s.pkl' % saveto, 'rb') as f:
        model_options = pkl.load(f)

    print ('Building model')
    params = init_params(model_options)

    # load model parameters and set theano shared variables
    params = load_params(saveto, params)
    tparams = init_tparams(params)

    # build the symbolic computational graph
    trng, use_noise, \
        x, x_mask, \
        opt_ret, \
        cost = \
        build_model(tparams, model_options)
    inps = [x, x_mask]

    print ('Buliding sampler')
    f_next = build_sampler(tparams, model_options, trng)

    # before any regularizer
    print ('Building f_log_probs...')
    f_log_probs = theano.function(inps, cost, profile=profile)
    print ('Done')
    #print(f_log_probs)
    valid_errs = pred_probs(f_log_probs, prepare_data,
                            model_options, valid)

    #print(valid_errs)
    print("Length of Input Data is: ", len(valid_errs))
    pkl.dump( valid_errs, open( out_prob_file, "wb" ) )
    #f = open('smic_lmscores.txt', 'w')
    #f.write('\n'.join(valid_errs))
    #f.close()
    valid_err = valid_errs.mean()

    #print("Yo this the error: ", valid_err)
    return valid_err

if __name__ == '__main__':
    params = {
        'model': ['lm/models/model_session0.npz'],
        'dim_word': [512],
        'dim': [1024],
        'n-words': [30000],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [True]}

    testdata = argv[1]
    out_prob_file = argv[2]
    print("your testdata is ", testdata)
    print("your testdata is ", out_prob_file)
    logerror = main(saveto=params['model'][0],
        test_dataset = testdata,
        out_prob_file = out_prob_file,
        dictionary = 'lm/data/nyt_toutanova_lm.txt.tok.pkl',
        n_words = params['n-words'][0],
        valid_batch_size = 16,
        maxlen = 415
        )
    #print(logerror)
