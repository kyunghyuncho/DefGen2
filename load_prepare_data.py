import sys

import cPickle as pkl
import numpy
import gc

def prepare_data(seqs, contexts, maxlen=None, normalize_context=True):
    lengths = [len(s) for s in seqs]

    if maxlen != None:
        new_seqs = []
        new_contexts = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, contexts):
            if l < maxlen:
                new_seqs.append(s)
                new_contexts.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        contexts = new_contexts
        seqs = new_seqs

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)+1

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
    for idx, s in enumerate(seqs):
        x[:lengths[idx],idx] = s
        x_mask[:lengths[idx]+1,idx] = 1.

    if normalize_context:
        contexts = numpy.array(contexts)
        contexts = contexts / numpy.sqrt((contexts ** 2).sum(1)[:,None])

    return x, x_mask, contexts.astype('float32')

def load_data(data_name, n_words=20000, valid_portion=0.1):
    with open(data_name, 'rb') as f:
        x = pkl.load(f)
        y = pkl.load(f)

    n_samples = len(x)
    rndidx = numpy.random.permutation(n_samples)

    n_valid = numpy.round(n_samples * valid_portion)

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    x_val = [None] * n_valid
    print 'Valid: ',
    for jj, ii in enumerate(rndidx[-n_valid:]):
        x_val[jj] = x[ii]
        x[ii] = None
        print ii,
    y_val = remove_unk([y[ii] for ii in rndidx[-n_valid:]])
    print

    print 'Train: ',
    x_train = [None] * (n_samples - n_valid)
    for jj, ii in enumerate(rndidx[:-n_valid]):
        x_train[jj] = x[ii]
        x[ii] = None
        print ii,
    x = x_train
    y = remove_unk([y[ii] for ii in rndidx[:-n_valid]])
    print

    return (x,y), (x_val,y_val), None

