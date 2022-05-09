import os
import torch
import itertools, os
import numpy as np
from progress.bar import Bar

import config
import utils
from sequence import EventSeq, ControlSeq

# pylint: disable=E1101
# pylint: disable=W0101
# def expand_label(times,labelseq):
#     labelseq.repeat(times,1)
#     return


class Dataset_label:
    def __init__(self, root, verbose=False): #prepare dataset
        assert os.path.isdir(root), root
        paths = utils.find_files_by_extensions(root, ['.data'])
        self.root = root
        self.samples = []
        self.seqlens = []
        if verbose:
            paths = Bar(root).iter(list(paths))
        for path in paths:
            eventseq, labelseq = torch.load(path)
            labelseq = np.expand_dims(labelseq,0)
            labelseq = labelseq.repeat(len(eventseq),axis=0)
            assert len(eventseq) == len(labelseq)
            self.samples.append((eventseq, labelseq))
            self.seqlens.append(len(eventseq))
        self.avglen = np.mean(self.seqlens)
    
    def batches(self, batch_size, window_size, stride_size):
        indeces = [(i, range(j, j + window_size))
                   for i, seqlen in enumerate(self.seqlens)
                   for j in range(0, seqlen - window_size, stride_size)]
        while True:
            eventseq_batch = []
            labelseq_batch = []
            n = 0
            for ii in np.random.permutation(len(indeces)):
                i, r = indeces[ii]
                eventseq, labelseq = self.samples[i]
                eventseq = eventseq[r.start:r.stop]
                labelseq = labelseq[r.start:r.stop]
                eventseq_batch.append(eventseq)
                labelseq_batch.append(labelseq)
                n += 1
                if n == batch_size:
                    even_stack=np.stack(eventseq_batch, axis=1)
                    label_stack=np.stack(labelseq_batch, axis=1)
                    yield (even_stack,label_stack)
                    eventseq_batch.clear()
                    labelseq_batch.clear()
                    n = 0
    
    def __repr__(self):
        return (f'Dataset(root="{self.root}", '
                f'samples={len(self.samples)}, '
                f'avglen={self.avglen})')
