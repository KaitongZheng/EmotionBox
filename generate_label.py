import torch
import numpy as np
import os
import sys
import optparse
import time

import config
import utils
from config import device, model as model_config
# from model import PerformanceRNN
from model_label import PerformanceRNN_label

from sequence import EventSeq, Control, ControlSeq

# pylint: disable=E1101,E1102


# ========================================================================
# Settings
# ========================================================================

def getopt():
    parser = optparse.OptionParser()

    parser.add_option('-c', '--control',
                      dest='control',
                      type='string',
                      default=None,
                      help=('control or a processed data file path, '
                            'e.g., "PITCH_HISTOGRAM;NOTE_DENSITY" like '
                            '"2,0,1,1,0,1,0,1,1,0,0,1;4", or '
                            '";3" (which gives all pitches the same probability), '
                            'or "/path/to/processed/midi/file.data" '
                            '(uses control sequence from the given processed data)'))

    parser.add_option('-b', '--batch-size',
                      dest='batch_size',
                      type='int',
                      default=8)

    parser.add_option('-s', '--session',
                      dest='sess_path',
                      type='string',
                      default='save/train.sess',
                      help='session file containing the trained model')

    parser.add_option('-o', '--output-dir',
                      dest='output_dir',
                      type='string',
                      default='output/')

    parser.add_option('-l', '--max-length',
                      dest='max_len',
                      type='int',
                      default=500)

    parser.add_option('-g', '--greedy-ratio',
                      dest='greedy_ratio',
                      type='float',
                      default=1.0)

    parser.add_option('-B', '--beam-size',
                      dest='beam_size',
                      type='int',
                      default=0)

    parser.add_option('-S', '--stochastic-beam-search',
                      dest='stochastic_beam_search',
                      action='store_true',
                      default=False)

    parser.add_option('-T', '--temperature',
                      dest='temperature',
                      type='float',
                      default=1.0)

    parser.add_option('-z', '--init-zero',
                      dest='init_zero',
                      action='store_true',
                      default=False)

    return parser.parse_args()[0]


opt = getopt()

# ------------------------------------------------------------------------

output_dir = opt.output_dir
output_dir=output_dir+'/'+time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
sess_path = 'save/train_label5hrs.sess'  #!!!!!
batch_size = opt.batch_size
max_len = opt.max_len
# greedy_ratio = opt.greedy_ratio  调greedy可以避免一直重复
greedy_ratio = 0.8
control = opt.control    #!!!!!!!!!!!!
use_beam_search = opt.beam_size > 0
stochastic_beam_search = opt.stochastic_beam_search
beam_size = opt.beam_size
temperature = opt.temperature
init_zero = opt.init_zero

if use_beam_search:
    greedy_ratio = 'DISABLED'
else:
    beam_size = 'DISABLED'

assert os.path.isfile(sess_path), f'"{sess_path}" is not a file'
# label_dict={'1,0,1,0,1,1,0,1,0,1,0,1':'C大调','3,0,1,0,1,3,0,3,0,1,0,1':'C大调','2,0,1,0,1,2,0,2,0,1,0,1':'C大调',
#               '2,0,1,1,0,2,0,2,1,0,1,0':'C小调','3,0,1,1,0,3,0,3,1,0,1,0':'C小调','2,0,1,1,0,2,0,2,1,0,1,0':'C小调',}
# # labels=[';1',';2',';3',';4',';5',';6',';7',';8']
labels=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
label_dict={'1 0 0 0':'happy','0 1 0 0':'peaceful',
              '0 0 1 0':'sad','0 0 0 1':'tensional'}

for label in labels:
    listToStr = ' '.join(map(str, label))
    label_name= label_dict[listToStr]
    label = np.asarray(label)
    labels = torch.tensor(label, dtype=torch.float32)
    labels = labels.repeat(1, batch_size, 1).to(device) # 1Xbatch_sizeX labels
    label = repr(label)

    assert max_len > 0, 'either max length or label sequence length should be given'

    # ------------------------------------------------------------------------

    print('-' * 70)
    print('Session:', sess_path)
    print('Batch size:', batch_size)
    print('Max length:', max_len)
    print('Greedy ratio:', greedy_ratio)
    print('Beam size:', beam_size)
    print('Beam search stochastic:', stochastic_beam_search)
    print('Output directory:', output_dir)
    print('labels:', label)
    print('Temperature:', temperature)
    print('Init zero:', init_zero)
    print('-' * 70)

    # ========================================================================
    # Generating
    # ========================================================================

    state = torch.load(sess_path, map_location=device)
    model = PerformanceRNN_label(**state['model_config']).to(device)
    model.load_state_dict(state['model_state'])
    model.eval()
    print(model)
    print('-' * 70)

    if init_zero:
        init = torch.zeros(batch_size, model.init_dim).to(device)
    else:
        init = torch.randn(batch_size, model.init_dim).to(device)  #

    with torch.no_grad():
        if use_beam_search:
            outputs = model.beam_search(init, max_len, beam_size,
                                        labels=labels,
                                        temperature=temperature,
                                        stochastic=stochastic_beam_search,
                                        verbose=True)
        else:
            outputs = model.generate(init, max_len,
                                     labels=labels,
                                     greedy=greedy_ratio,
                                     temperature=temperature,
                                     verbose=True)

    outputs = outputs.cpu().numpy().T  # [batch, sample_length(event_num)],T=transport

    # ========================================================================
    # Saving
    # ========================================================================

    os.makedirs(output_dir, exist_ok=True)

    for i, output in enumerate(outputs):
        name = f'output-{i}{label_name}.mid'
        path = os.path.join(output_dir, name)
        n_notes = utils.event_indeces_to_midi_file(output, path)
        print(f'===> {path} ({n_notes} notes)')

    print('generation done!')