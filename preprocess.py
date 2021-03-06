import os
import re
import sys
import torch
import hashlib
from progress.bar import Bar
from concurrent.futures import ProcessPoolExecutor

from sequence import NoteSeq, EventSeq, ControlSeq
import utils
import config

def preprocess_midi(path): #midi file->pretty midi(note_seq)->event_seq->control_seq
    note_seq = NoteSeq.from_midi_file(path)
    note_seq.adjust_time(-note_seq.notes[0].start) #把起始时间设置到0
    event_seq = EventSeq.from_note_seq(note_seq)
    control_seq = ControlSeq.from_event_seq(event_seq)
    even_array=event_seq.to_array()
    return even_array, control_seq.to_compressed_array()

def preprocess_midi_files_under(midi_root, save_dir, num_workers):
    midi_paths = list(utils.find_files_by_extensions(midi_root, ['.mid', '.midi']))
    os.makedirs(save_dir, exist_ok=True)
    out_fmt = '{}-{}.data'
    
    results = []
    executor = ProcessPoolExecutor(num_workers)

    for path in midi_paths:
        try:
            results.append((path, executor.submit(preprocess_midi, path)))
        except KeyboardInterrupt:
            print(' Abort')
            return
        except:
            print(' Error')
            continue
    
    for path, future in Bar('Processing').iter(results):
        print(' ', end='[{}]'.format(path), flush=True)
        name = os.path.basename(path)
        code = hashlib.md5(path.encode()).hexdigest()  # convert path to hashcode
        save_path = os.path.join(save_dir, out_fmt.format(name, code))
        future_res=future.result() #event_seq.to_array(), control_seq.to_compressed_array() 组成的元组
        torch.save(future_res, save_path)

    print('Done')

if __name__ == '__main__':
    # preprocess_midi_files_under(
    #         midi_root=sys.argv[1],
    #         save_dir=sys.argv[2],
    #         num_workers=int(sys.argv[3]))
    preprocess_midi_files_under(
        midi_root=r'.\dataset\midi\NAME',
        save_dir=r'.\dataset\processed\NAME',
        num_workers=1 )

    # python3
    # preprocess.py
    # dataset / midi / NAME
    # dataset / processed / NAME
