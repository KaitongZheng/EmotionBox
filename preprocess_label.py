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

def label_to_array(label):
    switch = {"happy":utils.one_hot(0,4),
              "peaceful": utils.one_hot(1, 4),
              "sad": utils.one_hot(2, 4),
              "tensional": utils.one_hot(3, 4),
              }
    one_hot_label=switch[label]
    return one_hot_label

def preprocess_midi(path, label):  # midi file->pretty midi(note_seq)->event_seq->control_seq
    note_seq = NoteSeq.from_midi_file(path)
    note_seq.adjust_time(-note_seq.notes[0].start)
    event_seq = EventSeq.from_note_seq(note_seq)
    # control_seq = ControlSeq.from_event_seq(event_seq)
    even_array = event_seq.to_array()
    label_array = label_to_array(label)
    print(label_array)
    return even_array, label_array


def preprocess_midi_files_under(midi_root, save_dir, num_workers):
    midi_paths = list(utils.find_files_by_extensions(midi_root, ['.mid', '.midi'],emotion_label=True))
    os.makedirs(save_dir, exist_ok=True)
    out_fmt = '{}-{}.data'

    # results = []
    # executor = ProcessPoolExecutor(num_workers)

    for path, label in midi_paths:

        name = os.path.basename(path)
        code = hashlib.md5(path.encode()).hexdigest()  # convert path to 唯一的hashcode
        save_path = os.path.join(save_dir, out_fmt.format(name, code))
        print(label)
        print('\n')
        event_label = preprocess_midi(path, label)  # event_seq.to_array(), control_seq.to_compressed_array() 组成的元组
        torch.save(event_label, save_path)
        print(' ', end='[{}]'.format(path), flush=True)


    # executor = ProcessPoolExecutor(num_workers)
    #
    # for path, label in midi_paths:
    #     try:
    #         results.append((path, executor.submit(preprocess_midi, path, label)))
    #     except KeyboardInterrupt:
    #         print(' Abort')
    #         return
    #     except:
    #         print(' Error')
    #         continue
    #
    # for path, future in Bar('Processing').iter(results):
    #     print(' ', end='[{}]'.format(path), flush=True)
    #     name = os.path.basename(path)
    #     code = hashlib.md5(path.encode()).hexdigest()  # convert path to 唯一的hashcode
    #     save_path = os.path.join(save_dir, out_fmt.format(name, code))
    #     future_res = future.result()  # event_seq.to_array(), control_seq.to_compressed_array() 组成的元组
    #     torch.save(future_res, save_path)
    print('Preprocession Done')


if __name__ == '__main__':
    preprocess_midi_files_under(
        midi_root="dataset/midi/emotion_classical",
        save_dir="dataset/processed/emotion_classical",
        num_workers=2)

    # python3
    # preprocess.py
    # dataset / midi / NAME
    # dataset / processed / NAME
