#!/usr/bin/env python

import argparse
import h5py
import numpy as np
import os
import pandas as pd
import sys
import wfdb

from helper_code import sanitize_boolean_value

def get_parser():
    description = 'Prepare the CODE-15% dataset for the Challenge.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--signal_files', type=str, required=True, nargs='*')
    parser.add_argument('-d', '--demographics_file', type=str, required=True)
    parser.add_argument('-l', '--labels_file', type=str, required=True)
    parser.add_argument('-o', '--output_paths', type=str, required=True, nargs='*')
    parser.add_argument('-p', '--positive_examples', type=int, default=None, help='Numero di casi positivi da includere')
    parser.add_argument('-n', '--negative_examples', type=int, default=None, help='Numero di casi negativi da includere')
    return parser

def fix_checksums(record, checksums=None):
    if checksums is None:
        x = wfdb.rdrecord(record, physical=False)
        signals = np.asarray(x.d_signal)
        checksums = np.sum(signals, axis=0, dtype=np.int16)

    header_filename = os.path.join(record + '.hea')
    string = ''
    with open(header_filename, 'r') as f:
        for i, l in enumerate(f):
            if i == 0:
                arrs = l.split(' ')
                num_leads = int(arrs[1])
            if 0 < i <= num_leads and not l.startswith('#'):
                arrs = l.split(' ')
                arrs[6] = str(checksums[i-1])
                l = ' '.join(arrs)
            string += l

    with open(header_filename, 'w') as f:
        f.write(string)

def run(args):
    exam_id_to_age = {}
    exam_id_to_sex = {}

    df = pd.read_csv(args.demographics_file)
    for _, row in df.iterrows():
        exam_id = int(row['exam_id'])
        age = int(row['age'])
        is_male = sanitize_boolean_value(row['is_male'])
        sex = 'Male' if is_male else 'Female'
        exam_id_to_age[exam_id] = age
        exam_id_to_sex[exam_id] = sex

    exam_id_to_chagas = {}
    df = pd.read_csv(args.labels_file)
    for _, row in df.iterrows():
        exam_id = int(row['exam_id'])
        chagas = sanitize_boolean_value(row['chagas'])
        exam_id_to_chagas[exam_id] = bool(chagas)

    lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    sampling_frequency = 400
    units = 'mV'
    gain = 1000
    baseline = 0
    num_bits = 16
    fmt = str(num_bits)

    if len(args.output_paths) == len(args.signal_files):
        signal_files = args.signal_files
        output_paths = args.output_paths
    elif len(args.output_paths) == 1:
        signal_files = args.signal_files
        output_paths = [args.output_paths[0]]*len(args.signal_files)
    else:
        raise Exception('The number of signal files must match the number of output paths.')

    pos_count = 0
    neg_count = 0

    for k in range(len(signal_files)):
        signal_file = signal_files[k]
        output_path = output_paths[k]
        os.makedirs(output_path, exist_ok=True)

        with h5py.File(signal_file, 'r') as f:
            exam_ids = list(f['exam_id'])

            for i in range(len(exam_ids)):
                exam_id = exam_ids[i]
                if exam_id not in exam_id_to_chagas:
                    continue

                chagas = exam_id_to_chagas[exam_id]

                if chagas:
                    if args.positive_examples is not None and pos_count >= args.positive_examples:
                        continue
                else:
                    if args.negative_examples is not None and neg_count >= args.negative_examples:
                        continue

                physical_signals = np.array(f['tracings'][i], dtype=np.float32)
                num_samples, num_leads = np.shape(physical_signals)
                if num_leads != 12:
                    continue

                # Rimozione padding iniziale/finale
                r = 0
                while r < num_samples and np.all(physical_signals[r, :] == 0):
                    r += 1
                s = num_samples
                while s > r and np.all(physical_signals[s-1, :] == 0):
                    s -= 1
                if r >= s:
                    continue
                physical_signals = physical_signals[r:s, :]

                digital_signals = gain * physical_signals
                digital_signals = np.round(digital_signals)
                digital_signals = np.clip(digital_signals, -2**(num_bits-1)+1, 2**(num_bits-1)-1)
                digital_signals[~np.isfinite(digital_signals)] = -2**(num_bits-1)
                digital_signals = np.asarray(digital_signals, dtype=np.int32)

                age = exam_id_to_age[exam_id]
                sex = exam_id_to_sex[exam_id]
                source = 'CODE-15%'
                comments = [f'Age: {age}', f'Sex: {sex}', f'Chagas label: {chagas}', f'Source: {source}']

                record = str(exam_id)
                wfdb.wrsamp(record, fs=sampling_frequency, units=[units]*num_leads, sig_name=lead_names,
                            d_signal=digital_signals, fmt=[fmt]*num_leads, adc_gain=[gain]*num_leads,
                            baseline=[baseline]*num_leads, write_dir=output_path, comments=comments)

                checksums = np.sum(digital_signals, axis=0, dtype=np.int16)
                fix_checksums(os.path.join(output_path, record), checksums)

                if chagas:
                    pos_count += 1
                else:
                    neg_count += 1

                if (args.positive_examples is not None and pos_count >= args.positive_examples) and \
                   (args.negative_examples is not None and neg_count >= args.negative_examples):
                    return

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
