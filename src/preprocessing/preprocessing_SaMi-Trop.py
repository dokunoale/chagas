#!/usr/bin/env python

# Import libraries
import argparse
import h5py
import numpy as np
import os
import pandas as pd
import sys
import wfdb

# Utils
def is_integer(x):
    try:
        int(x)
        return True
    except:
        return False

def is_boolean(x):
    return str(x).lower() in ['true', 'false', '1', '0']

def sanitize_boolean_value(x):
    if str(x).lower() in ['true', '1']:
        return True
    elif str(x).lower() in ['false', '0']:
        return False
    else:
        raise ValueError(f'Invalid boolean value: {x}')

def get_parser():
    """ Create an argument parser for the preprocessing script. """
    description = "Preprocessing script for SaMi-Trop dataset.\n\n"
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--signal_file', type=str, required=True,
                        help='Path to the HDF5 signal file')
    parser.add_argument('-d', '--demographics_file', type=str, required=True,
                        help='Path to the CSV demographics file')
    parser.add_argument('-o', '--output_path', type=str, required=True,
                        help='Path to the output folder')
    parser.add_argument('-c', '--count', type=int, required=False, default=None,
                        help='Maximum number of data to preprocess (default: all)')
    parser.add_argument('--clean_output', action='store_true',
                        help='Clean the output folder before starting')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip records already present in the output folder')
    return parser

def fix_checksums(record, checksums=None):
    """ Fix the checksums from the Python WFDB library. """
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
    if args.clean_output and os.path.isdir(args.output_path):
        print(f"[INFO] Cleaning output directory: {args.output_path}")
        for f in os.listdir(args.output_path):
            full_path = os.path.join(args.output_path, f)
            if os.path.isfile(full_path):
                os.remove(full_path)

    os.makedirs(args.output_path, exist_ok=True)

    df = pd.read_csv(args.demographics_file)
    exam_ids = []
    exam_id_to_age = {}
    exam_id_to_sex = {}

    for _, row in df.iterrows():
        exam_id = int(row['exam_id']) if is_integer(row['exam_id']) else None
        age = int(row['age']) if is_integer(row['age']) else None
        is_male = sanitize_boolean_value(row['is_male']) if is_boolean(row['is_male']) else None
        if exam_id is not None and age is not None and is_male is not None:
            exam_ids.append(exam_id)
            exam_id_to_age[exam_id] = age
            exam_id_to_sex[exam_id] = 'Male' if is_male else 'Female'

    lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    sampling_frequency = 400
    gain = 1000
    baseline = 0
    num_bits = 16
    fmt = str(num_bits)
    units = 'mV'
    max_samples = 2800  # 7 secondi

    num_exam_ids = len(exam_ids)
    if args.count is not None:
        num_exam_ids = min(num_exam_ids, args.count)

    processed = 0
    with h5py.File(args.signal_file, 'r') as f:
        for i in range(len(exam_ids)):
            if args.count is not None and processed >= args.count:
                break

            exam_id = exam_ids[i]
            record_name = str(exam_id)
            out_dat = os.path.join(args.output_path, f"{record_name}.dat")

            if args.skip_existing and os.path.exists(out_dat):
                print(f"[SKIP] {record_name} già esistente")
                continue

            physical_signals = np.array(f['tracings'][i], dtype=np.float32)
            num_samples, num_leads = physical_signals.shape
            assert num_leads == 12

            # Rimuove padding di zeri
            r = 0
            while r < num_samples and np.all(physical_signals[r, :] == 0):
                r += 1
            s = num_samples
            while s > r and np.all(physical_signals[s-1, :] == 0):
                s -= 1

            if r >= s:
                continue

            physical_signals = physical_signals[r:s, :]

            if physical_signals.shape[0] > max_samples:
                physical_signals = physical_signals[:max_samples, :]
            elif physical_signals.shape[0] < max_samples:
                pad_len = max_samples - physical_signals.shape[0]
                padding = np.zeros((pad_len, num_leads), dtype=physical_signals.dtype)
                physical_signals = np.vstack((physical_signals, padding))

            digital_signals = gain * physical_signals
            digital_signals = np.round(digital_signals)
            digital_signals = np.clip(digital_signals, -2**(num_bits-1)+1, 2**(num_bits-1)-1)
            digital_signals[~np.isfinite(digital_signals)] = -2**(num_bits-1)
            digital_signals = np.asarray(digital_signals, dtype=np.int32)

            comments = [
                f'Age: {exam_id_to_age[exam_id]}',
                f'Sex: {exam_id_to_sex[exam_id]}',
                f'Chagas label: True',
                f'Source: SaMi-Trop'
            ]

            wfdb.wrsamp(record_name, fs=sampling_frequency, units=[units]*num_leads,
                        sig_name=lead_names, d_signal=digital_signals,
                        fmt=[fmt]*num_leads, adc_gain=[gain]*num_leads,
                        baseline=[baseline]*num_leads, write_dir=args.output_path,
                        comments=comments)

            checksums = np.sum(digital_signals, axis=0, dtype=np.int16)
            fix_checksums(os.path.join(args.output_path, record_name), checksums)
            print(f"[OK] Processato {record_name}")
            processed += 1

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
