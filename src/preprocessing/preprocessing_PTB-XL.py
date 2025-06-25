#!/usr/bin/env python

# Load libraries.
import argparse
import numpy as np
import os
import os.path
import pandas as pd
import shutil
import sys
import wfdb

from helper_code import find_records, get_signal_files, is_integer

# Parse arguments.
def get_parser():
    description = 'Prepare the PTB-XL database for use in the Challenge.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input_folder', type=str, required=True)  # records100 or records500
    parser.add_argument('-d', '--ptbxl_database_file', type=str, required=True)  # ptbxl_database.csv
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-c', '--count', type=int, default=None, required=False,
                        help='Number of records to process (default: all)')
    parser.add_argument('--skipexisting', action='store_true',
                        help='Skip processing records that already exist in output folder')
    parser.add_argument('--clean', action='store_true',
                        help='Clean the output folder before processing')
    return parser

# Suppress stdout for noisy commands.
from contextlib import contextmanager
@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = stdout

# Fix the checksums from the Python WFDB library.
def fix_checksums(record, checksums=None):
    if checksums is None:
        x = wfdb.rdrecord(record, physical=False)
        signals = np.asarray(x.d_signal)
        checksums = np.sum(signals, axis=0, dtype=np.int16)

    header_filename = record + '.hea'
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

# Run script.
def run(args):
    # Load the demographic information.
    df = pd.read_csv(args.ptbxl_database_file, index_col='ecg_id')

    # Identify the header files.
    records = find_records(args.input_folder)

    # Clean output folder if requested.
    if args.clean and os.path.isdir(args.output_folder):
        print(f'Cleaning output folder {args.output_folder}...')
        for filename in os.listdir(args.output_folder):
            file_path = os.path.join(args.output_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    os.makedirs(args.output_folder, exist_ok=True)

    # Process records with optional count limit.
    processed_count = 0
    for record in records:
        if args.count is not None and processed_count >= args.count:
            break

        record_basename = os.path.basename(record)
        output_header_file = os.path.join(args.output_folder, record_basename + '.hea')
        
        # Skip existing if requested.
        if args.skipexisting and os.path.isfile(output_header_file):
            print(f'Skipping existing record {record_basename}')
            continue

        # Extract the demographics data.
        ecg_id = int(record_basename.split('_')[0])
        row = df.loc[ecg_id]

        recording_date_string = row['recording_date']
        date_string, time_string = recording_date_string.split(' ')
        yyyy, mm, dd = date_string.split('-')
        date_string = f'{dd}/{mm}/{yyyy}'

        age = row['age']
        age = int(age) if is_integer(age) else float(age)

        sex = row['sex']
        if sex == 0:
            sex = 'Male'
        elif sex == 1:
            sex = 'Female'
        else:
            sex = 'Unknown'

        # Assume that all of the patients are negative for Chagas disease.
        label = False

        # Specify the label.
        source = 'PTB-XL'

        input_header_file = os.path.join(args.input_folder, record + '.hea')

        with open(input_header_file, 'r') as f:
            input_header = f.read()

        lines = input_header.split('\n')
        record_line = ' '.join(lines[0].strip().split(' ')[:4]) + '\n'
        signal_lines = '\n'.join(l.strip() for l in lines[1:] \
            if l.strip() and not l.startswith('#')) + '\n'
        comment_lines = '\n'.join(l.strip() for l in lines[1:] \
            if l.startswith('#') and not any((l.startswith(x) for x in ('# Age:', '# Sex:', '# Height:', '# Weight:', '# Chagas label:', '# Source:')))) + '\n'

        record_line = record_line.strip() + f' {time_string} {date_string} ' + '\n'
        signal_lines = signal_lines.strip() + '\n'
        comment_lines = comment_lines.strip() + f'# Age: {age}\n# Sex: {sex}\n# Chagas label: {label}\n# Source: {source}\n'

        output_header = record_line + signal_lines + comment_lines

        with open(output_header_file, 'w') as f:
            f.write(output_header)

        # Copy the signal files into the output folder (flat, no subfolders).
        signal_files = get_signal_files(input_header_file)
        for input_signal_file in signal_files:
            signal_filename = os.path.basename(input_signal_file)
            output_signal_file = os.path.join(args.output_folder, signal_filename)
            if os.path.isfile(input_signal_file):
                shutil.copy2(input_signal_file, output_signal_file)
            else:
                raise FileNotFoundError(f'{input_signal_file} not found.')

        # Fix checksums.
        fix_checksums(os.path.join(args.output_folder, record_basename))

        processed_count += 1
        print(f'Processed record {record_basename} ({processed_count}/{len(records) if args.count is None else args.count})')

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))