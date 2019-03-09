import argparse
import time
import os
import pickle
import json
import warnings
import re

import pandas as pd
import numpy as np

import parse_raw_logs
import extract_features


DATA_DIR = './'  # Can change for testing if needed.
MODEL_LABELS = ['code_ontask', 'code_ontasknoint', 'code_peerint', 'code_tasktalk', 'code_silent',
                'code_taclass']
OUTPUT_FEATURES = ['prop_students_acting', 'ratio_2ndmost_to_most_active']


parser = argparse.ArgumentParser()
parser.add_argument('logfile_dir', help='Path to directory with the CSTEPS log files (*.logfile)')
parser.add_argument('max_files', help='Maximum number of log files to consider (the newest)',
                    type=int)
parser.add_argument('max_age_minutes', help='Maximum age of log files to consider', type=int)
parser.add_argument('output_file', help='Path to output file, which will be regularly overwritten')
parser.add_argument('-v', '--verbose', help='Print logging info to STDOUT', action='store_true')
args = parser.parse_args()


def verbose_print(msg):
    if args.verbose:
        print(int(time.time() * 1000), msg)


# Load any required files (models, etc.) in advance
verbose_print('Loading models and data')
models = {}
for label in MODEL_LABELS:
    models[label] = {}
    with open(DATA_DIR + 'model_' + label + '.pkl', 'rb') as infile:
        models[label]['model'] = pickle.load(infile)
    with open(DATA_DIR + 'model_' + label + '-features.txt') as infile:
        models[label]['features'] = infile.readline().split(',')
possible_events = pd.read_csv(DATA_DIR + 'all_possible_events.csv').event_type.values

# Start an infinite loop to make predictions
verbose_print('Starting predictions')
while True:
    time.sleep(.1)  # To avoid dominating the CPU.

    # Build a sorted list of log files and keep only the max num. of newest files.
    ages_sec = {}
    for fname in os.listdir(args.logfile_dir):
        if fname.endswith('.logfile') or re.match(r'[0-9]+', fname):
            ages_sec[fname] = time.time() - os.path.getmtime(os.path.join(args.logfile_dir, fname))
    ages_sec = sorted(ages_sec.items(), key=lambda x: x[1])[:args.max_files]
    log_files = [os.path.join(args.logfile_dir, f) for f, age in ages_sec
                 if age / 60 < args.max_age_minutes]
    if len(log_files) > 0:
        verbose_print('Found ' + str(len(ages_sec)) + ' log(s), processing ' + str(len(log_files)))

    # Process each log file.
    results = []
    for log_file in log_files:
        verbose_print('Parsing ' + log_file)
        df, pid_map = parse_raw_logs.parse_logfile(log_file)  # JSON -> pandas DataFrame
        with warnings.catch_warnings():  # Ignore some repeated NumPy warnings
            warnings.filterwarnings('ignore', r'invalid value encountered.*')
            warnings.filterwarnings('ignore', r'Mean of empty slice.*')
            # Extract clip features.
            verbose_print('Extracting clip features')
            clip_start_ms = int(time.time() * 1000) - extract_features.CLIP_LEN_MS
            try:
                clip_df = df[df.timestamp > clip_start_ms]
                features = extract_features.extract_features(clip_df, len(pid_map), possible_events)
            except Exception as e:  # Something very unexpected in log file
                verbose_print('Skipping ' + log_file + ' due to clip parsing error')
                verbose_print(e)
                continue
            features['start_sec_into_log'] = (clip_start_ms - df.timestamp.iloc[0]) / 1000
            # Extract cumulative features.
            verbose_print('Extracting cumulative features')
            try:
                cum_features = extract_features.extract_features(df, len(pid_map), possible_events)
            except Exception as e:
                verbose_print('Skipping ' + log_file + ' due to cumulative parsing error')
                verbose_print(e)
                continue
            for k, v in cum_features.items():
                features['cum_' + k] = v

        # Apply models.
        group_result = {'source_log_file': re.sub(r'.*/', '', log_file)}
        for label, model in models.items():
            verbose_print('Applying model: ' + label)
            x = np.array([features[f] if f in features else np.nan for f in model['features']])
            # Replace missing values with zero (TODO: something cleverer might be better).
            x[np.isnan(x)] = 0
            group_result['pred_' + label] = model['model'].predict_proba(x.reshape(1, -1))[0][1]

        # Add certain features to output that may be useful themselves.
        for feat in OUTPUT_FEATURES:
            group_result[feat] = features[feat] if feat in features else ''
        results.append(group_result)

    # Output results.
    if len(results) > 0:
        verbose_print('Saving predictions')
    with open(args.output_file + '.tmp', 'w') as outfile:
        json.dump(results, outfile)
    try:
        # This is an atomic operation per Python docs.
        os.replace(args.output_file + '.tmp', args.output_file)
    except:
        # Extremely unlikely that the file will be in use, so we'll just try again next time.
        verbose_print('Error: Could not write predictions because output file was in use.')
