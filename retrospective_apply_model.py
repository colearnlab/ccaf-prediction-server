# Take a trained model and retrospectively apply it to a log file, to allow examination of
# predictions
import argparse
import pickle
import warnings
import re
from collections import OrderedDict

import pandas as pd
import numpy as np
from tqdm import tqdm

import parse_raw_logs
import extract_features


# TODO: import prediction_server and get values from there
MODEL_LABELS = ['code_ontask', 'code_ontasknoint', 'code_peerint', 'code_tasktalk', 'code_silent',
                'code_taclass']
OUTPUT_FEATURES = ['prop_students_acting', 'ratio_2ndmost_to_most_active']


parser = argparse.ArgumentParser()
parser.add_argument('log_file', help='Path to a CSTEPS log file (*.logfile)')
parser.add_argument('prediction_interval', help='Number of seconds between predictions in output '
                    '(e.g., predict every 20 seconds)', type=int)
parser.add_argument('output_file', help='Path to output file, which will be written as a CSV')
args = parser.parse_args()

# Load any required files (models, etc.)
print('Loading models and data')
models = {}
for label in MODEL_LABELS:
    models[label] = {}
    with open('models/model_' + label + '.pkl', 'rb') as infile:
        models[label]['model'] = pickle.load(infile)
    with open('models/model_' + label + '-features.txt') as infile:
        models[label]['features'] = infile.readline().split(',')
possible_events = pd.read_csv('models/all_possible_events.csv').event_type.values
df, pid_map = parse_raw_logs.parse_logfile(args.log_file)  # JSON -> pandas DataFrame

print('Making predictions')
result = []
for clip_start_ms in tqdm(range(df.timestamp.iloc[0], df.timestamp.iloc[-1],
                          args.prediction_interval * 1000)):
    clip_end_ms = clip_start_ms + extract_features.CLIP_LEN_MS - 1
    result.append(OrderedDict({
        'source_log_file': re.sub(r'.*/', '', args.log_file),
        'clip_start_ms': clip_start_ms,
        'clip_end_ms': clip_end_ms,
        'minutes_since_start': '%.3f' % ((clip_start_ms - df.timestamp.iloc[0]) / 1000 / 60)
    }))
    with warnings.catch_warnings():  # Ignore some repeated NumPy warnings
        warnings.filterwarnings('ignore', r'invalid value encountered.*')
        warnings.filterwarnings('ignore', r'Mean of empty slice.*')
        features = {'start_sec_into_log': (clip_start_ms - df.timestamp.iloc[0]) / 1000}
        # Extract cumulative features
        cum_df = df[df.timestamp <= clip_end_ms]
        for k, v in extract_features.extract_features(
                cum_df, len(pid_map), possible_events).items():
            features['cum_' + k] = v
        # Extract clip features
        clip_df = cum_df[cum_df.timestamp >= clip_start_ms]
        features.update(extract_features.extract_features(clip_df, len(pid_map), possible_events))

    # Apply models
    for label, model in models.items():
        x = np.array([features[f] if f in features else np.nan for f in model['features']])
        # Replace missing values with zero (TODO: something cleverer might be better)
        x[np.isnan(x)] = 0
        result[-1]['pred_' + label] = model['model'].predict_proba(x.reshape(1, -1))[0][1]

    # Add certain features to output that may be useful themselves
    for feat in OUTPUT_FEATURES:
        result[-1][feat] = features[feat] if feat in features else ''

print('Saving result')
pd.DataFrame.from_records(result).to_csv(args.output_file, index=False)
