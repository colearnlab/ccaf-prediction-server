# Take real log files and update all the timestamps so that they look like new log files, to test
# the prediction server. This assumes that you have the original CSTEPS log file data from the
# Cyberlearning server, in a folder called "data", and that you run this script from the parent
# folder of ccaf-prediction-server.
import random
import time


INPUT_LOGS = ['data/2017-11-28/ADA/log_350.logfile', 'data/2017-11-28/ADA/log_351.logfile']
OUTPUT_LOG = 'ccaf-prediction-server/test_updated_log%d.logfile'


for log_i, input_log in enumerate(INPUT_LOGS):
    print('Loading')
    lines = []
    with open(input_log) as infile:
        for line in infile:
            lines.append(line.strip())

    # Randomly chop off lines at some point to simulate mid-session.
    lines = lines[:random.randint(0, len(lines))]

    # Update times relative to now and save to file.
    print('Saving')
    last_time = int(lines[-1][8:21])
    now = int(time.time() * 1000)
    with open(OUTPUT_LOG % log_i, 'w') as outfile:
        for line in lines:
            outfile.write(line[:8] + str(int(line[8:21]) - last_time + now) + line[21:] + '\n')
