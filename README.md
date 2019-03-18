# ccaf-prediction-server

## Installation

Clone this repository to the machine running _ccaf-web_, and install the prerequisites. The prerequisites are:

* Python 3.5, 3.6, or 3.7
* _pandas_, _scikit-learn_, and _tqdm_ packages, which can be installed with:
    * `pip3 install pandas scikit-learn tqdm`
    * If you need to install these packages for all users, prepend `sudo -H ` to the `pip3` command

## Running the prediction server

A typical command to run the prediction server looks like this:

`nohup python3 -u prediction_server.py ../ccaf-web/stores 8 10 ../ccaf-web/prediction_server/ml_predictions.json --verbose &`

Where:

* `nohup` means run without closing after the current SSH session ends
* `python3 -u prediction_server.py` means run the prediction server Python script, writing output to nohup.out immediately (`-u` means unbuffered)
* `../ccaf-web/stores` is the folder where the CCAF stores (per-group log files) are written
* `8` is the maximum number of groups to make predictions for
* `10` is the maximum number of minutes since the last action before a log file will be ignored
* `ml_predictions.json` is the output filename for predictions
* `--verbose` means to print info (which will be written to `nohup.out`) about which log files are being used for predictions

Information about options for the prediction server can be found by running `python3 prediction_server.py --help`

## Stopping the prediction server

There isn't a polite way to stop the server, so you will have to simply kill it.

1. Discover the process ID by running `ps aux | grep prediction_server'; if it is running you might see something like:
    * `pnb     124235 15.0  2.4 676904 196832 pts/1   S    14:54   0:01 python3 prediction_server.py ../ccaf-web/stores 8 10 ../ccaf-web/prediction_server/ml_predictions.json --verbose`
    * The second field, `124235` in this case, is the process ID
2. Kill the process: `kill 124235`
    * If the process was started by another user, you may have to prepend the `kill` command with `sudo `
