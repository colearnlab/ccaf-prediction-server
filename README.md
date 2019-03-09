# ccaf-prediction-server

## Running the prediction server

A typical command to run the prediction server looks like this:

`nohup python3 prediction_server.py ../ccaf-web/stores 8 10 ../ccaf-web/prediction_server/ml_predictions.json --verbose &`

Where:

* `nohup` means run without closing after the current SSH session ends
* `python3 prediction_server.py` means run the prediction server Python script
* `../ccaf-web/stores` is the folder where the CCAF stores (per-group log files) are written
* `8` is the maximum number of groups to make predictions for
* `10` is the maximum number of minutes since the last action before a log file will be ignored
* `ml_predictions.json` is the output filename for predictions
* `--verbose` means to print info (which will be written to `nohup.out`) about which log files are being used for predictions

Information about options for the prediction server can be found by running `python3 prediction_server.py --help`
