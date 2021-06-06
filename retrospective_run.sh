#!/bin/bash
# Search for files in subdirectories and apply the retrospective_apply_model.py script to them

if [ -z "$1" ]; then
    echo "Please specify an input directory to recursively search for *.logfile"
    exit
fi
in_dir="$1"
if [[ "$in_dir" =~ .*/$ ]]; then
    in_dir=`echo "$in_dir" | sed 's/.$//'`  # Remove trailing forward slash
fi
if [ -z "$2" ]; then
    echo "Please specify a prediction interval in seconds (e.g., 20)"
    exit
fi
if [ -z `which python3 | grep csteps2apply` ]; then
    echo "This should be run from the csteps2apply conda environment"
    exit
fi

for fname in `find "$in_dir" -name '*.logfile'`; do
    echo "Processing $fname"
    if [ -f "$fname-predictions.csv" ]; then
        echo "Output already exists; skipping"
        continue
    fi
    echo "Output will be $fname-predictions.csv"
    # Having `|| exit` here allows for Ctrl+C to stop the whole script
    python3 retrospective_apply_model.py "$fname" $2 "$fname-predictions.csv" || exit
done

out_file="$in_dir/combined_predictions.csv"
echo "Combining all predictions into one CSV: $out_file"
include_header=1
for fname in `find "$in_dir" -name '*-predictions.csv'`; do
    echo "$fname"
    if [ $include_header = 1 ]; then
        cp "$fname" "$out_file"
        include_header=0
    else
        tail -n +2 "$fname" >> "$out_file"
    fi
done

echo "Done with all files"
