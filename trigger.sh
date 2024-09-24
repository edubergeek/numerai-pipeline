#!/bin/bash

ROUND_FILE=last_round
cd /ml/numerai-pipeline
last=`cat  $ROUND_FILE`
current=`python trigger.py`

# wait 2 minutes before downloading data
#sleep 120

echo "Launching round $current"
echo $current >$ROUND_FILE

#Set NAI tokens as environment variables
#$NAI_SECRET and $NAI_IDENTITY
source ./.nairc
mkdir -p ./round

# Create a subdirectory for the new round
bash newround.sh
cd round/$current

# Download live data 
python download.py --live --version ${DR_VERSION}

# Transform parquet file(s) to TFRecord files
mkdir ./data
python etl.py --live

# Make and upload predictions
bash predict.sh

exit 0
