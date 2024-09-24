#!/bin/bash

source ./.nairc
year=`date +%Y`
echo year: $year
jday=`date +%j`
echo week: $jday

echo initializing year $year week $week ...
weekDir=${year}d${jday}

echo Preparing round directory $weekDir
mkdir -p $weekDir
cp bestepoch.sh distrib.sh predict.sh download.py etl.py nai*.py numerai.py predict.py tournament.py predict.conf $weekDir
# fix up model paths
sed -i -e 's/numerai-pipeline/./g' $weekDir/predict.conf

cd $weekDir
python tournament.py | tee round.log
round=`head -1 round.log`
ln -s ../`basename \`pwd\`` ../round/$round
echo $round >.round

ls -lt
echo Round $round is ready
