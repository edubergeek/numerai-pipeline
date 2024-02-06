#!/bin/bash

source ./nairc
#data=`ls *.zip`
#echo data: $data
#base=`basename $data .zip`
#echo base: $base
year=`date +%Y`
echo year: $year
#week=`date +%W`
#echo week: $week
jday=`date +%j`
echo week: $jday
#round=`expr $base : 'numerai_dataset_*\([0-9]*\)'`
#echo round: $round

#echo initializing year $year week $week round $round ...
echo initializing year $year week $week ...
weekDir=${year}d${jday}

echo getting code
git clone git@github.com:edubergeek/py-nai.git $weekDir

cd $weekDir
#mv $data $weekDir
python tournament.py | tee round.log
round=`head -1 round.log`
ln -s ../`basename \`pwd\`` ../round/$round
echo $round >.round

#unzip -o $data

ls -lt
echo Round $round is ready
