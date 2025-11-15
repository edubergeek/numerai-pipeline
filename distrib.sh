#!/bin/bash

file=$1*.csv
for d in 0 1 2 3 4 5 6 7 8 9
do
n=`grep "0\.$d" $file|wc -l`
printf "%d0 %d\n" $d $n
done
n=`grep "1\.0" $file|wc -l`
printf "%d %d\n" 100 $n
