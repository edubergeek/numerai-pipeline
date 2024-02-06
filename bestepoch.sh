#!/bin/bash

model=$1
version=$2
revision=$3
trial=$4
path=./$model/

f=`ls -lt ${path} | grep "${model}v${version}r${revision}t${trial}-e*" | head -1`
bestEpoch=`expr "$f" : '.*-e\([0-9]*\)'`
echo ${bestEpoch}
