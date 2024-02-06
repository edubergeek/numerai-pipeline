#!/bin/bash

ETLDIR=./live

python download.py --live
python etl.py --live --dir $ETLDIR

MODELS='cnn'

for m in $MODELS
do
  #model      arch version round epoch transform	e1_model,arch,ver	e2_model,arch,ver
  grep "^$m" ./predict.conf | while read model path arch version rev trial epoch batch transform
  do
    echo python predict.py --path $path --model $model --version $version --revision $rev --trial $trial --epoch $epoch --batch_size $batch --arch $arch --transform \'$transform\'
    if false
    then
      date >$model.log
      python predict.py --filepat "${ETLDIR}/live_*.tfr" --path $path --model $model --version $version --revision $rev --trial $trial --epoch $epoch --batch_size $batch --arch $arch --transform $transform
      date >>$model.log
      bash distrib.sh $model >>$model.log
    else
      python predict.py --filepat "${ETLDIR}/live_*.tfr" --path $path --model $model --version $version --revision $rev --trial $trial --epoch $epoch --batch_size $batch --arch $arch --transform $transform
      bash distrib.sh $model
    fi
  done
done
