#!/bin/bash
# train models 
trainAll=true
trainEra=false

# Launch tensorboard if needed
useTensorboard=0
tb=`ps -ef|grep tensorboard | wc -l`
case $((tb*useTensorboard)) in
1) echo "Launching Tensorboard"
   tensorboard --logdir=./logs &
   tensorboard='--tensorboard'
   ;;
0) echo "Tensorboard Disabled"
   tensorboard=''
   ;;
*) echo "Tensorboard is running"
   tensorboard='--tensorboard'
   ;;
esac

# required parameters:
#  model
#  round 1
#  revision
#  transform
#  arch
#  batch_size
#  train | trainera
#  
# default parameter values:
#  trial 1
#  epoch 0
#  patience 0
#  threshold 1e-5
#  lr 1e-3
#  epsilon 1.0
#  epochs 20
#  datadir ./data
#  trainpat train*.tfr
#  validpat valid*.tfr
#  loss mse
#  monitor val_loss

# Autotrain Config:
cat <<EOF >train.conf
model		target	ver	rev	trial	arch	epoch	epochs	batch	lr	epsilon	loss	monitor		mode	begin	transform
mlp		36	1	2	1	NR	0	50	1024	1e-4	0.0	mae	val_loss	all	0	NaN,2,2|Slice,0,2376
cnn 		37	1	1	1	NR	0	50	1024	1e-4	0.0	mae	val_loss	all	0	NaN,2,2|Slice,0,2376|YX,2376,1
EOF

MODELS='mlp cnn'

for m in $MODELS
do
  grep "^$m" train.conf | while read model target version rev trial arch epoch epochs batch lr epsilon loss monitor mode begin transform
  do
    case $mode in
      all)
        python train.py --begin $begin --model $model --version $version --revision $rev --trial $trial --epoch $epoch --arch $arch --transform $transform --round 1 --epochs $epochs --batch_size $batch --lr $lr --epsilon $epsilon --loss $loss --monitor $monitor --target $target --train $tensorboard
	# poke the best epoch into predict.conf
	epoch=`bash bestepoch.sh $model $version $revision $trial`
	echo "Best epoch is $epoch"
	#grep -v "^$model" predict.conf >temp.conf
	#grep "^$model" predict.conf | awk -v epoch=$epoch '{print $1,$2,$3,$4,$5,$6,epoch,$8,$9}' >>temp.conf
	#mv predict.conf predict.old
	#mv temp.conf predict.conf
        ;;
      era)
        python train.py --model $model --version $version --trial $trial --arch $arch --transform $transform --round 1 --epochs $epochs --batch_size $batch --lr $lr --epsilon $epsilon --loss $loss --monitor $monitor --target $target --trainera $tensorboard
        ;;
      *)
        echo mode error in train.conf
        ;;
    esac
  done
done
