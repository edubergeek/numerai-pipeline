#!/bin/bash

docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --net=host -v /home/cdodds/ml/:/ml/ --env="CUDA_VISIBLE_DEVICES=0,1" --gpus 2 nvml /bin/bash /ml/numerai-pipeline/trigger.sh

