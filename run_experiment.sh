#! /bin/bash

rm -rf ./log
rm -rf ./log_tensorboard
python ./run_recbole_group.py -m GRU4Rec,Ours,SASRec -d ml-1m --config_files recbole/properties/overall.server.yaml