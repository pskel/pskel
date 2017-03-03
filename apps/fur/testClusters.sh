#!/bin/bash

for NUMBER_CLUSTERS in 2 4 6 8 10 12 14 16
do
    ./run.sh 2048 2048 128 128 1 1 $NUMBER_CLUSTERS 16 &>> ./test_${NUMBER_CLUSTERS}.txt
    sleep 1
done
