#!/bin/bash

AREA=(1 2 3 4 5 6)
seed=1

for test_area in "${AREA[@]}"
do
    option="--exp_name=mysemsegv4_${seed}_${test_area} --test_area=${test_area}
            --batch_size=12 --test_batch_size=6 --use_sgd= -k=21
            --seed=${seed}"
    cmd="python main_semseg.py ${option}"
    echo $cmd
    eval $cmd
done
