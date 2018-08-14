#!/bin/bash

#for d in resnet alexnet mnist ptb nmt inception_inference; do
#for d in resnet alexnet mnist ptb nmt; do
for d in resnet; do
    echo "Generating results for ${d}"
    cd ${d}
    ./run_me.sh
    cd ..
done

