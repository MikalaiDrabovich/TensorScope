#!/bin/bash

for d in alexnet mnist resnet nmt ptb; do
    echo "Generating results for ${d}"
    cd ${d}
    ./run_me.sh
    cd ..
done
