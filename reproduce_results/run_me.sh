#!/bin/bash

for d in alexnet mnist resnet inception_inference nmt ptb; do
    echo "Generating results for ${d}"
    cd ${d}
    ./run_me.sh
    cd ..
done
