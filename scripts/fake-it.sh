#!/bin/bash
if [ $# -lt 2 ]
  then
    echo "USAGE: fake-it.sh content-image style-image"
    echo "Current available styles: "
    for i in `ls styles`; do echo ${i%.*}; done
    exit 1
fi
output_name="output/${1%.*}_by_${2%.*}.png""
time sudo nvidia-docker run --rm -v $(pwd):/images lherrera/neural-style -backend cudnn -cudnn_autotune -content_image contents/$1 -style_image styles/$2 -output_image $output_name 



