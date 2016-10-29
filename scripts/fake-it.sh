#!/bin/bash
#
# Easy to use script to run neural-style
#

if [ $# -lt 2 ]
  then
    echo "USAGE: fake-it.sh content-image style-image"
    echo "Current available contents: "
    for i in `ls contents`; do echo -e "\t${i}"; done
    echo "Current available styles: "
    for i in `ls styles`; do echo -e "\t${i}"; done
	echo "Example: fake-it.sh goldengate.jpg vangogh.jpg"
	echo "To add more contents or styles, simply add them to the folders above"
    exit 1
fi

# Check docker permissions
if groups $USER | grep &>/dev/null '\bdocker\b'; then SU=""
else SU="sudo"; fi

output_name="output/${1%.*}_by_${2%.*}.png"
time $SU nvidia-docker run --rm -v $(pwd):/images lherrera/neural-style -backend cudnn -cudnn_autotune -normalize_gradients -init image -content_weight 100 -style_weight 1500 -content_image contents/$1 -style_image styles/$2 -output_image $output_name



