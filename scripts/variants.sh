#!/bin/bash
#
# Given a set of content images and a set of style images, generates
# all possible combinations of content and styles. For each possible
# combination several variants are tested, altering content to style weight and
# scaling of style.
#
# @author: Álvaro Barbero Jiménez

DEFAULT_WEIGHTS="375 750 1500 2000 5000 10000"
DEFAULT_SCALES="1"

HELP="
Generate neural-style variants of several contents and styles.

Arguments:
	--contents list of content files to blend.
	--styles list of style files to blend.
	--weights list of styles weights to try. Default: $DEFAULT_WEIGHTS
	--scales list of style scales to try. Default: $DEFAULT_SCALES

All content and style files must be located in the current folder or any of its subfolders.
"

# Auxiliary functions

function filename {
	# Returns just the file name of a file, without path or extension
	name=$(basename $1)
	echo ${name%.*}
}

function outname {
	# Builds the name of an output image given all its blend parameters
	content=$(filename $1)
	style=$(filename $2)
	weight=$3
	scale=$4
	echo "${content}_${style}_sw${weight}_ss${scale}.png"
}

# Parse arguments
contents=""
styles=""
scales=""
weights=""
for i in $@; do
	if [ "$i" = "-h" ] || [ "$i" = "--help" ] ; then
		echo "$HELP"
		exit 0
	elif [[ $i == --* ]]; then
		mode=$i
	elif [ "$mode" = "--contents" ]; then
		contents="$contents $i"
	elif [ "$mode" = "--styles" ]; then
		styles="$styles $i"
	elif [ "$mode" = "--scales" ]; then
		scales="$scales $i"
	elif [ "$mode" = "--weights" ]; then
		weights="$weights $i"
	else
		echo "$HELP"
		echo "ERROR: unrecognized argument $i"
		exit 1
	fi
done

# Check for mandatory arguments
if [[ -z $contents ]]; then
	echo "$HELP"
	echo "ERROR: --contents parameter is mandatory"
	exit 1
fi
if [[ -z $styles ]]; then
	echo "$HELP"
	echo "ERROR: --styles parameter is mandatory"
	exit 1
fi

# Use default ranges for rest of parameters if not provided
if [[ -z $scales ]]; then
	scales=$DEFAULT_SCALES
fi
if [[ -z $weights ]]; then
	weights=$DEFAULT_WEIGHTS
fi

# Check docker permissions
if groups $USER | grep &>/dev/null '\bdocker\b'; then SU=""
else SU="sudo"; fi

# Generate images
for content in $contents; do
	for style in $styles; do
		for scale in $scales; do
			for weight in $weights; do
				name=$(outname $content $style $weight $scale)
				if [ ! -f $name ]; then
					echo "Drawing $content with style $style, style weight $weight, style scale $scale"
					nvidia-docker run --rm -v $(pwd):/images albarji/neural-style -backend cudnn -cudnn_autotune -normalize_gradients -init image -content_weight 100 -style_weight $weight -style_scale $scale -content_image $content -style_image $style -output_image $name -save_iter 10000
				else
					echo "Skipping already drawn $name"
				fi
			done
		done
	done
done

