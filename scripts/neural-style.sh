#!/bin/bash

# Launches neural-style with the appropriate model location parameters

th /neural-style/neural_style.lua -proto_file /neural-style/models/VGG_ILSVRC_19_layers_deploy.prototxt -model_file /neural-style/models/VGG_ILSVRC_19_layers.caffemodel "$@"

