# -*- coding: utf-8 -*-
"""
Controller implementing the API endpoints of the neural style server.

@author: Álvaro Barbero Jiménez
"""
from skimage import io
import tempfile
from subprocess import call
from flask import send_file
import os

def blend_post(content, style) -> str:
    # Load input images
    imgcont, imgstyle = (io.imread(i) for i in (content, style))
    
    # Save images to local file    
    fcont = tempfile.NamedTemporaryFile(delete=False)
    fstyle = tempfile.NamedTemporaryFile(delete=False)
    io.imsave(fcont, imgcont)
    io.imsave(fstyle, imgstyle)
    
    # Run torch neural style program
    call(["th", "neural_style.lua", 
          "-backend", "cudnn",
          "-cudnn_autotune",
          "-style_image", fstyle.name,
          "-content_image", fcont.name,
          "-output_image", "out.png",
          #"-image_size", "64" #TODO
          ])

    # Erase temp files
    os.remove(fcont.name)
    os.remove(fstyle.name)
    
    # Return output image
    return send_file("out.png", mimetype='image/png')
    