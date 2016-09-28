# -*- coding: utf-8 -*-
"""
Given a set of content images and a set of style images, generates
all possible combinations of content and styles. For each possible
combination several variants are tested, altering content to style weight and
scaling of style.

@author: Álvaro Barbero Jiménez
"""
import argparse
from subprocess import call
import os
from ast import literal_eval

def blendimages(content, style, outname, styleweight=1500, stylescale=1):
    """Blends content and style images with some given parameters
    
    The blended image is saved into the provided outfolder, with a name
    that encodes the parameters used for generation.
    """    
    print("Generating blend for", content, "+", style, "with styleweight",
          str(styleweight), ", stylescale", str(stylescale))
    call(["th", "neural_style.lua", 
          "-backend", "cudnn",
          "-cudnn_autotune",
          "-normalize_gradients",
          "-init", "image",
          "-style_image", style,
          "-content_image", content,
          "-output_image", outname,
          "-content_weight", "100", 
          "-style_weight", str(styleweight),
          "-style_scale", str(stylescale),
          "-save_iter", "10000" # Don't save intermediate results
          ])
          
def getoutname(content, style, outfolder, styleweight, stylescale):
    """Generates a name for an output image, depending on its parameters"""
    stylename = os.path.splitext(os.path.basename(style))[0]
    contentname = os.path.splitext(os.path.basename(content))[0]
    outname = (
        outfolder + "/" 
        + contentname 
        + "_" + stylename 
        + "_sw" + str(styleweight)
        + "_ss" + str(stylescale)
        + ".png"
    )
    return outname

def generatevariants(content, style, outfolder,
                     styleweights=[375, 750, 1500, 2000, 5000, 10000],
                     stylescales=[1]):
    """Generate several variantes of the same content+style blend"""
    print("Generating variants for", content, "+", style)
    for styleweight in styleweights:
        for stylescale in stylescales:
            outname = getoutname(content, style, outfolder, styleweight, stylescale)
            # Generate only if doesn't exist already
            if not os.path.isfile(outname):
                blendimages(content, style, outname, styleweight, stylescale)

def generatelists(contents, styles, outfolder):
    """Generates all possible blends of given content and style images"""
    for content in contents:
        for style in styles:
            generatevariants(content, style, outfolder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate variants of neural-style blends')
    parser.add_argument('--contents', nargs='+', type=str, help='list of content images')
    parser.add_argument('--styles', nargs='+', type=str, help='list of style images')
    parser.add_argument('--outfolder', type=str, help='output folder')
    args = parser.parse_args()
    
    generatelists(args.contents, args.styles, args.outfolder)
    
