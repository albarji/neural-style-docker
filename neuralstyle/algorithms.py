# Callers to neural style algorithms
from subprocess import call
from itertools import product
from tempfile import TemporaryDirectory, NamedTemporaryFile
from shutil import copyfile
import logging
from math import ceil
from neuralstyle.utils import filename, fileext
from neuralstyle.imagemagick import (convert, resize, shape, assertshape, choptiles, feather, smush, composite,
                                     extractalpha, mergealpha)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Folders and commands for style-transfer algorithms
ALGORITHMS = {
    "gatys": {
        "folder": "/app/neural-style",
        "command": "th neural_style.lua",
        "defaultpars": [
            "-backend", "cudnn",
            "-cudnn_autotune",
            "-normalize_gradients",
            "-init", "image",
            "-content_weight", "100",
            "-save_iter", "10000",
            "-proto_file", "/app/neural-style/models/VGG_ILSVRC_19_layers_deploy.prototxt",
            "-model_file", "/app/neural-style/models/VGG_ILSVRC_19_layers.caffemodel",
            "-num_iterations", "500"
        ]
    },
    "chen-schmidt": {
        "folder": "/app/style-swap",
        "command": "th style-swap.lua",
        "defaultpars": [
            "--patchSize", "7",
            "--patchStride", "3"
        ]
    },
    "chen-schmidt-inverse": {
        "folder": "/app/style-swap",
        "command": "th style-swap.lua",
        "defaultpars": [
            "--decoder", "models/dec-tconv-sigmoid.t7"
        ]
    }
}


def styletransfer(contents, styles, savefolder, size=None, alg="gatys", weights=None, stylescales=None,
                  maxtilesize=400, tileoverlap=100, algparams=None):
    """General style transfer routine over multiple sets of options"""
    # Check arguments
    if alg not in ALGORITHMS.keys():
        raise ValueError("Unrecognized algorithm %s, must be one of %s" % (alg, str(list(ALGORITHMS.keys()))))

    # Plug default options
    if alg != "gatys":
        if weights is not None:
            LOGGER.warning("Only gatys algorithm accepts style weights. Ignoring style weight parameters")
        weights = [None]
    else:
        if weights is None:
            weights = [5.0]
    if stylescales is None:
        stylescales = [1.0]
    if maxtilesize is None:
        maxtilesize = 400
    if tileoverlap is None:
        tileoverlap = 100
    if algparams is None:
        algparams = []

    # Iterate through all combinations
    for content, style, weight, scale in product(contents, styles, weights, stylescales):
        outfile = outname(savefolder, content, style, alg, scale, weight)
        # If the desired size is smaller than the maximum tile size, use a direct neural style
        if fitsingletile(targetshape(content, size), maxtilesize):
            styletransfer_single(content=content, style=style, outfile=outfile, size=size, alg=alg, weight=weight,
                                 stylescale=scale, algparams=algparams)
        # Else use a tiling strategy
        else:
            neuraltile(content=content, style=style, outfile=outfile, size=size, maxtilesize=maxtilesize,
                       overlap=tileoverlap, alg=alg, weight=weight, stylescale=scale, algparams=algparams)


def styletransfer_single(content, style, outfile, size=None, alg="gatys", weight=5.0, stylescale=1.0, algparams=None):
    """General style transfer routine over a single set of options"""
    workdir = TemporaryDirectory()

    # Cut out alpha channel from content
    rgbfile = workdir.name + "/" + "rgb.png"
    alphafile = workdir.name + "/" + "alpha.png"
    extractalpha(content, rgbfile, alphafile)

    # Transform style to png, as some algorithms don't understand other formats
    stylepng = workdir.name + "/" + "style.png"
    convert(style, stylepng)

    # Call style transfer algorithm
    algfile = workdir.name + "/" + "algoutput.png"
    if alg == "gatys":
        gatys(rgbfile, stylepng, algfile, size, weight, stylescale, algparams)
    elif alg in ["chen-schmidt", "chen-schmidt-inverse"]:
        chenschmidt(alg, rgbfile, stylepng, algfile, size, stylescale, algparams)
    # Enforce correct size
    correctshape(algfile, content, size)

    # Recover alpha channel
    correctshape(alphafile, content, size)
    mergealpha(algfile, alphafile, outfile)


def neuraltile(content, style, outfile, size=None, maxtilesize=400, overlap=100, alg="gatys", weight=5.0,
               stylescale=1.0, algparams=None):
    """Strategy to generate a high resolution image by running style transfer on overlapping image tiles"""
    LOGGER.info("Starting tiling strategy")
    if algparams is None:
        algparams = []
    workdir = TemporaryDirectory()

    # Gather size info from original image
    fullshape = targetshape(content, size)

    # Compute number of tiles required to map all the image
    xtiles, ytiles = tilegeometry(fullshape, maxtilesize, overlap)

    # First scale image to target resolution
    firstpass = workdir.name + "/" + "lowres.png"
    convert(content, firstpass)
    resize(firstpass, fullshape)

    # Chop the styled image into tiles with the specified overlap value.
    lowrestiles = choptiles(firstpass, xtiles=xtiles, ytiles=ytiles, overlap=overlap,
                            outname=workdir.name + "/" + "lowres_tiles")

    # High resolution pass over each tile
    highrestiles = []
    for i, tile in enumerate(lowrestiles):
        name = workdir.name + "/" + "highres_tiles_" + str(i) + ".png"
        styletransfer_single(tile, style, name, size=None, alg=alg, weight=weight, stylescale=stylescale,
                             algparams=algparams)
        highrestiles.append(name)

    # Feather tiles
    featheredtiles = []
    for i, tile in enumerate(highrestiles):
        name = workdir.name + "/" + "feathered_tiles_" + str(i) + ".png"
        feather(tile, name)
        featheredtiles.append(name)

    # Smush the feathered tiles together
    smushedfeathered = workdir.name + "/" + "feathered_smushed.png"
    smush(featheredtiles, xtiles, ytiles, overlap, overlap, smushedfeathered)

    # Smush also the non-feathered tiles
    smushedhighres = workdir.name + "/" + "highres_smushed.png"
    smush(highrestiles, xtiles, ytiles, overlap, overlap, smushedhighres)

    # Combine feathered and un-feathered output images to disguise feathering
    composite([smushedfeathered, smushedhighres], outfile)

    # Adjust back to desired size
    assertshape(outfile, fullshape)


def gatys(content, style, outfile, size, weight, stylescale, algparams):
    """Runs Gatys et al style-transfer algorithm

    References:
        * https://arxiv.org/abs/1508.06576
        * https://github.com/jcjohnson/neural-style
    """
    # Gatys can only process one combination of content, style, weight and scale at a time, so we need to iterate
    tmpout = NamedTemporaryFile(suffix=".png")
    runalgorithm("gatys", [
        "-content_image", content,
        "-style_image", style,
        "-style_weight", weight * 100,  # Because content weight is 100
        "-style_scale", stylescale,
        "-output_image", tmpout.name,
        "-image_size", size if size is not None else shape(content)[0],
        *algparams
    ])
    # Transform to original file format
    convert(tmpout.name, outfile)
    tmpout.close()


def chenschmidt(alg, content, style, outfile, size, stylescale, algparams):
    """Runs Chen and Schmidt fast style-transfer algorithm

    References:
        * https://arxiv.org/pdf/1612.04337.pdf
        * https://github.com/rtqichen/style-swap
    """
    if alg not in ["chen-schmidt", "chen-schmidt-inverse"]:
        raise ValueError("Unnaceptable subalgorithm %s for Chen-Schmidt family")

    # Rescale style as requested
    instyle = NamedTemporaryFile()
    copyfile(style, instyle.name)
    resize(instyle.name, int(stylescale * shape(style)[0]))
    # Run algorithm
    outdir = TemporaryDirectory()
    runalgorithm(alg, [
        "--save", outdir.name,
        "--content", content,
        "--style", instyle.name,
        "--maxContentSize", size if size is not None else shape(content)[0],
        "--maxStyleSize", size if size is not None else shape(content)[0],
        *algparams
    ])
    # Gather output results
    output = outdir.name + "/" + filename(content) + "_stylized" + fileext(content)
    convert(output, outfile)
    instyle.close()


def runalgorithm(alg, params):
    """Run a style transfer algorithm with given parameters"""
    # Move to algorithm folder
    command = "cd " + ALGORITHMS[alg]["folder"] + "; "
    # Algorithm command with default parameters
    command += ALGORITHMS[alg]["command"] + " " + " ".join(ALGORITHMS[alg]["defaultpars"])
    # Add provided parameters, if any
    command += " " + " ".join([str(p) for p in params])
    LOGGER.info("Running command: %s" % command)
    call(command, shell=True)


def outname(savefolder, content, style, alg, scale, weight=None, ext=None):
    """Creates an output filename that reflects the style transfer parameters"""
    return (
        savefolder + "/" +
        filename(content) +
        "_" + filename(style) +
        "_" + alg +
        "_ss" + str(scale) +
        ("_sw" + str(weight) if weight is not None else "") +
        (ext if ext is not None else fileext(content))
    )


def correctshape(result, original, size=None):
    """Corrects the result of style transfer to ensure shape is coherent with original image and desired output size

    If output size is not specified, the result image is corrected to have the same shape as the original.
    """
    assertshape(result, targetshape(original, size))


def tilegeometry(imshape, maxtilesize=400, overlap=50):
    """Given the shape of an image, computes the number of X and Y tiles to cover it"""
    xtiles = ceil(float(imshape[0] - maxtilesize) / float(maxtilesize - overlap) + 1)
    ytiles = ceil(float(imshape[1] - maxtilesize) / float(maxtilesize - overlap) + 1)
    return xtiles, ytiles


def fitsingletile(imshape, maxtilesize):
    """Returns whether a given image shape will fit in a single tile or not"""
    return all([x <= maxtilesize for x in imshape])


def targetshape(content, size=None):
    """Computes the shape the resultant image will have after a reshape of the size given

    If size is None, return original shape.
    """
    contentshape = shape(content)
    if size is None:
        return contentshape
    else:
        return [size, int(size * contentshape[1] / contentshape[0])]
