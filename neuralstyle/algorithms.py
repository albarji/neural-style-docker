# Callers to neural style algorithms
from subprocess import call
from itertools import product
from tempfile import TemporaryDirectory, NamedTemporaryFile
from shutil import copyfile
import logging
from neuralstyle.utils import filename, fileext
from neuralstyle.imagemagick import convert, resize, shape, assertshape

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
            "-model_file", "/app/neural-style/models/VGG_ILSVRC_19_layers.caffemodel"
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


def styletransfer(contents, styles, savefolder, size=None, alg="gatys", weights=None, stylescales=None, *args):
    """General style transfer routine"""
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
            weights = [10.0]
    if stylescales is None:
        stylescales = [1.0]

    # Iterate through all combinations
    for content, style, weight, scale in product(contents, styles, weights, stylescales):
        outfile = outname(savefolder, content, style, alg, scale, weight)
        # Call style transfer algorithm
        if alg == "gatys":
            gatys(content, style, outfile, size, weight, scale, *args)
        elif alg in ["chen-schmidt", "chen-schmidt-inverse"]:
            chenschmidt(alg, content, style, outfile, size, scale, *args)
        # Enforce correct size
        correctshape(outfile, content, size)


def gatys(content, style, outfile, size, weight, stylescale, *args):
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
        *args
    ])
    # Transform to original file format
    convert(tmpout.name, outfile)
    tmpout.close()


def chenschmidt(alg, content, style, outfile, size, stylescale, *args):
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
        *args
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
    originalshape = shape(original)
    if size is not None:
        targetshape = [size, int(size * originalshape[1] / originalshape[0])]
    else:
        targetshape = originalshape
    assertshape(result, targetshape)
