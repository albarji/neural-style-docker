# Convenience functions to perform Image Magicks
from subprocess import run, PIPE
from glob import glob
from tempfile import TemporaryDirectory
from neuralstyle.utils import filename


def convert(origin, dest):
    """Transforms the format of an image in a file, by creating a new file with the new format

    If the input file has several layers, they are flattened.
    """
    command = "convert " + origin + " -flatten " + dest
    run(command, shell=True)


def shape(imfile):
    """Returns the shape of an image file

    If the input file has several layers, it is flattened before computing the shape.
    """
    tmpdir = TemporaryDirectory()
    tmpname = tmpdir.name + "/" + "image.png"
    convert(imfile, tmpname)
    result = run("convert " + tmpname + ' -format "%w %h" info:', shell=True, check=True, stdout=PIPE)
    return [int(x) for x in result.stdout.decode("utf-8").split(" ")]


def resize(imfile, newsize):
    """Resize an image file to a new size.

    If a single value for newsize is provided, the image is rescaled to that size while keeping proportion.
    If an tuple/list with two values are given, the proportions of the image are changed to meet them.
    """
    if isinstance(newsize, int):
        command = "convert " + imfile + " -resize " + str(newsize) + " " + imfile
    else:
        command = "convert " + imfile + " -resize " + str(newsize[0]) + "x" + str(newsize[1]) + "! " + imfile
    run(command, shell=True)


def assertshape(imfile, shp):
    """Checks the shape of an image file, and if not equal the given one, reshapes it"""
    if shape(imfile) != shp:
        resize(imfile, shp)


def choptiles(imfile, xtiles, ytiles, overlap, outname):
    """Chops an image file into a geometry of overlapping tiles. Returns ordered list of generated tiles image files"""
    command = 'convert %s -crop %dx%d+%d+%d@ +repage +adjoin %s_%%d.png' % (
                imfile, xtiles, ytiles, overlap, overlap, outname
    )
    run(command, shell=True, check=True)
    return sorted(glob(outname + "_*.png"), key=lambda x: int(filename(x).split("_")[-1]))


def feather(imfile, outname):
    """Produces a feathered version of an image. Note the output format must allow for an alpha channel"""
    command = 'convert %s -alpha set -virtual-pixel transparent -channel A -morphology Distance Euclidean:1,50\! ' \
              '+channel %s' % (imfile, outname)
    run(command, shell=True, check=True)


def smush(tiles, xtiles, ytiles, smushw, smushh, outname):
    """Smush previously tiled images together"""
    if len(tiles) != xtiles * ytiles:
        raise ValueError("Geometry (%d,%d) is incompatible with given number of tiles (%d)"
                         % (xtiles, ytiles, len(tiles)))
    command = "convert -background transparent "
    i = 0
    for _ in range(ytiles):
        rowcmd = ""
        for __ in range(xtiles):
            rowcmd += " " + tiles[i]
            i += 1
        rowcmd += " +smush -%d -background transparent" % smushw
        command += " '(' %s ')'" % rowcmd
    command += " -background none -background transparent -smush -%s %s" % (smushh, outname)
    run(command, shell=True, check=True)


def composite(imfiles, outname):
    """Blends several image files together"""
    command = "composite"
    for imfile in imfiles:
        command += " " + imfile
    command += " " + outname
    run(command, shell=True, check=True)


def extractalpha(imfile, rgbfile, alphafile):
    """Decomposes an image file into the RGB channels and the alpha channel, saving both as separate image files"""
    # Alpha channel extraction
    command = "convert -alpha extract %s %s" % (imfile, alphafile)
    run(command, shell=True, check=True)
    # RGB channels extraction
    command = "convert -alpha off %s %s" % (imfile, rgbfile)
    run(command, shell=True, check=True)


def mergealpha(rgbfile, alphafile, resfile):
    """Applies an alpha channel image to an RGB image"""
    if shape(rgbfile) != shape(alphafile):
        raise ValueError("Cant merge RGB and alpha images of differing sizes: %s vs %s" %
                         (str(shape(rgbfile)), str(shape(alphafile))))
    command = "convert %s %s -compose CopyOpacity -composite %s" % (rgbfile, alphafile, resfile)
    run(command, shell=True, check=True)


def equalimages(imfile1, imfile2):
    """Returns True if two image files have equal content, False if not"""
    # If sizes differ, the images are not equal
    if shape(imfile1) != shape(imfile2):
        return False
    # Run imagemagick comparison commmand
    # This command returns with 0 if images are equal, 1 if they are not, 2 in case of error
    command = "compare -metric rmse %s %s null:" % (imfile1, imfile2)
    result = run(command, shell=True)
    if result.returncode == 2:
        raise IOError("Error while calling imagemagick compare method")
    return result.returncode == 0
