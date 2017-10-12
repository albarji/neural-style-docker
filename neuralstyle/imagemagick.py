# Convenience functions to perform Image Magicks
from subprocess import call, check_output


def convert(origin, dest):
    """Transforms the format of an image in a file, by creating a new file with the new format"""
    command = "convert " + origin + " " + dest
    call(command, shell=True)


def shape(imfile):
    """Returns the shape of an image file"""
    textout = check_output("convert " + imfile + ' -format "%w %h" info:', shell=True).decode("utf-8")
    return [int(x) for x in textout.split(" ")]


def resize(imfile, newsize):
    """Resize an image file to a new size.

    If a single value for newsize is provided, the image is rescaled to that size while keeping proportion.
    If an tuple/list with two values are given, the proportions of the image are changed to meet them.
    """
    if isinstance(newsize, int):
        command = "convert " + imfile + " -resize " + str(newsize) + " " + imfile
    else:
        command = "convert " + imfile + " -resize " + str(newsize[0]) + "x" + str(newsize[1]) + "! " + imfile
    call(command, shell=True)


def assertshape(imfile, shp):
    """Checks the shape of an image file, and if not equal the given one, reshapes it"""
    if shape(imfile) != shp:
        resize(imfile, shp)
