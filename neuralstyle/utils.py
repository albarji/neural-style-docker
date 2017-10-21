# General utils
from os.path import basename, splitext


def sublist(lst, stopper):
    """Reads a list of strings until a stopper value is found at the beginning of a string"""
    gathered = []
    for item in lst:
        if item.startswith(stopper):
            break
        gathered.append(item)
    return gathered


def filename(path):
    """Given a full path to a file, returns just its name, without path or extension"""
    return splitext(basename(path))[0]


def fileext(path):
    """Given a full path to a file, returns just its extension"""
    return splitext(basename(path))[1]
