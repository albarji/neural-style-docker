#
# Tests for the imagemagick module
#
from tempfile import TemporaryDirectory
from shutil import copyfile
from neuralstyle.imagemagick import convert, shape, resize

CONTENTS = "/app/entrypoint/tests/contents/"


def test_shape():
    """The shape of an image can be correctly recovered"""
    tests = [  # Inputs, expected outputs
        (CONTENTS + "docker.png", [508, 443]),
        (CONTENTS + "goldengate.jpg", [1920, 1080])
    ]

    for imfile, expected in tests:
        result = shape(imfile)
        print("Input", imfile)
        print("Expected", expected)
        print("Output", result)
        assert result == expected


def test_resize_keepproportions():
    """Resizing an image without changing proportions works correctly"""
    tmpdir = TemporaryDirectory()
    fname = tmpdir.name + "/docker.png"
    copyfile(CONTENTS + "docker.png", fname)
    resize(fname, 1016)
    assert shape(fname) == [1016, 886]


def test_resize_changeproportions():
    """Resizing an image changing proportions works correctly"""
    tmpdir = TemporaryDirectory()
    fname = tmpdir.name + "/docker.png"
    copyfile(CONTENTS + "docker.png", fname)
    resize(fname, [700, 300])
    assert shape(fname) == [700, 300]
