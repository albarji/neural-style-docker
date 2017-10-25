#
# Tests for the imagemagick module
#
from tempfile import TemporaryDirectory
from shutil import copyfile
from glob import glob
from neuralstyle.imagemagick import shape, resize, choptiles, feather, extractalpha, mergealpha, equalimages, convert
from neuralstyle.utils import filename

CONTENTS = "/app/entrypoint/tests/contents/"


def test_equalimages():
    """Image comparison works"""
    tmpdir = TemporaryDirectory()
    # Compare different images
    assert equalimages(CONTENTS + "docker.png", CONTENTS + "goldengate.jpg") is False
    # Compare equal images
    copied = tmpdir.name + "/docker.png"
    copyfile(CONTENTS + "docker.png", copied)
    assert equalimages(CONTENTS + "docker.png", copied) is True


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


def test_convert_nolayers():
    """Convert a single image with no layers works as expected"""
    for content in [CONTENTS + f for f in ["docker.png", "goldengate.jpg"]]:
        for ext in [".png", ".jpg", ".psd", ".tga"]:
            tmpdir = TemporaryDirectory()
            outname = tmpdir.name + "/" + "output" + ext
            convert(content, outname)
            assert len(glob(tmpdir.name + "/" + filename(outname) + ext)) == 1
            assert shape(outname) == shape(content)


def test_convert_layers():
    """Convert a single image with layers works as expected"""
    for content in [CONTENTS + f for f in ["oldtelephone.psd"]]:
        for ext in [".png", ".jpg", ".psd", ".tga"]:
            tmpdir = TemporaryDirectory()
            outname = tmpdir.name + "/" + "output" + ext
            convert(content, outname)
            assert len(glob(tmpdir.name + "/" + filename(outname) + ext)) == 1
            assert shape(outname) == shape(content)


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


def test_choptiles():
    """Chopping an image into tiles works as expected"""
    tmpdir = TemporaryDirectory()
    content = CONTENTS + "/goldengate.jpg"
    tiles = choptiles(content, xtiles=2, ytiles=3, overlap=50, outname=tmpdir.name + "/tiles")
    print("Generated tiles", tiles)
    assert len(tiles) == 6
    for i, tile in enumerate(tiles):
        assert int(filename(tile).split("_")[-1]) == i


def test_feather():
    """Feathering an image produces noticeable changes"""
    tmpdir = TemporaryDirectory()
    content = CONTENTS + "/clock.jpg"
    outfile = tmpdir.name + "/feathered.jpg"
    feather(content, outfile)
    assert not equalimages(content, outfile)


def test_extractalpha():
    """Extracting the alpha channel from an image works as expected"""
    tmpdir = TemporaryDirectory()
    img = CONTENTS + "/alphasample.png"
    alphafile = tmpdir.name + "/alpha.png"
    rgbfile = tmpdir.name + "/rgb.png"
    extractalpha(img, rgbfile, alphafile)
    assert shape(rgbfile) == shape(img)
    assert shape(alphafile) == shape(img)
    assert not equalimages(img, rgbfile)
    assert not equalimages(img, alphafile)
    assert not equalimages(rgbfile, alphafile)


def test_mergealpha():
    """Extracting the alpha channel from an image, then merging it back, produces the same image"""
    contents = [CONTENTS + imname for imname in ["alphasample.png", "docker.png"]]
    for content in contents:
        tmpdir = TemporaryDirectory()
        alphafile = tmpdir.name + "/alpha.png"
        rgbfile = tmpdir.name + "/rgb.png"
        extractalpha(content, rgbfile, alphafile)
        recfile = tmpdir.name + "/reconstructed.png"
        mergealpha(rgbfile, alphafile, recfile)
        assert equalimages(content, recfile)
