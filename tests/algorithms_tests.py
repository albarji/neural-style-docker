#
# Tests for the algorithms module
#
from tempfile import TemporaryDirectory
from glob import glob
from neuralstyle.algorithms import styletransfer, neuraltile, ALGORITHMS
from neuralstyle.imagemagick import shape, equalimages
from neuralstyle.utils import filename

CONTENTS = "/app/entrypoint/tests/contents/"
STYLES = "/app/entrypoint/tests/styles/"


def assertalldifferent(pattern, expected=None):
    """Asserts that all images that follow a given glob pattern have different contents

    An expected number of images can also be provided to be checked.
    """
    files = glob(pattern)
    if expected is not None:
        assert len(files) == expected
    for f1, f2 in zip(files, files[1:]):
        assert not equalimages(f1, f2)


def test_styletransfer_gatys():
    """Style transfer works without error for the Gatys algorithm"""
    tmpdir = TemporaryDirectory()
    styletransfer([CONTENTS + "dockersmall.png"], [STYLES + "cubism.jpg"], tmpdir.name, alg="gatys")
    assert len(glob(tmpdir.name + "/dockersmall*cubism*")) == 1


def test_styletransfer_gatys_parameters():
    """Algorithm parameters can be passed to the Gatys method"""
    tmpdir = TemporaryDirectory()
    algparams = ("-num_iterations", "50")
    styletransfer([CONTENTS + "dockersmall.png"], [STYLES + "cubism.jpg"], tmpdir.name, algparams=algparams)
    assert len(glob(tmpdir.name + "/dockersmall*cubism*")) == 1


def test_styletransfer_gatysmultiresolution():
    """Style transfer works without error for the Gatys algorithm with multiresolution"""
    tmpdir = TemporaryDirectory()
    styletransfer([CONTENTS + "docker.png"], [STYLES + "cubism.jpg"], tmpdir.name, alg="gatys-multiresolution",
                  size=600)
    assert len(glob(tmpdir.name + "/docker*cubism*")) == 1


def test_styletransfer_chenschmidt():
    """Style transfer method works without error for the Chend-Schmidt algorithm"""
    tmpdir = TemporaryDirectory()
    styletransfer([CONTENTS + "dockersmall.png"], [STYLES + "cubism.jpg"], tmpdir.name, alg="chen-schmidt")
    assert len(glob(tmpdir.name + "/dockersmall*cubism*")) == 1


def test_styletransfer_chenschmidtinverse():
    """Style transfer method works without error for the Chend-Schmidt Inverse algorithm"""
    tmpdir = TemporaryDirectory()
    styletransfer([CONTENTS + "dockersmall.png"], [STYLES + "cubism.jpg"], tmpdir.name, alg="chen-schmidt-inverse")
    assert len(glob(tmpdir.name + "/dockersmall*cubism*")) == 1


def test_styletransfer_keepsize():
    """Style transfer keeps the original image size if no size paramenter is given"""
    for alg in ALGORITHMS.keys():
        tmpdir = TemporaryDirectory()
        img = CONTENTS + "dockersmall.png"
        styletransfer([img], [STYLES + "cubism.jpg"], tmpdir.name, alg=alg)
        files = glob(tmpdir.name + "/" + filename(img) + "*cubism*")
        print("Expected size", shape(img))
        print("Actual shape", shape(files[0]))
        assert len(files) == 1
        assert shape(files[0]) == shape(img)


def test_styletransfer_size():
    """Style transfer works for varying image sizes, producing correctly scaled images"""
    for alg in ALGORITHMS.keys():
        for size in [50, 100, 200]:
            for img in ["docker.png", "obama.jpg"]:
                originalshape = shape(CONTENTS + img)
                tmpdir = TemporaryDirectory()
                styletransfer([CONTENTS + img], [STYLES + "cubism.jpg"], tmpdir.name, alg=alg, size=size)
                files = glob(tmpdir.name + "/" + filename(img) + "*cubism*")
                resultshape = shape(files[0])
                rescalefactor = size / originalshape[0]
                expectedshape = [size, int(rescalefactor * originalshape[1])]
                print("Expected shape", expectedshape)
                print("Actual shape", resultshape)
                assert len(files) == 1
                assert expectedshape == resultshape


def test_styletransfer_ss():
    """Style transfer works for varying style scales"""
    stylescales = [0.75, 1, 1.25]
    for alg in ALGORITHMS.keys():
        img = "docker.png"
        tmpdir = TemporaryDirectory()
        styletransfer([CONTENTS + img], [STYLES + "cubism.jpg"], tmpdir.name, alg=alg, size=100,
                      stylescales=stylescales)
        assertalldifferent(tmpdir.name + "/" + filename(img) + "*cubism*", len(stylescales))


def test_styletransfer_sw():
    """Style transfer works for varying style weights"""
    styleweights = [1, 5, 10]
    alg = "gatys"
    img = "docker.png"
    tmpdir = TemporaryDirectory()
    styletransfer([CONTENTS + img], [STYLES + "cubism.jpg"], tmpdir.name, alg=alg, size=100,
                  weights=styleweights)
    assertalldifferent(tmpdir.name + "/" + filename(img) + "*cubism*", len(styleweights))


def test_neuraltile():
    """The neural tiling procedure can be run without issues"""
    tmpdir = TemporaryDirectory()
    content = CONTENTS + "avila-walls.jpg"
    outfile = tmpdir.name + "/tiled.png"
    neuraltile(content, STYLES + "cubism.jpg", outfile, alg="chen-schmidt-inverse", overlap=100)
    assert shape(outfile) == shape(content)


def test_formattga():
    """TGA format images can be processed correctly"""
    contents = [CONTENTS + f for f in ["tgasample.tga", "marbles.tga"]]
    tmpdir = TemporaryDirectory()
    styletransfer(contents, [STYLES + "cubism.jpg"], tmpdir.name, alg="chen-schmidt-inverse")
    assert len(glob(tmpdir.name + "/*cubism*")) == 2


def test_formatpsd():
    """PSD format images can be processed correctly"""
    contents = [CONTENTS + f for f in ["oldtelephone.psd"]]
    tmpdir = TemporaryDirectory()
    styletransfer(contents, [STYLES + "cubism.jpg"], tmpdir.name, alg="chen-schmidt-inverse")
    assert len(glob(tmpdir.name + "/*cubism*")) == 1


def test_alpha():
    """Transformation of images with an alpha channel preserve transparency"""
    tmpdir = TemporaryDirectory()
    # Transform image with alpha
    styletransfer([CONTENTS + "dockersmallalpha.png"], [STYLES + "cubism.jpg"], tmpdir.name, alg="chen-schmidt-inverse")
    assert len(glob(tmpdir.name + "/*dockersmallalpha_cubism*")) == 1
    # Transform image without alpha
    styletransfer([CONTENTS + "dockersmall.png"], [STYLES + "cubism.jpg"], tmpdir.name, alg="chen-schmidt-inverse")
    assert len(glob(tmpdir.name + "/*dockersmall_cubism*")) == 1
    # Check correct that generated image are different
    assertalldifferent(tmpdir.name + "/*cubism*")


def test_alpha_tiling():
    """Transformation of images with an alpha channel preserve transparency, even when a tiling strategy is used"""
    tmpdir = TemporaryDirectory()
    # Transform image with alpha
    styletransfer([CONTENTS + "dockersmallalpha.png"], [STYLES + "cubism.jpg"], tmpdir.name, alg="chen-schmidt-inverse",
                  size=500)
    assert len(glob(tmpdir.name + "/*dockersmallalpha_cubism*")) == 1
    # Transform image without alpha
    styletransfer([CONTENTS + "dockersmall.png"], [STYLES + "cubism.jpg"], tmpdir.name, alg="chen-schmidt-inverse",
                  size=500)
    assert len(glob(tmpdir.name + "/*dockersmall_cubism*")) == 1
    # Check correct that generated image are different
    assertalldifferent(tmpdir.name + "/*cubism*")
