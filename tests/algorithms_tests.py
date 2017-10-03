#
# Tests for the algorithms module
#
from tempfile import TemporaryDirectory
from glob import glob
import filecmp
from neuralstyle.algorithms import styletransfer, ALGORITHMS
from neuralstyle.imagemagick import shape
from neuralstyle.utils import filename

CONTENTS = "/app/entrypoint/tests/contents/"
STYLES = "/app/entrypoint/tests/styles/"


def test_styletransfer_gatys():
    """Style transfer works without error for the Gatys algorithm"""
    tmpdir = TemporaryDirectory()
    styletransfer([CONTENTS + "dockersmall.png"], [STYLES + "cubism.jpg"], tmpdir.name, alg="gatys")
    assert len(glob(tmpdir.name + "/dockersmall*cubism*")) == 1


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
        files = glob(tmpdir.name + "/" + filename(img) + "*cubism*")
        # Check correct number of generated images, and that they are different
        assert len(files) == len(stylescales)
        for f1, f2 in zip(files, files[1:]):
            assert not filecmp.cmp(f1, f2, shallow=False)


def test_styletransfer_sw():
    """Style transfer works for varying style weights"""
    styleweights = [1, 5, 10]
    alg = "gatys"
    img = "docker.png"
    tmpdir = TemporaryDirectory()
    styletransfer([CONTENTS + img], [STYLES + "cubism.jpg"], tmpdir.name, alg=alg, size=100,
                  weights=styleweights)
    files = glob(tmpdir.name + "/" + filename(img) + "*cubism*")
    # Check correct number of generated images, and that they are different
    assert len(files) == len(styleweights)
    for f1, f2 in zip(files, files[1:]):
        assert not filecmp.cmp(f1, f2, shallow=False)
