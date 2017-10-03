#
# Tests for the utils module
#

from neuralstyle.utils import sublist


def test_sublist():
    """The sublist method to gather arguments works as expected"""
    tests = [  # Inputs, expected outputs
        (["bla", "ble", "--someoption"], ["bla", "ble"]),
        (["bla", "--someoption"], ["bla"]),
        (["--someoption"], []),
        (["bla"], ["bla"]),
        ([], [])
    ]

    for lst, expected in tests:
        result = sublist(lst, stopper="--")
        print("Input", lst)
        print("Expected", expected)
        print("Output", result)
        assert result == expected
