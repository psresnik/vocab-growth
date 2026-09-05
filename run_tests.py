"""
Run the test suite.

You can run the tests either with pytest:

    pytest

or, if you do not have pytest installed, with this script:

    python run_tests.py

Both do the same thing. This script exists so that you never have to install
anything extra just to check whether your code is correct.

Add -v for the name of every test as it runs.
"""

import argparse
import importlib
import os
import sys
import traceback


def find_test_modules(test_directory: str) -> list:
    """Find every test file in the tests directory.

    Args:
        test_directory: folder to search.

    Returns:
        Sorted list of module names, e.g. ["test_analysis", "test_simulation"].
    """
    names = []
    for filename in sorted(os.listdir(test_directory)):
        if filename.startswith("test_") and filename.endswith(".py"):
            names.append(filename[:-3])
    return names


def find_test_functions(module) -> list:
    """Find every test function inside one module.

    Args:
        module: an imported module object.

    Returns:
        List of (name, function) pairs, in the order they appear.
    """
    functions = []
    for name in dir(module):
        if name.startswith("test_"):
            attribute = getattr(module, name)
            if callable(attribute):
                functions.append((name, attribute))
    return functions


def main(argv=None) -> int:
    """Run all tests and report the result.

    Args:
        argv: argument list, or None to read from the command line.

    Returns:
        0 if every test passed, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print each test name as it runs")
    parser.add_argument("--only", default=None,
                        help="run only tests whose name contains this text")
    arguments = parser.parse_args(argv)

    project_root = os.path.dirname(os.path.abspath(__file__))
    test_directory = os.path.join(project_root, "tests")
    sys.path.insert(0, project_root)
    sys.path.insert(0, test_directory)

    n_passed = 0
    failures = []

    for module_name in find_test_modules(test_directory):
        module = importlib.import_module(module_name)
        for test_name, test_function in find_test_functions(module):
            full_name = "%s.%s" % (module_name, test_name)
            if arguments.only and arguments.only not in full_name:
                continue
            try:
                test_function()
            except Exception:
                failures.append((full_name, traceback.format_exc()))
                print("FAIL  %s" % full_name)
            else:
                n_passed += 1
                if arguments.verbose:
                    print("ok    %s" % full_name)

    print("\n%d passed, %d failed" % (n_passed, len(failures)))

    if failures:
        print("\n" + "=" * 70)
        for full_name, message in failures:
            print("\nFAILED: %s\n%s" % (full_name, message))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
