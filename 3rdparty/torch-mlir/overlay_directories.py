#!/bin/python3

import argparse
import errno
import os
import sys


def _check_python_version():
    if sys.version_info[0] < 3:
        raise RuntimeError(
            "Must be invoked with a python 3 interpreter but was %s" % sys.executable
        )


def _check_dir_exists(path):
    if not os.path.isdir(path):
        raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), path)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""
    Overlays two directories into a target directory using symlinks.

    Tries to minimize the number of symlinks created (that is, does not symlink
    every single file). Symlinks every file in the overlay directory. Only
    symlinks individual files in the source directory if their parent directory
    is also contained in the overlay directory tree.
    """
    )
    parser.add_argument(
        "--overlay",
        required=True,
        help="Directory to overlay on top of the source directory.",
    )
    parser.add_argument(
        "target",
        help="Directory in which to place the fused symlink directories.",
    )

    args = parser.parse_args()

    _check_dir_exists(args.target)
    _check_dir_exists(args.overlay)

    return args


def _symlink_abs(from_path, to_path):
    if not os.path.exists(to_path):
        os.symlink(os.path.abspath(from_path), os.path.abspath(to_path))


def main(args):
    for root, _, files in os.walk(args.overlay):
        rel_root = os.path.relpath(root, start=args.overlay)
        if rel_root != "." and not os.path.exists(os.path.join(args.target, rel_root)):
            os.mkdir(os.path.join(args.target, rel_root))

        for file in files:
            relpath = os.path.join(rel_root, file)
            _symlink_abs(
                os.path.join(args.overlay, relpath), os.path.join(
                    args.target, relpath)
            )


if __name__ == "__main__":
    _check_python_version()
    main(parse_arguments())
