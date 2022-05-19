# -*- coding: utf-8 -*-

import glob

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 0
_version_micro = 0
_version_extra = ''

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 0 - WIP",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: TODO",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "Carreno Cytonemes: cytonemes metrics from volumetric data"
# Long description will go up on the pypi page
long_description = """
Carreno Cytonemes
========
Carreno Cytonemes is a small library for ... TODO
License
=======
``carreno`` is not licensed yet ... TODO
"""

NAME = "carreno"
MAINTAINER = "Philippe Lemieux"
MAINTAINER_EMAIL = "TODO"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://gitlab.linum.info.uqam.ca/philippe/carreno-cytonemes"
DOWNLOAD_URL = ""
LICENSE = "None"
AUTHOR = "LINUM developers"
AUTHOR_EMAIL = ""
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
SCRIPTS = glob.glob("scripts/*.py")

PREVIOUS_MAINTAINERS=[""]