#!/usr/bin/env python3

# pylint: disable=missing-module-docstring

import os
import platform


MAJOR, MINOR, *_ = platform.python_version_tuple()
CIBW_BUILD = f'CIBW_BUILD=*{platform.python_implementation().lower()[0]}p{MAJOR}{MINOR}-*'

print(CIBW_BUILD)
with open(os.getenv('GITHUB_ENV'), mode='a', encoding='utf-8') as file:
    print(CIBW_BUILD, file=file)
