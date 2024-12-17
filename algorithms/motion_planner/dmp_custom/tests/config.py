# -----------------------------------------------------------------------------
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of the LogicLfD project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yan Zhang <yan.zhang@idiap.ch>
# -----------------------------------------------------------------------------

import sys
from os.path import join, abspath, dirname

def absjoin(*args):
    return abspath(join(*args))

import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = absjoin(dirname(__file__), '..')

sys.path.append(PROJECT_DIR)