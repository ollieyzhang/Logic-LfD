# -----------------------------------------------------------------------------
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of the LogicLfD project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yan Zhang <yan.zhang@idiap.ch>
# -----------------------------------------------------------------------------

import os
import sys
from os.path import join, abspath, dirname

def absjoin(*args):
    return abspath(join(*args))

import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = absjoin(dirname(__file__), '..')
ALGO_PATH = absjoin(PROJECT_DIR, 'algorithms')
LGP_PATH = absjoin(ALGO_PATH, 'lgp')
EXP_PATH = absjoin(PROJECT_DIR, 'experiments')
STATIC_PATH = absjoin(PROJECT_DIR, 'statistics')
TEST_PATH = absjoin(PROJECT_DIR, 'tests')
TRAINED_MODEL_PATH = absjoin(PROJECT_DIR, 'statistics/trained_models')
SCENE_CONFIG_PATH = absjoin('..', 'pybullet_planning', 'pipelines')

sys.path.append(PROJECT_DIR)
sys.path.append(absjoin(PROJECT_DIR, 'env'))
sys.path.append(absjoin(PROJECT_DIR, 'stations/panda_station'))
sys.path.append(absjoin(PROJECT_DIR, 'algorithms/pddlstream'))
sys.path.append(absjoin(PROJECT_DIR, 'algorithms/lgp'))
sys.path.append(absjoin(PROJECT_DIR, 'env-thirdparty/pandapybullet'))
sys.path.append(absjoin(PROJECT_DIR, 'env-thirdparty/pybullet_planning'))
sys.path.append(absjoin(PROJECT_DIR, 'algorithms/motion_planner/pydmps/pydmps'))

# =====================================
MODEL_PATH = absjoin(dirname(__file__), 'models')
PANDA_URDF = absjoin(MODEL_PATH, 'franka_panda/panda.urdf')
PANDA_ARM_URDF = absjoin(MODEL_PATH, 'franka_description/robots/panda_arm.urdf')
PANDA_HAND_URDF = absjoin(MODEL_PATH, 'franka_description/robots/hand.urdf')
PANDA_ARM_HAND_URDF = absjoin(MODEL_PATH, 'franka_description/robots/panda_arm_hand.urdf')
DRAKE_IIWA_URDF = absjoin(MODEL_PATH, 'iiwa_description/urdf/iiwa14_polytope_collision.urdf')

BLOCK_URDF = absjoin(MODEL_PATH, 'block')
OBJECT_URDF = absjoin(MODEL_PATH, 'objects')
SMALL_BLOCK_URDF = absjoin(MODEL_PATH, 'objects/block_for_pick_and_place.urdf')
MIDLLE_BLOCK_URDF = absjoin(MODEL_PATH, 'objects/block_for_pick_and_place_mid_size.urdf')
SINK_URDF = absjoin(MODEL_PATH, 'objects/sink.urdf')
STOVE_URDF = absjoin(MODEL_PATH, 'objects/stove.urdf')
