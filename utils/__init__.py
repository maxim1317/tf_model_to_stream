# -*- coding: utf-8 -*-
"""
File: __init__.py
Description: Different tools for object detection and face recognition
"""

from utils import cognitive_face
from utils import consts
from utils import facerec
from utils import misc

from utils.consts import (FRAME_HEIGHT, FRAME_WIDTH, MODEL_NAME, PATH_TO_CKPT,
                          back_path, back_war, cooling_time, cv_path,
                          drawable_classes, front_headers, front_path,
                          front_url, groupID, label_color, model_parts,
                          pic_dir, pics_needed, reg_path, subscription_key,
                          threshold, uri_base, wait_time,)
from utils.facerec import (checkIfTrained, sendDataRequest, sendToDetection,
                           sendToFront, sendToIdentification,)
from utils.misc import (download_weights, ensure_dir, join, readsize, spc,
                        testDevice,)

__all__ = ['FRAME_HEIGHT', 'FRAME_WIDTH', 'MODEL_NAME', 'PATH_TO_CKPT',
           'back_path', 'back_war', 'checkIfTrained', 'cognitive_face',
           'consts', 'cooling_time', 'cv_path', 'download_weights',
           'drawable_classes', 'ensure_dir', 'facerec', 'front_headers',
           'front_path', 'front_url', 'groupID', 'join', 'label_color', 'misc',
           'model_parts', 'pic_dir', 'pics_needed', 'readsize', 'reg_path',
           'sendDataRequest', 'sendToDetection', 'sendToFront',
           'sendToIdentification', 'spc', 'subscription_key', 'testDevice',
           'threshold', 'uri_base', 'wait_time']
