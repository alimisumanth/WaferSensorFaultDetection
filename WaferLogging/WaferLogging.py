# -*- coding: utf-8 -*-
"""
=============================================================================
Created on: 18-02-2022 08:45 PM
Created by: ASK
=============================================================================

Project Name: WaferSensorFaultDetection

File Name: WaferLogging.py

Description:

Version:

Revision:

=============================================================================
"""

import yaml
import logging
import logging.config
from Utils import Utils


class WaferLogging:

    def __init__(self):
        self.utils = Utils.utils()
        with open('logging.yaml', 'r') as f:
            self.log_cfg = yaml.safe_load(f.read())

    def getLogger(self, name):
        self.utils.dirCheck('logs')
        self.log_cfg['handlers']['info_file_handler']['filename'] = 'logs/' + name + '.log'
        logging.config.dictConfig(self.log_cfg)
        logger = logging.getLogger('Wafer')
        return logger
