# encoding: utf-8
"""
@time: 2019/12/13 3:23
@desc: an easy to use logger wrapper
"""

import logging
from logging import handlers
import sys
import os

logging.getLogger('pillow').setLevel(logging.WARNING)
logging.getLogger('pil').setLevel(logging.WARNING)


def log_printer(logger, message, level='info'):
    if logger is None:
        print(message)
    elif level == 'debug':
        logger.debug(message)
    elif level == 'warning':
        logger.warning(message)
    else:
        logger.info(message)


class Logger(object):
    def __init__(self, log_file='log.txt', debug=False):
        logger = logging.getLogger('')
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        self.streamhandler = logging.StreamHandler(sys.stdout)
        self.streamhandler.setFormatter(format)
        logger.addHandler(self.streamhandler)

        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.filehandler = handlers.RotatingFileHandler(log_file, maxBytes=(1048576*5), backupCount=7)
        self.filehandler.setFormatter(format)
        logger.addHandler(self.filehandler)
        self.logger = logger

    def remove_handle(self):
        self.logger.removeHandler(self.streamhandler)
        self.logger.removeHandler(self.filehandler)

    def log(self, message='', level='info'):
        if level == 'debug':
            self.logger.debug(message)
        elif level == 'warning':
            self.logger.warning(message)
        else:
            self.logger.info(message)

    def info(self, message):
        return self.log(message, level='info')

    def debug(self, message):
        return self.log(message, level='debug')

    def warning(self, message):
        return self.log(message, level='warning')

    @staticmethod
    def printer(logger, message):
        if logger is None:
            print(message)
        else:
            logger.info(message)

    

    
