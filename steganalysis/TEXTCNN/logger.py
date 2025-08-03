# -*- coding: utf-8 -*-

import logging
import os


class Logger(object):
    def __init__(self, log_file):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Set up the logger
        self.logger = logging.getLogger(log_file)
        self.logger.setLevel(logging.INFO)
        
        # Check if the logger already has handlers
        if not self.logger.handlers:
            # Create a file handler in append mode
            fh = logging.FileHandler(log_file, mode='a')
            fh.setLevel(logging.INFO)
            
            # Create a logging format
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            fh.setFormatter(formatter)
            
            # Add the handlers to the logger
            self.logger.addHandler(fh)

    def info(self, message):
        self.logger.info(message)
