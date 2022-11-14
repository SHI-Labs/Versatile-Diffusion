import torch
import numpy as np
import lpips

from .. import nputils
from ..log_service import print_log

from .eva_base import base_evaluator, register

@register('null')
class null_evaluator(base_evaluator):
    def __init__(self, **dummy):
        super().__init__()

    def add_batch(self, 
                  **dummy):
        pass

    def compute(self):
        return None

    def one_line_summary(self):
        print_log('Evaluator null')

    def clear_data(self):
        pass