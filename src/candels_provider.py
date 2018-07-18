import os

import numpy as np
import pandas as pd
import tensorflow as tf
from astropy.io import fits

class DataProvider:

    def __init__(self, data_dir='../data', split=0.8, batch_size=25):
        self.data_dir = data_dir
        self.split = split
        self.batch_size = batch_size