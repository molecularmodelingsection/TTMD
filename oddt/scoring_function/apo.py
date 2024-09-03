import os
import importlib
import MDAnalysis as mda
import oddt
from oddt import fingerprints
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics



class score:
    def __init__(self, vars):
        self.__dict__ = vars


    def reference(self):
        update = {
                'ref': None
                }

        return update



    def score(self, topology, trajectory, temperature):
        u = mda.Universe(topology, trajectory)

        outscore = []

        for ts in u.trajectory:
            outscore.append(-1)

        return outscore
