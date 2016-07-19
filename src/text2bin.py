# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 17:19:22 2016

@author: whr94621
"""

import numpy as np
import sys
f_in  = sys.argv[1]
f_out = sys.argv[2]


def List2NdArray(l):
    L = len(l)
    vec = np.empty(shape=L, dtype=np.float64)
    for i, e in enumerate(l):
        vec[i] = np.float64(e)
    return vec


with open(f_in, 'r') as f, open(f_out, 'wb') as g:
    f.readline()
    for line in f:
        line = line.strip().split()[1:]
        vec = List2NdArray(line)
        g.write(vec.tostring())