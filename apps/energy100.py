#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
if sys.version_info[0] < 3:
   from StringIO import StringIO
else:
   from io import StringIO

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd

mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['font.size'] = 19
#mpl.rcParams['font.weight'] = 'bold'

markers = ['o','D','s','x']
xs = [2,4,6,8,10,12,14,16]
#Jacobi
ys = [170.057392, 159.51608, 158.86605, 166.5261, 170.45336, 177.531676, 185.7713, 197.62524]
plt.plot(xs,ys,marker=markers[2],linewidth=2,label='Jacobi')
#GoL
ys = [236.8245, 196.824906, 187.091874, 188.091325, 189.387796, 201.652684, 200.5392, 210.64826]
plt.plot(xs,ys,marker=markers[1],linewidth=2,label='GoL')
#Fur
ys = [666.4132, 406.377271, 317.137397, 281.21183, 305.401604, 318.773052, 310.473192, 310.459355]
plt.plot(xs,ys,marker=markers[0],linewidth=2,label='Fur')

plt.xticks([(w+1)*2 for w in range(8)])
# plt.yticks([0, 1])
plt.xlim(1.8,16.2)
plt.ylim(0,667)

plt.xlabel('Number of clusters')
plt.ylabel('Execution Time')
   #plt.xlim(-0.15, 2.9)
plt.grid(which='major')
#plt.tick_params(labelsize=12)
#plt.tick_params(pad=10)

plt.tick_params(labelsize=19, pad=5, width=2)
plt.legend(loc='best', fontsize = 'small', ncol=1)

plt.savefig('mppa-energy.pdf')
