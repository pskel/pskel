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
# import pandas as pd

mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['font.size'] = 19
#mpl.rcParams['font.weight'] = 'bold'

markers = ['o','D','s','x']
xs = [2,4,6,8,10,12,14,16]
#Jacobi
ys = [85.425648, 78.877656, 79.459938, 82.709682, 84.0129, 87.7114, 90.46736, 97.902666]
plt.plot(xs,ys,marker=markers[2],linewidth=2,label='Jacobi')
#GoL
ys = [121.648925, 99.243912, 94.42964, 95.484208, 96.384414, 97.746021, 99.163982, 103.819725]
plt.plot(xs,ys,marker=markers[1],linewidth=2,label='GoL')
#Fur
ys = [334.027446, 204.701031, 160.111826, 140.512448, 153.958992, 161.088392, 155.025689, 156.51127]
plt.plot(xs,ys,marker=markers[0],linewidth=2,label='Fur')

plt.xticks([(w+1)*2 for w in range(8)])
# plt.yticks([0, 1])
plt.xlim(1.8,16.2)
plt.ylim(0,350)

plt.suptitle('Consumo de Energia', fontsize=27)
plt.xlabel('Número de Clusters', fontsize=22)
plt.ylabel('Energia (J)', fontsize=22)
#plt.xlim(-0.15, 2.9)
plt.grid(which='major')
#plt.tick_params(labelsize=12)
#plt.tick_params(pad=10)

plt.tick_params(labelsize=20, pad=2, width=2)
plt.legend(loc='best', fontsize = 'medium', ncol=1)

plt.savefig('mppa-energy.pdf')
