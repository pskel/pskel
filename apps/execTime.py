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
ys = [18.4904, 16.1304, 15.3102, 15.1206, 14.51, 14.57, 14.3599, 14.7001]
plt.plot(xs,ys,marker=markers[2],linewidth=2,label='Jacobi')
#GoL
ys = [25.6103, 19.7304, 17.6504, 16.8106, 16.0107, 15.7401, 15.4702, 15.3807]
plt.plot(xs,ys,marker=markers[1],linewidth=2,label='GoL')
#Fur
ys = [69.3003, 38.5501, 28.0406, 23.1106, 24.3606, 24.2603, 22.5001, 21.4399]
plt.plot(xs,ys,marker=markers[0],linewidth=2,label='Fur')

plt.xticks([(w+1)*2 for w in range(8)])
# plt.yticks([0, 1])
plt.xlim(1.8,16.2)
plt.ylim(0, 80)

plt.suptitle('Tempo de Execução', fontsize=27)
plt.xlabel('Número de Clusters', fontsize=22)
plt.ylabel('Tempo de Execução (s)', fontsize=22)
   #plt.xlim(-0.15, 2.9)
plt.grid(which='major')
#plt.tick_params(labelsize=12)
#plt.tick_params(pad=10)

plt.tick_params(labelsize=20, pad=2, width=2)
plt.legend(loc='best', fontsize = 'medium', ncol=1)

plt.savefig('mppa-execTime.pdf')
