# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 11:11:58 2022

@author: py221
"""
from src.Problems.ToyProblem1 import centralised
import matplotlib.pyplot as plt
import pyomo.environ as pyo
import numpy as np

x3 =np.arange(-1.0, 6.0, 0.01)
fun = []
for i in x3:
    data_centr = {None: {'x_init':  {1:1,2:1}, 'x3':  {None: i}}}
    res = centralised(data_centr)
    fun.append(pyo.value(res.obj))



plt.plot(x3,fun)

plt.xlabel('shared variable, $x_3$')
plt.ticklabel_format(style='sci', axis='x')
plt.ylabel('Global optimum')
plt. savefig('./Figures/0908/3.svg', format = "svg")