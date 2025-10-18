# coding:utf-8
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt


labels = ['with edge', 'without edge']

colors = ['r--', 'g--', 'b--', 'm--', 'c--']

resultE = np.loadtxt('logE.txt')
lossE = resultE[:, 2]
valE = resultE[:, -1]

result = np.loadtxt('log.txt')
loss = result[:, 2]
val = result[:, -1]

plt.figure(1)
xE = list(range(len(lossE)))
plt.plot(xE, lossE, colors[0], mec='k', label=labels[0], lw=3)
x = list(range(len(loss)))
plt.plot(x, loss, colors[2], mec='k', label=labels[1], lw=3)
plt.ylabel('Value of loss', fontsize=16)
plt.xlabel('Epoch', fontsize=16)
plt.legend(loc='upper right', fontsize=16)
plt.grid(axis='y', linestyle='--')


plt.figure(2)
xE = list(range(len(valE)))
plt.plot(xE, valE, colors[0], mec='k', label=labels[0], lw=3)
x = list(range(len(val)))
plt.plot(x, val, colors[2], mec='k', label=labels[1], lw=3)
plt.ylabel('Validation score', fontsize=16)
plt.xlabel('Epoch', fontsize=16)
plt.legend(loc='lower right', fontsize=16)
plt.grid(axis='y', linestyle='--')

plt.show()