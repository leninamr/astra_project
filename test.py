#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy.linalg as scp
import xlrd
import matplotlib.pyplot as plt

from processing import processing

# N - размер сигнала
# K - размер сжатого сообщения
N = 256
K = 60

# вектор(сигнал) - много нулей
x = [0, 1, -1]
x = np.random.choice(x, size=(N, 1), p=[0.96, 0.02, 0.02])

#x = np.zeros((N, 1))

#for i in range(N-1):
#    if i % 50 == 0:
#        x[i] = 1

#print(x.transpose())

# A =  np.random.randint(2, size = (10, 10))

fig, axs = plt.subplots(3, 3, figsize=(12, 7), constrained_layout=True)
# рандомное распределение бернулли
A = [0, 1]
A = np.random.choice(A, size=(K, N), p=[0.6, 0.4])
# ортоганализируем строки для линейнонезависимости уравнений
A = scp.orth(A.transpose()).transpose()

y = np.dot(A, x)
x0 = (A.transpose()).dot(y)
xp = processing(A, y)
# погрешность
er_b = abs(x-xp).sum()
#картиночки
axs[0,0].set_title('Исходный сигнал\n')
axs[0,1].set_title('Начальная точка\nРаспределение Бернулли')
axs[0,2].set_title('Восстановленный сигнал\n')
axs[0,0].plot(range(0, N), x)
axs[0,1].plot(range(0, N), x0)
axs[0,2].plot(range(0, N), xp)

# функция rand - равномерное распределение
A = np.random.rand(K, N)
A = scp.orth(A.transpose()).transpose()

y = np.dot(A, x)
x0 = (A.transpose()).dot(y)
xp = processing(A, y)
er_norm = abs(x-xp).sum()
#картиночки
axs[1,1].set_title('Равноемрное распределение')
axs[1,0].plot(range(0, N), x)
axs[1,1].plot(range(0, N), x0)
axs[1,2].plot(range(0, N), xp)

# функция randn генерит случайные величины по распределению гаусса
A = np.random.randn(K, N)
A = scp.orth(A.transpose()).transpose()

y = np.dot(A, x)
x0 = (A.transpose()).dot(y)
xp = processing(A, y)
er_g = abs(x-xp).sum()

axs[2,1].set_title('Распределение Гаусса')
axs[2,0].plot(range(0, N), x)
axs[2,1].plot(range(0, N), x0)
axs[2,2].plot(range(0, N), xp)

#print(xp)
# print(A)

#file_loc = "/Users/marina/PycharmProjects/untitled2/test_A.xls"
#book = xlrd.open_workbook(file_loc)
#sheet = book.sheet_by_index(0)
#print(sheet.ncols)
#A = []
#for row in range(sheet.nrows):
   # _row = []
  #  for col in range(sheet.ncols):
 #       _row.append(sheet.cell_value(row, col))
#    A.append(_row)

# A = np.asarray(A)
#A = np.array(A, dtype='float64')

print("Погрешность при распределении Бернулли: ", er_b)
print("Погрешность при равномерном распределении: ", er_norm)
print("Погрешность при распределении Гаусса: ", er_g)

