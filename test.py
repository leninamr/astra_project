import numpy as np
import scipy.linalg as scp
import xlrd

from processing import processing
# N - размер сигнала
# K - размер сжатого сообщения
N = 256
K = 60

# вектор(сигнал) - много нулей
x = [0, 1, -1]

x = np.random.choice(x, size=(N, 1), p=[0.96, 0.02, 0.02])

# x = np.zeros((N, 1))

# for i in range(N-1):
#    if i % 50 == 0:
#        x[i] = 1

print(x.transpose())

# A =  np.random.randint(2, size = (10, 10))

# рандомное распределение бернулли
A = [0, 1]
A = np.random.choice(A, size=(K, N), p=[0.6, 0.4])
# ортоганализируем строки для линейнозависимости уравнений
A = scp.orth(A.transpose()).transpose()

y = np.dot(A, x)
xp = processing(A, y)
er_b = abs(x-xp).sum()

# функция rand - равномерное распределение
A = np.random.rand(K, N)
A = scp.orth(A.transpose()).transpose()

y = np.dot(A, x)
xp = processing(A, y)
er_norm = abs(x-xp).sum()

# функция randn генерит случайные величины по распределению гаусса
A = np.random.randn(K, N)
A = scp.orth(A.transpose()).transpose()

y = np.dot(A, x)
xp = processing(A, y)
er_g = abs(x-xp).sum()

print(xp)
# print(A)

#file_loc = "/Users/marina/PycharmProjects/astra_project/test_A.xls"
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

print("Погрешность при Бернулли: ", er_b)
print("Погрешность при равномерном распределении: ", er_norm)
print("Погрешность при Гаусса: ", er_g)