import numpy as np
import matplotlib.pyplot as plt

A2 = [[3, 4, 1, 2, 1, 5],
     [5, 2, 3, 2, 2, 1],
     [6, 2, 2, 6, 4, 5],
     [1, 2, 1, 3, 1, 2],
     [1, 5, 2, 3, 3, 3],
     [1, 2, 2, 4, 2, 1]]
A3 = [[2, 1, 1, 2],
      [1, 2, 3,2],
      [2, 1, 1, 2],
      [3, 1, 4, 1]]

A= A3
A = np.matrix(A)
A_T = np.transpose(A)
#A_inv = np.linalg.inv(A)
I = np.identity(A.shape[0])
#print(A_inv)

learning_rates = [0.003]
iters = 10000

for lr in learning_rates:
     plt.figure()
     B = np.random.rand(A.shape[0], A.shape[1])
     costs = []
     for i in range(iters):
          cost = np.sum((B@A-I)**2)
          costs.append(cost)
          B = B - lr * (2 * (B@A-I)@A_T)
     plt.plot(range(iters), costs)
     plt.title(lr)
     print(B)
plt.show()



