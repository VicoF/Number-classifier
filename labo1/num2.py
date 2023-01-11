import numpy as np
import matplotlib.pyplot as plt
N = 2
rx = [-0.95, -0.82, -0.62, -0.43, -0.17, -0.07, 0.25, 0.38, 0.61, 0.79, 1.04]
x = np.asarray([[x**i for i in range(N+1)] for x in [-0.95, -0.82, -0.62, -0.43, -0.17, -0.07, 0.25, 0.38, 0.61, 0.79, 1.04]]).T
print(x)
y = np.asarray([0.02, 0.03, -0.17, -0.12, -0.37, -0.25, -0.10, 0.14, 0.53, 0.71, 1.53])

a = np.random.rand(N+1)

lr = 0.01
iters = 1000
costs = []
for i in range(iters):
    print(a)
    print(x.T)
    y2 = [a@x1.T for x1 in x.T]
    print(y2)
    cost = np.sum((y2-y)**2)
    grad = np.asarray([2*np.sum((y2-y)@xi) for xi in x])
    a = a - lr*grad
    costs.append(cost)
plt.scatter(rx, y)
y2 = [a@xi for xi in [[x**i for i in range(N+1)] for x in np.linspace(-1.25, 1.25, 1000)]]
plt.plot(np.linspace(-1.25, 1.25, 1000), y2)
plt.show()
print(a)
