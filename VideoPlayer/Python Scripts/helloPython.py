import numpy as np

x = np.array([0,1,2,3,4,5])
y = np.array([2.1, 2.9, 4.15, 4.98, 5.5, 6])

z = np.polyfit(x, y, 1)
p = np.poly1d(z)

#plotting
import matplotlib.pyplot as plt
xp = np.linspace(-1, 6, 100)
plt.plot(x, y, '.', xp, p(xp))
plt.show()