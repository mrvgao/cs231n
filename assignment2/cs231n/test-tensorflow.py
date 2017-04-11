__author__ = 'Minchiuan Gao'
__mail__ = 'minchiuan.gao@gmail.com'

import seaborn
import numpy as np
import matplotlib.pyplot as plt

X = np.arange(100, step=.1)
y = X = 20 * np.sin(X/10)

#plt.interactive(False)
plt.scatter(X, y)
plt.show()

print('done')





