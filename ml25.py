import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')






data_dict = {-1:np.array([[1,7],
                         [2,8],
                         [3,8],]),
             1:np.array([[5,1],
                         [6,-1],
                         [7,3],])}


[[plt.scatter(ii[0],ii[1], color='b' if i == -1 else 'r') for ii in data_dict.get(i)] for i in data_dict]
# # [[plt.scatter(ii[0],ii[1], color=i == -1 and 'k' or 'g') for ii in data_dict.get(i)] for i in data_dict]
plt.show()