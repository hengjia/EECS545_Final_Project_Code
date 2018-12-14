from matplotlib import pyplot as plt
import numpy as np

label_number = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
# label_number = [1,
# 3,
# 5,
# 10,
# 20,
# 50,
# 100,
# 150
# ]
LapSVM = [ 0.1850,
    0.1850,
    0.1870,
    0.1920,
    0.2370,
    0.3100,
    0.3840,
    0.4150
]
# LapSVM = [0.4790, 0.3170, 0.2890, 0.2390, 0.1150, 0.0810, 0.0720, 0.0480]
# SVM = [0.4780,
#     0.3600,
#     0.3210,
#     0.2800,
#     0.1890,
#     0.0870,
#     0.0730,
#     0.0470]
# LapSVM = [x * 100 for x in LapSVM]
# SVM = [x * 100 for x in SVM]
# LapSVM = [18.3, 44.4, 44.8, 45.6, 45.7, 45.8, 46, 45.8]
plt.semilogx(label_number, LapSVM, 'o-', label = 'LapRLS')
# plt.plot(label_number, SVM, 'd--')
plt.legend(loc = 'lower right')
plt.xlabel('gamma_i')
plt.ylabel('error rate')
plt.show()