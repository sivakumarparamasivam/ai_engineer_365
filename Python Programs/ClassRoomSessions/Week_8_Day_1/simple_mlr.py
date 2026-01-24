import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import subprocess

data = {'x1': [1, 2, 3], 'x2': [2, 1, 4], 'Output': [6, 8, 14]}
x1 = np.array(data['x1'])
x2 = np.array(data['x2'])
y = np.array(data['Output'])

X = np.column_stack((np.ones(len(x1)), x1, x2))
coef, *_ = np.linalg.lstsq(X, y, rcond=None)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y, c='r')

x1_s = np.linspace(x1.min(), x1.max(), 10)
x2_s = np.linspace(x2.min(), x2.max(), 10)
x1g, x2g = np.meshgrid(x1_s, x2_s)
zg = coef[0] + coef[1] * x1g + coef[2] * x2g

ax.plot_surface(x1g, x2g, zg, color='red', alpha=0.3)

out_path = 'Python Programs/ClassRoomSessions/Week_8_Day_1/simple_mlr_3d.png'
plt.savefig(out_path)
print('Saved:', out_path)
if matplotlib.get_backend().lower() != 'agg':
    plt.show()
else:
    try:
        if sys.platform == 'darwin':
            subprocess.run(['open', out_path], check=False)
        elif sys.platform.startswith('linux'):
            subprocess.run(['xdg-open', out_path], check=False)
        elif sys.platform.startswith('win'):
            os.startfile(out_path)
    except Exception:
        pass
print('\nPredict the value')

try:
    x1_val = int(input('Enter x1 value: '))
    x2_val = int(input('Enter x2 value: '))
    predicted_value = coef[0] + coef[1] * x1_val + coef[2] * x2_val
    print('Predicted Value:', predicted_value)
except Exception as e:
    print('Error:', e)
