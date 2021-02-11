import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_path = '../output/kfold_results.csv'

df = pd.read_csv(file_path, thousands=',')
df = df[['#features tot', 'Macro-F1', 'Micro-F1']]
n_features = df['#features tot']
macro_f1 = df['Macro-F1']
micro_f1 = df['Micro-F1']

macro_color = 'tab:blue'
micro_color = 'tab:orange'

plt.scatter(n_features, macro_f1, label='Macro-F1')
plt.scatter(n_features, micro_f1, label='Micro-F1')
plt.plot(np.unique(n_features),
         np.poly1d(np.polyfit(n_features, macro_f1, 1))(np.unique(n_features)), color=macro_color)
plt.plot(np.unique(n_features),
         np.poly1d(np.polyfit(n_features, micro_f1, 1))(np.unique(n_features)), color=micro_color)
plt.xlabel('#features tot')
plt.ylabel('F1')
plt.legend(loc="upper right")
plt.show()