import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv("../datasets/train.csv")

# del df["Name"], df["Ticket"], df["Cabin"] # 今回使わない列データは削除する

df.head()
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10 # グラフが見きれないようにするためサイズを大きくしておく
df.hist(); # 一括でヒストグラムを描画する
plt.tight_layout() # グラフ同士が重ならないようにする関数
plt.show() # グラフの表示
