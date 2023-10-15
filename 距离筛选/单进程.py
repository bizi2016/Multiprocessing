####################
# 读取文件
####################

import pandas as pd

# 读取CSV文件
df = pd.read_csv('processed_data.csv')

# 只取部分数据
df = df.head(20000)

# 提取标签列和数据列
data = df.drop(columns=['class'])
label = df['class']

print('data.shape =', data.shape)
print('label.shape =', label.shape)

####################
# 计算距离
####################

from scipy.spatial.distance import cdist

# 计算每一条记录与所有记录的距离之和
dists = cdist(data, data, 'euclidean').sum(axis=1)

# 将距离之和保存到一个CSV文件中
with open('dist_single.csv', 'w') as f:
    for dist in dists:
        f.write(f'{dist}\n')
