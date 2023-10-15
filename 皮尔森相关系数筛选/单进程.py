####################
# 读取文件
####################

import pandas as pd

# 读取CSV文件
df = pd.read_csv('processed_data.csv')

# 只取部分数据
df = df.head(100)

# 提取标签列和数据列
data = df.drop(columns=['class'])
label = df['class']

print('data.shape =', data.shape)
print('label.shape =', label.shape)
print()

####################
# 计算距离
####################

from scipy.stats import pearsonr
import numpy as np

# 皮尔森相关系数
def calc_r(index):
    
    r_list = np.zeros(len(data))
    
    for i in range(len(data)):
        r, _ = pearsonr(data.iloc[index], data.iloc[i])
        r_list[i] = r
        
    return r_list.sum()



# 计算皮尔森相关系数
dists = []
for index in range(len(data)):
    if index%10 == 0: print(index//10, end=' ')  # 进度条
    correlations = calc_r(index)
    dists.append(correlations)



# 将距离之和保存到一个CSV文件中
with open('dist_single.csv', 'w') as f:
    for dist in dists:
        f.write(f'{dist}\n')















