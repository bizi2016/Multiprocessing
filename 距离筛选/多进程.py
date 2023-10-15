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
labels = df['class']

print('data.shape =', data.shape)
print('labels.shape =', labels.shape)

####################
# 计算距离
####################

from sklearn.metrics.pairwise import euclidean_distances
from multiprocessing import Pool, freeze_support

# 定义计算距离的函数
def calc_dist(index):
    
    dists_list = euclidean_distances( data.loc[index].values.reshape(1, -1),
                                      data.values )
    total_dist = dists_list.sum(axis=1)
    
    return total_dist



if __name__ == '__main__':
    
    # 调用freeze_support()确保在Windows系统上正常工作
    freeze_support()
        
    # 使用多进程计算每一条记录与所有记录的距离之和
    with Pool(processes=8) as pool:
        dists = pool.map(calc_dist, range(len(data)))

    # 将结果保存到一个CSV文件中，不包含表头
    with open('dist_multi.csv', 'w') as f:
        for dist in dists:
            f.write(f'{dist[0]}\n')




