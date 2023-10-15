from scipy.spatial.distance import cdist
import numpy as np

'''
nums = np.array(list(range(1, 10)))
nums = nums.reshape(3, 3)
'''
nums = np.random.rand(5, 5)

print(nums)
print()

# 计算每一条记录与所有记录的距离之和
dist = cdist(nums, nums, 'euclidean').sum(axis=1)
print(dist)

# 自己的方法
dist_list = []
for i in nums:
    dist = 0
    for j in nums:
        temp = np.linalg.norm(i - j)
        dist += temp
    dist_list.append(dist)
print(np.array(dist_list))
