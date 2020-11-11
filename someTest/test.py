# 我的环境里运行要导入这个包
from typing import List


class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        length = len(nums)
        if length <= 1:
            return nums[0]
        left = self.maxSubArray(nums[0: len(nums) // 2])
        right = self.maxSubArray(nums[len(nums) // 2: len(nums)])

        middle_left = nums[len(nums) // 2 - 1]
        auxiliary_1 = 0
        for index1 in range(len(nums) // 2 - 1, -1, -1):
            auxiliary_1 += nums[index1]
            middle_left = max(middle_left, auxiliary_1)

        middle_right = nums[len(nums) // 2]
        auxiliary_2 = 0
        for index2 in range(len(nums) // 2, len(nums), 1):
            auxiliary_2 += nums[index2]
            middle_right = max(middle_right, auxiliary_2)

        middle = middle_left + middle_right

        result = max(left, right, middle)

        return result


def main():
    # 如果写死数组就注释里这样就可以了
    # res = Solution().maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4])
    # print(res)
    strlist = input('请输入一个整数数组用“,”隔开每个元素：').split(',')
    mylist = []
    for i in strlist:
        mylist.append(int(i))

    print('最大和为：', Solution().maxSubArray(mylist))


if __name__ == '__main__':
    main()
