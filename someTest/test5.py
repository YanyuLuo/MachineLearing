# 给你两个二进制字符串，返回它们的和（用二进制表示）。
# 输入为 非空 字符串且只包含数字1和0。
# 示例1:
# 输入: a = "11", b = "1"
# 输出: "100"
# 示例2:
# 输入: a = "1010", b = "1011"
# 输出: "10101"


def binarysystem(a: str, b: str):
    # j, k 是用来记录字符在字符串中的下标值的
    sum1, sum2, j, k = 0, 0, 0, 0
    for i in a:
        if int(i):
            sum1 += 2 ** (len(a) - (j + 1))
        j += 1

    for i in b:
        if int(i):
            sum2 += 2 ** (len(b) - (k + 1))
        k += 1
# [2:]是截取字符串，你可以去掉看看是什么效果
    return bin(sum1 + sum2)[2:]


print(binarysystem('1010', '1011'))
print(binarysystem('11', '1'))
