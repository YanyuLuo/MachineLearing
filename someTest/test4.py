# int函数可以用来把其他进制的数转为10进制，在后面加参数就好比如int(x, 2)，2进制转十进制
# bin函数可以把10进制转2进制
# 这些都是python的内置函数，直接用就好


def intsum(x: str, y: str):
    return bin(int(x, 2)+int(y, 2))[2:]


str1, str2 = input('请输入第一组二进制字符串：'), input('请输入第一组二进制字符串：')
s = intsum(str1, str2)
print('二进制求和结果：', s)
