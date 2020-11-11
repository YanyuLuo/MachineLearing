class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode):
        if not l1 and not l2:
            return None
        if not l1 and l2:
            return l2
        if l1 and not l2:
            return l1
        headForList = ListNode(0)
        myListNode = headForList
        while l1 and l2:
            if l1.val <= l2.val:
                myListNode.next = l1
                l1 = l1.next
            else:
                myListNode.next = l2
                l2 = l2.next
            myListNode = myListNode.next
        if l1:
            myListNode.next = l1
        else:
            myListNode.next = l2
        return headForList.next


def main():
    # 初始化节点，并拆分输入的字符串，用来拼接出两个链表
    node1, node2 = ListNode(1), ListNode(1)
    n1, n2 = node1, node2
    str1 = input('请输入链表A的元素，每个元素用"->"隔开:')
    str2 = input('请输入链表B的元素，每个元素用"->"隔开:')
    list1, list2 = str1.split('->'), str2.split('->')
    for i in list1:
        n1.next = ListNode(int(i))
        n1 = n1.next
    for j in list2:
        n2.next = ListNode(int(j))
        n2 = n2.next

    # 调用合并函数
    node3 = Solution().mergeTwoLists(node1, node2)
    mynode = node3

    # 下面代码是为了输出漂亮一点
    flag = 0
    print('链表A与B合并后：')
    while mynode:
        if flag:
            print('->', end='')
        print(mynode.val, end='')
        mynode = mynode.next
        flag = 1


if __name__ == '__main__':
    main()
