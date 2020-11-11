import requests
import random
import hashlib
import re


# 生成nonce
# 貌似是验证数据安全用的
def gen_nonce():
    a = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"]
    d = 0  # 计数
    nonce = ''
    while d < 9:
        r = 16 * random.random()  # 随机选择一个16以内的数值
        e = int(r)  # 数值取整
        nonce += a[e]
        d += 1
    return nonce


# 生成XYZ，nonce = nonce的值，api_part url接口的后面部分的路径，data POST提交的数值
# 也是用来验证之类的，没太细究
def gen_xyz(nonce, tapi_part, tdata):
    keys = list(tdata.keys())
    keys.sort()
    xyz_ = tapi_part + '?AppKey=joker'
    for key in keys:
        xyz_ += '&' + key + '=' + tdata[key]
    xyz_ += '&' + 'nonce=' + nonce
    xyz = hashlib.md5()
    xyz.update(xyz_.encode('utf-8'))
    xyz = xyz.hexdigest()
    return xyz


# 请求网页返回json数据
# 这个地方的cookies是第三方平台'新榜'上用自己微信登陆后，页面检查中copy下来的
def get_json(cookies, api_part, data):
    nonce = gen_nonce()
    xyz = gen_xyz(nonce, api_part, data)
    newdata = {
        'nonce': nonce,
        'xyz': xyz,
    }
    api_url = 'https://www.newrank.cn' + api_part
    data.update(newdata)
    res = requests.post(api_url, cookies=cookies, data=data)
    res.encoding = 'utf-8'
    return res.json()


# 获取微信公众号的uuid 通过帐号名来获取
# 这个方法正则那行会出问题,原因是返回的页面里没有uuid,导致search不到，所以后面没用它
def get_uuid(cookies, account):
    api_url = 'https://www.newrank.cn/public/info/detail.html?account=' + account
    res = requests.get(api_url, cookies=cookies)
    res.encoding = 'utf-8'
    r = re.search('uuid=(.+?)', res.text)
    tuuid = r.group(1)
    return tuuid


# 请使用自己登陆后获取的token
# 这个地方的cookies是第三方平台'新榜'上用自己微信登陆后，页面检查中copy下来的
cookies = {'token': '347F4E78462F4C289AAED0C856A6DF07'}
# -------------------搜索微信公众号信息-------------------
api_part = '/xdnphb/data/weixinuser/searchWeixinDataByCondition'
# 搜索用的关键字keyword
keyword = '浙江理工大学'
data = {
    'filter': '',
    'hasDeal': 'false',
    'keyName': keyword,
    'order': 'relation',
}
j = get_json(cookies, api_part, data)

print('--------------------------搜索含有关键词的公众号的结果----------------------------')
results = j['value']['result']
# myuuidlist是我暂时想到用来把循环中每次公众号的uuid记录下来
myuuidlist = []
for result in results:
    account = result['account']
    name = result['name'].replace('@font', '').replace('#font', '')
    tags = result['tags']
    certifiedText = result['certifiedText']
    area = result['area']
    myuuid = result['uuid']
    myuuidlist.append(myuuid)
    try:
        description = result['description'].replace('@font', '').replace('#font', '')
    except:
        description = ''
    # print(account, myuuid, name, certifiedText, area, description)
    print(account, name, certifiedText, area, description)

print('myuuidlist 所有值:', myuuidlist)

# -------------------查看公众号文章情况-------------------
print('')
print('-------------------公众号的最近七天热门文章信息点击数与在看数等-------------------------')
api_part = '/xdnphb/detail/getAccountArticle'
# 源代码为演示方便直接使用上面获取到的帐号，但get_uuid方法有问题所以我注释掉了
# uuid = get_uuid(cookies, myaccount)
uuid = myuuidlist[0]
data = {
    'flag': 'true',
    'uuid': uuid,
}
j = get_json(cookies, api_part, data)
topArticle = j['value']['topArticle']
for x in topArticle:
    # print(x)
    title = x['title']
    clickcount = x['clicksCount']
    likeCount = x['likeCount']
    publickTime = x['publicTime']
    print(title, clickcount, likeCount, publickTime)
# 最后这波拿到的是‘总浏览’数和‘在看’数，‘点赞’数第三方页面上有显示，但数据返回还是none,原因还没找到
# 另外一个问题就是这个第三方平台目前我看只能爬最近7天的，而且数据有延迟、有缺漏，不是实时的
