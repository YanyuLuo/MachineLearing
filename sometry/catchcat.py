
import requests
import pandas as pd
from lxml import etree


# 首先定义一个类，并定义一些超参数
class taobaoSpider_1:
    def __init__(self):
        self.headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/86.0.4240.183 Safari/537.36',
            'Referer': 'https://login.taobao.com/',
            'cookie': '_med=dw:1920&dh:1080&pw:1920&ph:1080&ist:0; cna=1JkhEmU8+1oCAdpM/iZ1zFrw; '
                      't=70a038addc09c0f89223302a2d096de8; miid=1966132398820328471; '
                      'sgcookie=E100MYaeAbSKL%2BQZJXufFV8bVdH14dUIp9BqBMs9nnC%2Fme9XhH%2FUSJv6R%2B%2B'
                      '%2B4BuyMTs453uCdniUVeCYEIY02d1cEg%3D%3D; '
                      'uc3=vt3=F8dCufJKlxq4qPvtax0%3D&nk2=1oenv%2BO1xDoiLhj6DD4a4A%3D%3D&lg2=U%2BGCWk%2F75gdr5Q%3D%3D'
                      '&id2=UUwU3R9GXYQG9A%3D%3D; '
                      'lgc=%5Cu7B49%5Cu5F85%5Cu4EA6%5Cu662F%5Cu4E00%5Cu79CD%5Cu4EAB%5Cu53D7; '
                      'uc4=id4=0%40U27L8Ki9UGwX1IwxnXDvxdyzAOHn&nk4=0%401Dbxw7Du9rWPolPZb2TJmatdejIPSRXr9yxD; '
                      'tracknick=%5Cu7B49%5Cu5F85%5Cu4EA6%5Cu662F%5Cu4E00%5Cu79CD%5Cu4EAB%5Cu53D7; '
                      '_cc_=Vq8l%2BKCLiw%3D%3D; mt=ci=65_1; thw=cn; '
                      'enc=8yq7jtJTINoQPuAd6hNehNAfEAylZEgnVb%2FZSMdo0g29hYG2%2BryWTtU%2Bh0H%2FsThNjegNjKAo2pNlGirmlb'
                      '%2FOQA%3D%3D; hng=CN%7Czh-CN%7CCNY%7C156; '
                      '_m_h5_tk=5dce96b357c1e799b07d1277f2ad4e9f_1604678520261; '
                      '_m_h5_tk_enc=897f21ef2be25dfae5e69a184b4b2f69; v=0; uc1=cookie14=Uoe0abRrDlWGEw%3D%3D; '
                      '_tb_token_=31ee7dbedb18e; '
                      'isg=BPn5naJ_gk0SW1xTme5yG9mNCGXTBu24eFWvTxsudSCfohk0Y1b9iGfwJKZUGoXw; '
                      'tfstk=ceRRByawTmmuN__v7LH0OB2so6oGw3idczs39VwQmc-WWi1xq5qrBpWZ36IG.; '
                      'l=eBILFmX7qS5fRh2EBOfanurza77OSIRYYuPzaNbMiOCPO7fB5fiFWZSWky86C3GVh6ceR3J4RKMJBeYBqQAonxvtIosM_Ckmn; xlly_s=2 '
        }

    # 读取文件，把每件商品的url地址传入到列表中
    def get_url_list(self):
        file_path = "datalab/a.csv"
        df = pd.read_csv(file_path)
        url_list = df["link"].tolist()

        return url_list

    # 解析url地址，发送请求获得响应，返回页面源码
    def parse_url(self, url):
        response = requests.get(url, headers=self.headers)
        # print(response.content)
        return response.content.decode(encoding='gbk')

    # 使用xpath取出内容，然后返回一个字典列表
    def get_content_list(self, html_str):  # 提取数据
        html = etree.HTML(html_str)
        content_list = []
        item = {}
        # item["title"] = html.xpath("//*[@id=\"activity-name\"]/text()")
        # item["title"] = [i.replace("\n", "").replace(" ", "") for i in item["title"]]
        item["collect"] = html.xpath("//*[@id=\"J_CollectCount\"]//text()")
        item["collect"] = [i.replace("\n", "").replace(" ", "") for i in item["collect"]]

        content_list.append(item)

        return content_list

    # 保存html的函数
    def save_html(self, html_str, page_name):
        file_path = "html/lufa/{}.html".format(page_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_str)

    # 实现主要逻辑
    def run(self):
        # 获取url列表
        url_list = self.get_url_list()
        # 遍历url列表，发送请求，获取响应
        for url in url_list:
            num = url_list.index(url)
            print(num)
            # 解析url，获得html
            html_str = self.parse_url(url)
            # 获取内容
            content_list = self.get_content_list(html_str)

            # name = ['title', 'content']
            name = ['collect']
            if num < 1:
                test = pd.DataFrame(columns=name, data=content_list)
            else:
                test = test.append(content_list)
        # test.to_csv("datalab/b.csv", mode='a', encoding='utf-8')
        print('akak:', test)
        print('保存成功')


# 运行程序
if __name__ == '__main__':
    taobao_spider = taobaoSpider_1()
    taobao_spider.run()
