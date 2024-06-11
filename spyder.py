import requests
from bs4 import BeautifulSoup
import os

# 定义搜索关键词
search_term = "庆余年 袁梦"

# 创建保存图片的文件夹
if not os.path.exists(search_term):
    os.makedirs(search_term)

# 构建Google图片搜索URL
url = f"https://www.google.com/search?q={search_term}&tbm=isch"

# 发送HTTP请求并获取页面内容
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# 找到所有图片链接
img_tags = soup.find_all("img")

# 下载并保存图片
for i, img_tag in enumerate(img_tags):
    img_url = img_tag.get("src")
    if "http" in img_url:
        response = requests.get(img_url)
        with open(f"{search_term}/{i}.jpg", "wb") as f:
            f.write(response.content)
    print(f"已下载图片 {i+1}/{len(img_tags)}")

print("图片下载完成!")