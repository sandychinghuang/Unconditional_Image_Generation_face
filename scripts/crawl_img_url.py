#python crawl_img_url.py --start 3650 --end 3660

import os
import json
import requests
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import argparse

output_dir = "."
os.makedirs(output_dir, exist_ok=True)

def extract_image_urls_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    image_urls = []
    for a in soup.find_all("a", href=True):
        url = a["href"]
        # if re.match(r"^https?://", url) and re.search(r'\.(jpg|jpeg|png|gif)$', url, re.IGNORECASE):
        if re.match(r"^https?://", url) and re.search(r'\.(jpg|jpeg|png)$', url, re.IGNORECASE):

            image_urls.append(url)
    return image_urls

def get_article_list(page_content):
    soup = BeautifulSoup(page_content, "html.parser")
    urls = []
    for div in soup.find_all("div", class_="r-ent"):
        a_tag = div.find("div", class_="title").find("a")
        if a_tag and a_tag.get("href"):
            urls.append("https://www.ptt.cc" + a_tag["href"])
    return urls

def save_image_urls(image_urls, start_idx, end_idx):
    filename = f"image_{start_idx}_{end_idx}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"image_urls": image_urls}, f, ensure_ascii=False, indent=2)
    print(f"[✓] 已儲存目前累積的圖片數量：{len(image_urls)} 至 {filepath}")

def collect_image_urls(start_idx, end_idx):
    all_image_urls = []

    for idx, page in enumerate(tqdm(range(start_idx, end_idx + 1), desc="處理索引頁"), 1):
        url = f"https://www.ptt.cc/bbs/Beauty/index{page}.html"
        try:
            res = requests.get(url, cookies={"over18": "1"})
            if res.status_code != 200:
                continue
            articles = get_article_list(res.text)
            for article_url in articles:
                try:
                    art_res = requests.get(article_url, cookies={"over18": "1"})
                    art_res.encoding = "utf-8"
                    soup = BeautifulSoup(art_res.text, "html.parser")

                    main_content = soup.find(id="main-content")
                    pushes = soup.find_all("div", class_="push")
                    combined_html = str(main_content or "") + "".join(str(p) for p in pushes)

                    image_urls = extract_image_urls_from_html(combined_html)
                    all_image_urls.extend(image_urls)
                except Exception as e:
                    print(f"[!] 解析文章失敗 {article_url} — {e}")
        except Exception as e:
            print(f"[!] 讀取 index{page} 失敗 — {e}")

        # 每 10 頁儲存一次
        if idx % 10 == 0:
            save_image_urls(all_image_urls, start_idx, end_idx)

    # 最後補一次（確保結尾不是10整數時也會儲存）
    save_image_urls(all_image_urls, start_idx, end_idx)

    return all_image_urls

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True, help="起始 index")
    parser.add_argument("--end", type=int, required=True, help="結束 index")
    args = parser.parse_args()

    print(f"開始爬取 index{args.start} ~ index{args.end} ...")
    collect_image_urls(args.start, args.end)

if __name__ == "__main__":
    main()
