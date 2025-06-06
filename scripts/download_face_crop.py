import os
import json
import requests
from tqdm import tqdm
from urllib.parse import urlparse
from PIL import Image
from io import BytesIO
from facenet_pytorch import MTCNN
from torchvision.transforms.functional import to_pil_image

def download_and_crop(json_path, output_dir, image_size=64):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        image_urls = json.load(f)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    mtcnn = MTCNN(image_size=image_size, margin=20, post_process=False, keep_all=True)
    skipped_count = 0

    for url in tqdm(image_urls, desc=f"下載並裁切 {os.path.basename(json_path)} 中"):
        try:
            filename = os.path.basename(urlparse(url).path)
            name, _ = os.path.splitext(filename)

            # 嘗試取得圖片
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                print(f"[!] 狀態碼錯誤 {resp.status_code}：{url}")
                continue

            img = Image.open(BytesIO(resp.content)).convert("RGB")
            faces, probs = mtcnn(img, return_prob=True)
            if faces is None:
                continue

            for idx, (face, prob) in enumerate(zip(faces, probs)):
                if prob is None or prob < 0.98:
                    continue

                save_path = os.path.join(output_dir, f"{name}_{idx}.jpg")
                if os.path.exists(save_path):
                    skipped_count += 1
                    continue  # 已存在就跳過

                to_pil_image(face.byte()).save(save_path, format="JPEG")

        except Exception as e:
            print(f"[!] 錯誤處理 {url}：{e}")

    print(f"\n✅ 裁切完成！跳過 {skipped_count} 張（已存在）")
    print(f"📁 輸出資料夾：{output_dir}")

if __name__ == "__main__":
    # 👉 根據需要修改以下路徑
    json_path = "image_urls.json"
    face_crop_dir = "./face_64x64_new"

    download_and_crop(json_path, face_crop_dir)
