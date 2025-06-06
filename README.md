# Unconditional_Image_Generation_face
## 環境設置
您可以使用以下指令安裝爬蟲以及訓練模型所需環境:
```
git clone https://github.com/sandychinghuang/Unconditional_Image_Generation_face.git
cd Unconditional_Image_Generation_face
pip install -r requirements.txt
```

## 爬蟲設置(dataset 準備)
1. 您可以使用以下指令，爬取 ptt beauty 版指定 idex 之所有 image 連結(正文以及留言):
```
python scripts/crawl_img_url.py --start oooo --end xxxx
```
> 會自動從 idex=oooo 逐漸增加至 idex=xxxx

2. 您可以使用以下指令，下載圖片、裁切出人臉並以64x64儲存:
```
python scripts/download_face_crop.py
```
> `json_path`為準備爬取的 img_url 存放之 json 檔，`face_crop_dir`為爬取下來之圖片存放位置。

## train model
1. `scripts/train_att_Res.ipynb`為模型訓練程式。

## generate image
您可以使用以下指令使用訓練完的模型生成圖片:
```
python scripts/generate_image.py
```
> 需更改程式內`model_gen_path`為想使用的模型路徑。