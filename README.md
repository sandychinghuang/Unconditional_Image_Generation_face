# Unconditional_Image_Generation_face
## 環境設置
您可以使用以下指令安裝爬蟲以及訓練模型所需環境：：
```
git clone https://github.com/sandychinghuang/Unconditional_Image_Generation_face.git
cd Unconditional_Image_Generation_face
pip install -r requirements.txt
```

## 爬蟲設置(dataset 準備)
### 您可以使用以下指令，爬取 ptt beauty 版指定 idex 之所有 image 連結 (正文以及留言)：
```
python scripts/crawl_img_url.py --start oooo --end xxxx
```
> 會自動從 idex=oooo 逐漸增加至 idex=xxxx

### 您可以使用以下指令，下載圖片、裁切出人臉並以64x64儲存：

    - 需更改`json_path`為準備爬取的 img_url 存放之 json 檔。
    - 需更改`face_crop_dir`為爬取下來之圖片存放位置。
```
python scripts/download_face_crop.py
```

## train model
### `scripts/train_att_Res.ipynb`為模型訓練程式
- 使用前須先更改`dataset_path`，請將所有訓練影像放在以下結構的資料夾中，方便`torchvision.datasets.ImageFolder`正確讀取：
    ```
    dataset/
    └── images/
    ├── img1.png
    ├── img2.png
    ├── img3.png
    └── ...
    ```
> 由於本專案是無條件生成任務，因此不需要真實的類別標籤，僅需確保所有圖片都在同一資料夾（例如`images/`）中即可。

- 模型儲存

訓練過程中，每 10 個 epoch 會同時儲存兩個模型權重檔案到`checkpoints/`資料夾：

- `unet_epoch{N}.pth`：訓練中的模型權重，代表每個 epoch 當下模型的參數狀態。
- `ema_unet_epoch{N}.pth`：指數移動平均（EMA）平滑後的模型權重，提供更穩定的模型參數。

建議 inference 使用`ema_unet_epoch{N}.pth`。

## generate image
您可以使用以下指令使用訓練完的模型生成圖片：
- 需更改程式內`model_gen_path`為欲使用的模型路徑。
```
python scripts/generate_image.py
```
> 會自動建立`generated_images`資料夾儲存生成圖片。