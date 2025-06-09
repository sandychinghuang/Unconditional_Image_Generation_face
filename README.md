# Unconditional_Image_Generation_face
## 環境設置
您可以使用以下指令安裝爬蟲、訓練、生成模型所需環境：
```
git clone https://github.com/sandychinghuang/Unconditional_Image_Generation_face.git
cd Unconditional_Image_Generation_face
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

下載圖片、裁切出人臉時，需額外用下面指令安裝`facenet-pytorch`套件:
```
pip install facenet-pytorch
```

因為此套件會自重新安裝`torch>=2.2.0,<2.3.0`，會和訓練、生成所需的torch版本衝突，因此處理完資料及後，訓練、生成前需重新安裝`torch`:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```


## 爬蟲設置 ( dataset 準備)
- 您可以使用以下指令，爬取 ptt beauty 版指定 idex 之所有 image 連結 ( 正文以及留言 )：
```
python crawl_img_url.py --start oooo --end xxxx
```
> 會自動從 idex=oooo 逐漸增加至 idex=xxxx

- 您可以使用以下指令，下載圖片、裁切出人臉並以64x64儲存：

    - 需更改`json_path`為準備爬取的 img_url 存放之 json 檔。
    - 需更改`face_crop_dir`為爬取下來之圖片存放位置。
```
python download_face_crop.py
```

## train model
`train_att_Res.ipynb`為模型訓練程式

- 使用前須先更改`dataset_path`，請將所有訓練影像放在以下結構的資料夾中，方便`torchvision.datasets.ImageFolder`正確讀取：

    ```
    dataset/
    └── images/
        ├── img1.jpg
        ├── img2.jpg
        ├── img3.jpg
        └── ...
    ```
> 由於本專案是無條件生成任務，因此不需要真實的類別標籤，僅需確保所有圖片都在同一資料夾（例如`images/`）中即可。

- 模型儲存

    訓練過程中，每 10 個 epoch 會同時儲存兩個模型權重檔案到`checkpoints/`資料夾：

    - `unet_epoch{N}.pth`：訓練中的模型權重，代表每個 epoch 當下模型的參數狀態。
    - `ema_unet_epoch{N}.pth`：指數移動平均（EMA）平滑後的模型權重，提供更穩定的模型參數。

    建議 inference 使用`ema_unet_epoch{N}.pth`。

## generate images
您可以使用以下指令使用訓練完的模型生成圖片：
- 需更改程式內`model_gen_path`為欲使用的模型路徑。
```
python generate_image.py
```
> 會自動儲存生成圖片到`generated_images`資料夾。