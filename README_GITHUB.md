# Başka cihazda devam etme

Windows RX 6700 XT / Ryzen 5 3600X için [Windows rehberini](WINDOWS_CONTINUE.md) kullanın. Aşağıdaki Unix komutları yerine PowerShell komutları ve bağlantısız dataset taşıma adımları orada bulunur.

Bu depo kodu, YAML yapılandırmalarını, küçük test görsellerini ve iki gerekli eğitilmiş `best.pt` dosyasını taşır. `.venv`, DeepFashion2/Fashionpedia datasetleri, indirilen arşivler, dönüştürülmüş görseller ve diğer eğitim çıktıları GitHub'a konmaz. Özellikle Fashionpedia dönüşümündeki görseller bu bilgisayardaki mutlak yollara işaret eden sembolik bağlantılardır; doğrudan başka cihaza kopyalanmaları güvenilir değildir.

## Depodaki ağırlıklar

| Model | Yol | SHA-256 |
|---|---|---|
| DeepFashion2 7 sınıf başlangıç modeli | `runs/segment/runs/deepfashion2_7class/test_3epoch-2/weights/best.pt` | `3a39fbea6f9f5077c58414e02bc39f7f2a592edccf65a23165664e36b5dd72d6` |
| Fashionpedia dengeli 10k, 1 epoch | `runs/shelf_7class/finetune_20260910_143218/weights/best.pt` | `b08db72eb1dd5518c9113515518a850a8cfa6dbab36e48ce79fe24a7f418274d` |

## Yeni cihazda kodu açma

Önce Python 3.11 ve Git kurun. Apple Silicon Mac kullanıyorsanız MPS için PyTorch'un desteklediği güncel macOS ortamını kullanın. NVIDIA GPU varsa eğitim komutunda `--device 0`, GPU yoksa `--device cpu` seçin. Diğer cihazdaki kurulumun tam uyumluluğunu `pip install` ve `torch` kontrolü belirler.

```bash
git clone https://github.com/EfeAgaoglu/clothing-shelf-ai.git
cd clothing-shelf-ai
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -c "import torch; print('MPS:', torch.backends.mps.is_available(), 'CUDA:', torch.cuda.is_available())"
```

Windows'ta Python yolu `.venv\\Scripts\\python.exe` olur. Depo private olduğu için klonlama sırasında GitHub kimlik doğrulaması gerekir.

Test görselinde taşınan en yeni modelle tahmin:

```bash
.venv/bin/python scripts/predict_shelf.py --source test_images/shelf_test.jpg --device auto --conf 0.05 --imgsz 1024
```

Çıktı `runs/portable_predictions/` altında oluşur. Mevcut raf görselinde bu model yalnız bir gerçek üst giysiyi ve bir geniş yanlış bölgeyi `top` olarak işaretledi; yeni cihazda aynı sonucu görmek yalnız taşınmanın doğrulamasıdır, raf başarısı ölçümü değildir.

## Eğitim datasını yeniden hazırlama

Fashionpedia'nın resmî train ve validation/test görsel arşivlerini ve iki `instances_attributes_*2020.json` dosyasını `data/external/fashionpedia/` içine indirin. Kaynak URL'ler, boyutlar ve checksum'lar [Fashionpedia kaynak kaydında](data/external/fashionpedia/SOURCE.md) var. Arşivleri aynı klasöre açın:

```bash
unzip -nq data/external/fashionpedia/train2020.zip -d data/external/fashionpedia
unzip -nq data/external/fashionpedia/val_test2020.zip -d data/external/fashionpedia
.venv/bin/python scripts/convert_fashionpedia_to_yolo.py
.venv/bin/python scripts/create_balanced_fashionpedia_subset.py
.venv/bin/python scripts/check_shelf_dataset.py --data fashionpedia_balanced_10k.yaml
```

DeepFashion2 ham verisi GitHub'da yoktur. Orijinal 13 sınıf datasetini de kullanacaksanız onu ayrıca resmî kaynaktan temin edip ilgili `data/` konumuna yerleştirin; bu deponun içerdiği iki checkpoint için ham DeepFashion2 verisi gerekli değildir. Kendi etiketli raf görselleriniz de `data/shelf_dataset/images/{train,val}` ve eşleşen YOLO segmentasyon etiketleri `data/shelf_dataset/labels/{train,val}` altında ayrıca taşınmalıdır.

Dataseti doğruladıktan sonra kısa eğitim denemesi:

```bash
.venv/bin/python scripts/train_shelf.py --data fashionpedia_balanced_10k.yaml --device auto --batch 1 --imgsz 512 --epochs 1 --dry-run
.venv/bin/python scripts/train_shelf.py --data fashionpedia_balanced_10k.yaml --device auto --batch 1 --imgsz 512 --epochs 1 --execute
```

İkinci komut gerçek eğitimi başlatır. Yeni cihazda yalnız tahmin yapacaksanız Fashionpedia veya DeepFashion2 datasetlerini indirmenize gerek yoktur.
