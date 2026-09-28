# Windows'ta CPU ile devam

Ryzen 5 3600X / RX 6700 XT bilgisayarda güvenilir yol CPU'dur. Bu projede Windows AMD GPU eğitimi doğrulanmış değildir; AMD kart için `mps` veya CUDA `--device 0` seçmeyin. Ana eğitim akışı [Colab rehberindedir](COLAB_CONTINUE.md).

## 1. Yeni klonda ortamı hazırlama

Git ve Python 3.11 kurulu olmalıdır. Repo özel kalırsa GitHub erişimi gerekir. PowerShell'de:

```powershell
git clone https://github.com/EfeAgaoglu/clothing-shelf-ai.git
cd clothing-shelf-ai
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-portable.txt -r requirements-download.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch, ultralytics; print('PyTorch:', torch.__version__); print('Ultralytics:', ultralytics.__version__); print('CUDA:', torch.cuda.is_available())"
```

Mevcut çalışan `.venv` ortamını yeniden oluşturmayın, başka cihazdan kopyalamayın. Kullanılan Ultralytics sürümü `requirements-portable.txt` içinde sabittir. Hata olursa gerçek çıktıyı inceleyin; rastgele paket sürümü değiştirmeyin.

## 2. Son modeli alma ve CPU tahmini

Proje kökünde iki Colab modelini ve orijinal test görsellerini indirin:

```powershell
.\.venv\Scripts\python.exe scripts/download_drive_assets.py --models --images --download
.\.venv\Scripts\python.exe scripts/predict_shelf.py --source test_images/shelf_test.jpg --model transfer/trained_runs/mixed_df2_fp_2epoch_20260923_110902/weights/best.pt --device cpu --conf 0.05 --imgsz 1024
```

Son deney modeli açıkça seçildi. Script yüklenen checkpoint'in `segment` görevini ve yedi sınıfın sırasını kontrol eder. Maskeli çıktı, scriptin yazdırdığı `runs/portable_predictions/...` klasöründedir. Tahmin için eğitim datasetleri gerekli değildir.

İki modeli aynı görsellerde karşılaştırmak ve AVIF görsellerini Pillow ile açmak için [paylaşılan Drive rehberindeki Colab akışını](DRIVE_DOWNLOAD.md#3-colabda-paylaşılan-bağlantılardan-devam) kullanın. Yeni modelin raf görevi için yeterli olduğu henüz doğrulanmış değil.

## 3. İsteğe bağlı: dataset kontrolü ve yalnız dry-run

Hazır subset'ler gerekiyorsa indirme aracı SHA-256/CRC kontrolüyle geri alır ve tam dataset doğrulaması yapar. Mevcut geçerli dataset korunur; yeniden üretilmez:

```powershell
.\.venv\Scripts\python.exe scripts/download_drive_assets.py --datasets fashionpedia deepfashion2 --download --extract
.\.venv\Scripts\python.exe scripts/check_shelf_dataset.py --data fashionpedia_balanced_10k.yaml
.\.venv\Scripts\python.exe scripts/train_shelf.py --data fashionpedia_balanced_10k.yaml --model transfer/trained_runs/mixed_df2_fp_2epoch_20260923_110902/weights/best.pt --device cpu --batch 1 --imgsz 512 --epochs 1 --dry-run
```

Bu komut eğitim başlatmaz, hız benchmark'ı yapmaz. Karma dataset hazırlığı için [Colab rehberinin 6. bölümünü](COLAB_CONTINUE.md#6-birleşik-20k-düzenini-geri-kur-ve-dry-run-yap) kullanın. `best.pt` ile ek fine-tuning, kesintili eğitimin optimizer durumuyla gerçek resume edilmesi değildir. Eğitim komutu ancak yeni deney için açık onaydan sonra hazırlanıp çalıştırılmalıdır.
