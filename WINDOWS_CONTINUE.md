# Windows'ta devam: RX 6700 XT / Ryzen 5 3600X

Bu kurulum CPU üzerinde tahmin ve veri hazırlığı içindir. RX 6700 XT için resmî Windows PyTorch/ROCm desteği doğrulanmamıştır; `mps` veya NVIDIA'ya yönelik `--device 0` kullanmayın. Bu adımlar Mac'te hazırlandı; Windows üzerinde henüz çalıştırılmadı.

## 1. Kodu ve ağırlıkları alın

Git ve Python 3.11 kurduktan sonra PowerShell'de:

```powershell
git clone https://github.com/EfeAgaoglu/clothing-shelf-ai.git
cd clothing-shelf-ai
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-portable.txt
.\.venv\Scripts\python.exe -m pip check
```

Mac'in `.venv` klasörünü kopyalamayın. `requirements.txt` eski ortamın tam kaydıdır; Windows kurulumu için yukarıdaki daha küçük bağımlılık listesini kullanın. İki eğitilmiş `best.pt` zaten Git'te izleniyor ve klonla birlikte gelir.

## 2. Raf fotoğrafını deneyin

```powershell
.\.venv\Scripts\python.exe scripts/predict_shelf.py --source test_images/shelf_test.jpg --device cpu --conf 0.05 --imgsz 1024
```

Varsayılan model Fashionpedia 1 epoch ağırlığıdır. Maskeli çıktı `runs/portable_predictions/` içinde oluşur. Tahmin için büyük eğitim datasetini taşımak gerekmez.

## 3. İsteğe bağlı: hazır 10k datasetini taşıyın

Mac'te hazırlanan `transfer/fashionpedia_balanced_10k_windows.zip` dosyasını harici diskle Windows'a kopyalayın. Bu arşiv bağlantı yerine gerçek dosyalar içerir ve aynı 10.000 train / 1.143 validation seçimini korur. `transfer/` GitHub'a gönderilmez.

Windows'ta proje köküne ZIP dosyasını koyun, ardından:

```powershell
Get-FileHash .\fashionpedia_balanced_10k_windows.zip -Algorithm SHA256
.\.venv\Scripts\python.exe -m zipfile -e fashionpedia_balanced_10k_windows.zip .
.\.venv\Scripts\python.exe scripts/check_shelf_dataset.py --data fashionpedia_balanced_10k.yaml
```

Hash'i ZIP'in yanında verilen `.sha256` dosyasıyla karşılaştırın. Arşivi temiz hedefe açın; mevcut farklı datasetin üzerine açmayın. Bu yöntemde Windows Developer Mode veya sembolik bağlantı izni gerekmez. Veri zaten hazır olduğundan dönüşüm/subset scriptlerini yeniden çalıştırmayın. Tam Fashionpedia/DeepFashion2 kaynaklarını da saklamak isterseniz orijinal arşivleri ayrıca taşıyın; bu ZIP yalnız son deneyde kullanılan subset'i içerir.

## Kaldığımız nokta

- Model: YOLO26l-seg, sınıf sırası `top, outwear, sleeveless_top, shorts, trousers, skirt, dress`.
- Başlangıç: DeepFashion2 10k train / 2k val, 3 epoch.
- Son deney: Fashionpedia dengeli 10k train / 1.143 val, MPS, batch=1, imgsz=512, 1 epoch.
- Son model: `runs/shelf_7class/finetune_20260910_143218/weights/best.pt`.
- Gerçek kaydedilmiş epoch süresi 19.169,5 saniye (~5 saat 19 dakika); önceki kısa süre tahminleri iyimser kaldı.
- Raf testinde `imgsz=1024`, `conf=0.05`: sağdaki giysi %21,92 top; duvar/raf bölgesi %23,54 top yanlış pozitif. Güven artışı doğruluk artışı anlamına gelmez. Raf performansı yeterli değil.
- Yeni eğitim başlatılmadı. Yeni ağırlıktan fine-tuning için `--model` açık verilmelidir; `train_shelf.py` varsayılanı eski DeepFashion2 modelidir. `best.pt` ile yeni fine-tuning, optimizer durumuyla kesintisiz resume değildir.

Yalnız yapılandırmayı kontrol etmek için (eğitim yapmaz):

```powershell
.\.venv\Scripts\python.exe scripts/train_shelf.py --data fashionpedia_balanced_10k.yaml --model runs/shelf_7class/finetune_20260910_143218/weights/best.pt --device cpu --batch 1 --imgsz 512 --epochs 1 --dry-run
```

Yeni cihazda Codex'e şu mesajı verin:

> WINDOWS_CONTINUE.md ve PROJECT_REVIEW.md dosyalarını oku. Windows, RX 6700 XT ve Ryzen 5 3600X kullanıyorum. Önce ortamı, taşınan modelleri ve varsa dataset bütünlüğünü kontrol et; CPU ile raf tahminini doğrula. Son Fashionpedia ağırlığından devam edeceğiz. Eğitim başlatmadan önce onay iste. Mevcut dosyaları silme.
