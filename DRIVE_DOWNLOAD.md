# Paylaşılan Drive dosyalarından devam etme

Kontrol tarihi: 28 Eylül 2026. [Paylaşılan aktarım klasörü](https://drive.google.com/drive/folders/1HUMyWWL6uGnDtQXR5A6O_B6uwSTcDKcL) oturum açmadan listelenebiliyor. Bu rehber yalnız indirme, doğrulama ve mevcut karşılaştırma/dry-run akışına geçiş içindir; eğitim başlatmaz.

## 1. Klasörde gerekenler var mı?

Evet, aşağıdaki dosyalar görüldü:

| Dosya | Kullanım |
|---|---|
| [Fashionpedia ZIP](https://drive.google.com/file/d/1CcNss1cNZPqz_95FtP5RVmfhJRiRwwqp/view) ve [SHA-256 kaydı](https://drive.google.com/file/d/1vKKl81WN87xiiHIvPYJgXmQPTAdUleos/view) | Hazır 10.000 train / 1.143 val görüntü ve segmentasyon etiketleri |
| [DeepFashion2 ZIP](https://drive.google.com/file/d/1719jkrHjD6Egq4_HqKhXxlZLkIk2PAjs/view) ve [SHA-256 kaydı](https://drive.google.com/file/d/1f8-3ndrEMlx4tuREWihRpa29fTJjIvAA/view) | Hazır 10.000 train / 2.000 val görüntü ve segmentasyon etiketleri |
| [Fashionpedia Colab best.pt](https://drive.google.com/file/d/1IY695U_jSEpjVvnotpb19DdDSBQHdqD2/view) | `finetune_20260923_093619` karşılaştırma modeli |
| [Karma 2 epoch best.pt](https://drive.google.com/file/d/1zOZoK_Z7MdaJiW96bPdb73BuDAaWX5AG/view) | `mixed_df2_fp_2epoch_20260923_110902` son deney modeli |
| [Orijinal test görselleri](https://drive.google.com/drive/folders/1WxM-EuZ-ujfHeYiOf7C3vKWy2rlFykcp) | Bir JPG ve iki AVIF; maskeli çıktılar değil |
| [Eğitim koşuları](https://drive.google.com/drive/folders/1gN1nN7SWQW9Kg4jvYJG1FQ8VfcHpQaM3) | Her iki koşuda `args.yaml`, `results.csv`, grafikler ve `weights/` |

İki koşuda da `last.pt`; karma koşuda ayrıca `epoch0.pt` ve `epoch1.pt` var. Dosyanın varlığı, optimizer durumunun korunmuş olduğunu veya gerçek resume yapılabileceğini kanıtlamaz. İndirme aracı yalnız iki `best.pt` dosyasını seçer. Küçük eğitim kayıtlarının kopyası zaten [GitHub'daki rapor klasöründedir](reports/colab/20260928_100522/).

Klasördeki eski `clothing-shelf-ai-0d7c381.zip` ve geçmiş karşılaştırma çıktıları gerekli değil; kaynak kod için güncel GitHub `main` klonunu kullanın. Klasörün tamamını indirmeyin.

Bu kontrolde yalnız iki küçük `.sha256` dosyası indirildi ve aşağıdaki kayıtlı değerlerle eşleşti. Büyük ZIP'ler ve Colab modelleri bu güncelleme sırasında indirilmedi veya yeniden yüklenmedi. Klasör envanteri, büyük dosyaların içerik doğrulaması değildir; bu kontroller kullanıcı indirdiğinde yapılır.

```text
Fashionpedia: b2b4179c1a135aefa21e20efc4defe16b2adf810551d8d1a45796c1b0ff44add
DeepFashion2: 766710e3de61e5f094ad261a64674cfaea395001089c836fee0857eabb95968a
```

## 2. Yerel bilgisayardan seçerek indirme

Proje kökünde, temel ortamı [Windows rehberine](WINDOWS_CONTINUE.md) göre hazırladıktan sonra PowerShell'de:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-download.txt
# Sadece dosyaları ve bağlantıları listeler; indirme yapmaz.
.\.venv\Scripts\python.exe scripts/download_drive_assets.py
# Karşılaştırma için iki model ve üç orijinal görsel yeterli.
.\.venv\Scripts\python.exe scripts/download_drive_assets.py --models --images --download
```

İki `best.pt` toplam yaklaşık 121 MiB; görseller birkaç MiB. Son model şu konuma gelir:

```text
transfer/trained_runs/mixed_df2_fp_2epoch_20260923_110902/weights/best.pt
```

Yalnız eğitim hazırlığı/dataset incelemesi gerekiyorsa:

```powershell
.\.venv\Scripts\python.exe scripts/download_drive_assets.py --datasets fashionpedia deepfashion2 --download --extract
```

İki ZIP toplam yaklaşık 1,5 GiB; açılmış veri ve geçici doğrulama klasörleri için ayrıca boş disk gerekir. Hazır veri `data/fashionpedia_balanced_10k` ve `data/deepfashion2_yolo_7class` altına açılır. Ham veriden dönüştürme veya subset yeniden üretme yapılmaz.

Script mevcut dosyanın üzerine yazmaz. Mevcut dataset sayılar, görüntüler, etiketler, poligonlar, sınıf ID'leri ve split kopyaları açısından doğrulanır; geçerliyse ZIP yeniden indirilmez. Farklı/bozuk dosya veya dataset bulunursa durur; otomatik silme/düzeltme yapmaz. İndirilen ZIP'ler açıldıktan sonra korunur.

Windows'ta orijinal repo test görseline tek CPU tahmini için (eğitim değildir):

```powershell
.\.venv\Scripts\python.exe scripts/predict_shelf.py --source test_images/shelf_test.jpg --model transfer/trained_runs/mixed_df2_fp_2epoch_20260923_110902/weights/best.pt --device cpu --conf 0.05 --imgsz 1024
```

AVIF görselleri karşılaştırırken aşağıdaki Colab hücresinin Pillow ile açma yolunu kullanın; orijinalleri silmeyin.

## 3. Colab'da paylaşılan bağlantılardan devam

GitHub deposuna erişim ile Drive erişimi ayrıdır. Paylaşılan Drive dosyaları için Google hesabı, kendi Drive'ınıza kopyalama veya `drive.mount()` gerekmez. GitHub deposu özel kalırsa kodu klonlayacak kişinin ayrıca repo erişimi ve GitHub kimlik doğrulaması gerekir. Daha önce kurduğunuz `gh auth setup-git` ve Colab Secrets `GITHUB_TOKEN` akışını kullanabilirsiniz; token'ı kod/URL içine yazmayın.

Önce güncel `main` kodunu `/content/clothing-shelf-ai` altında bulundurun. Yeni oturumda GitHub kimlik doğrulaması hazırsa bu hücre klonu oluşturur; mevcut klasörü silmez veya otomatik güncellemez:

```python
from pathlib import Path
import subprocess
import sys

ROOT = Path('/content/clothing-shelf-ai')
if not ROOT.exists():
    subprocess.run([
        'git', 'clone', '--branch', 'main',
        'https://github.com/EfeAgaoglu/clothing-shelf-ai.git', str(ROOT)
    ], check=True)
assert (ROOT / '.git').is_dir(), 'Mevcut klasör Git klonu değil; dosyalar korunuyor.'
assert (ROOT / 'scripts/download_drive_assets.py').is_file(), 'Güncel main kodu gerekli; önce git status ve dalı kontrol edin.'

subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
                str(ROOT / 'requirements-portable.txt')], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
                str(ROOT / 'requirements-download.txt')], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
subprocess.run([sys.executable, str(ROOT / 'scripts/download_drive_assets.py'),
                '--models', '--images', '--download'], check=True)

# Burada kişisel Drive yolu değil, Colab'ın yerel aktarım klasörü kullanılır.
TRANSFER = ROOT / 'transfer'
print('Aktarım:', TRANSFER)
print('Eğitim başlatılmadı.')
```

Colab'ın hazır PyTorch kurulumu için ayrıca rastgele CUDA/PyTorch paketi yüklemeyin. Kurulum hata verirse gerçek çıktıyı inceleyin. Modeli yalnız güvendiğiniz kaynaktan indirdikten sonra görev/sınıf ve cihaz kontrolü:

```python
import torch
import ultralytics
import yaml
import os
from ultralytics import YOLO

DEVICE = 0 if torch.cuda.is_available() else 'cpu'
NAMES = ['top', 'outwear', 'sleeveless_top', 'shorts', 'trousers', 'skirt', 'dress']
MODEL = TRANSFER / 'trained_runs/mixed_df2_fp_2epoch_20260923_110902/weights/best.pt'
OLD_MODEL = TRANSFER / 'trained_runs/finetune_20260923_093619/weights/best.pt'
print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__, 'Ultralytics:', ultralytics.__version__)
print('CUDA:', torch.cuda.is_available(), 'Tahmin cihazı:', DEVICE)
print('GPU:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
for checkpoint in (OLD_MODEL, MODEL):
    assert checkpoint.is_file(), f'Model bulunamadı: {checkpoint}'
    candidate = YOLO(str(checkpoint))
    assert candidate.task == 'segment'
    assert candidate.names == dict(enumerate(NAMES)), 'Sınıf sırası farklı.'
    print(checkpoint, candidate.names)
    del candidate
print('Model kontrolü tamamlandı; eğitim başlatılmadı.')
```

Şimdi [Colab rehberinin 4. bölümündeki karşılaştırma hücresini](COLAB_CONTINUE.md#4-eğitim-olmadan-görsel-karşılaştırması) çalıştırabilirsiniz. Yukarıdaki `ROOT`, `TRANSFER`, `MODEL`, `NAMES`, `DEVICE`, `YOLO` değişkenlerini kullanır. O rehberin kişisel Drive kurulum hücrelerini yeniden çalıştırmayın; `TRANSFER` bu akışta `/content/clothing-shelf-ai/transfer` olarak kalmalıdır.

Karşılaştırma çıktıları bu kez yerel `transfer/shelf_comparison_<zaman>/` altına gelir; Colab Dosyalar panelinden indirin veya kendi Drive'ınıza yeni bir klasörle kaydedin. `/content` çıktıları oturum sona erdiğinde kaybolabilir. Orijinal fotoğraflar JPG/AVIF olarak korunur; Pillow ile açılıp modele verilir. Bu görsel test mAP ölçümü veya raf başarısı garantisi değildir.

## 4. Yalnız ek eğitim hazırlığı: dataset ve dry-run

Karşılaştırma için bu bölüm gerekli değil. Aynı Colab oturumunda hazır subset'leri indirmek/açmak için:

```python
subprocess.run([
    sys.executable, str(ROOT / 'scripts/download_drive_assets.py'),
    '--datasets', 'fashionpedia', 'deepfashion2', '--download', '--extract'
], check=True)
print('Hazır subset kontrolü tamamlandı; eğitim başlatılmadı.')
```

Ardından [Colab rehberinin yalnız 6. bölümünü](COLAB_CONTINUE.md#6-birleşik-20k-düzenini-geri-kur-ve-dry-run-yap) çalıştırın. Datasetler zaten açıldığı için oradaki 5. bölüm gerekli değildir. 6. bölüm, train/val ayrımını koruyarak 20.000 train / 3.143 val birleşimini hazırlar, mutlak yol içeren YAML üretir, tam kontrol ve `--dry-run` yapar. Aynı oturumda yukarıdaki model kontrolü hücresini çalıştırmış olun; orada tanımlanan değişkenlere ihtiyaç duyar.

`--execute` otomatik eklenmez. Notebook'ta **Tümünü çalıştır** kullanmayın. Yeni eğitim için ayrıca kullanıcı onayı, cihaz ve deney ayarlarının seçimi gerekir. `best.pt` üzerinden ek fine-tuning yeni optimizer ile başlar; gerçek resume ile aynı şey değildir.

## 5. Doğrulama, erişim ve lisans sınırları

- Dataset ZIP'lerinde manifestte sabit SHA-256, indirilmiş checksum kaydı ve ZIP CRC kontrol edilir; açma güvenli geçici klasörde yapılır, tam dataset kontrolü geçmeden hedefe taşınmaz.
- Model ve fotoğraflar için yayıncı SHA-256 değeri henüz manifestte yok. Script checkpoint ZIP yapısını/CRC'yi veya görüntünün okunabilirliğini kontrol edip yerel SHA-256 yazdırır. Bu, dosyanın kaynağının doğruluğunu veya model sınıflarını kanıtlamaz; model görevi/sınıfları yukarıdaki ayrı yükleme hücresiyle kontrol edilir. İndirme aracı pickle/model kodu çalıştırmaz.
- Script yalnız açık `--download` ile indirme yapar; varsayılanı listelemedir. Giriş çerezlerini kullanmaz, TLS doğrulamasını kapatmaz. Drive erişimi veya kota hatası olursa gerçek hata çıktısını inceleyin; herkese açık link de sınırsız indirme garantisi değildir. [gdown resmi açıklaması](https://github.com/wkentaro/gdown/blob/main/README.md).
- Dosya ID'leri/yolları [drive_assets.json](drive_assets.json) içindedir. Dosyalar silinip yeniden yüklenirse ID değişebilir; manifesti yeni dosyalarla ve dataset hash'leriyle birlikte yeniden doğrulayın. Yeni model eski koşunun üzerine konulmamalıdır.
- Herkese açık bağlantı, dataset/görselleri yeniden dağıtma izni değildir. DeepFashion2 erişimi resmi başvuru akışına tabidir: [kaynak proje](https://github.com/switchablenorms/DeepFashion2). Fashionpedia'nın annotation/model/API/ontology lisansı ile üçüncü taraf fotoğraflarının hakları farklıdır: [resmi indirme ve lisans açıklaması](https://github.com/Fashionpedia/home-private/blob/master/Fashionpedia_download.html). Bu aktarım klasörünün yeniden dağıtım izni ayrıca doğrulanmış değildir. Kullanım ve paylaşımda asıl kaynak koşullarına uyun.
- Drive paylaşım ayarları bu güncellemede değiştirilmedi. İzinler için [Google'ın paylaşım açıklamasını](https://support.google.com/drive/answer/2494822?hl=tr) kullanın. `transfer/`, `data/`, yeni checkpoint'ler ve erişim token'ları Git'e eklenmez.

İndirme/açma güvenlik testleri çevrimdışı çalışır; gerçek Drive indirmesi veya eğitim yapmaz:

```powershell
.\.venv\Scripts\python.exe scripts/test_drive_download.py
.\.venv\Scripts\python.exe scripts/test_shelf_workflow.py
```
