# Raf verisi ve segmentasyon etiketi kontrolü

Mevcut hazır eğitim verisi DeepFashion2 + Fashionpedia'dır; [Drive rehberinden](DRIVE_DOWNLOAD.md) geri alınır. Bu sayfa yeni bir raf dataset'i ileride hazır olduğunda uygulanacak veri sözleşmesini anlatır. Raf verisi üretmek veya elle maskeleme yapmak zorunlu bir sonraki adım olarak seçilmemiştir.

## Veri sözleşmesi

`shelf.yaml`, şu etiketli raf düzeni için ayrılmış yapılandırmadır. Git'teki `.gitkeep` dosyaları veri değildir; dataset hazır olmadan kontrolün hata vermesi beklenir.

```text
data/shelf_dataset/
  images/train/    labels/train/
  images/val/      labels/val/
```

Görüntü ve etiketin dosya gövdesi aynı olmalıdır: `images/train/raf_001.jpg` ve `labels/train/raf_001.txt`. Alt klasör yerine doğrudan split klasörlerini kullanın. Her görünür giysi instance'ı için ayrı poligon satırı gerekir:

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

Sınıf sırası `0: top`, `1: outwear`, `2: sleeveless_top`, `3: shorts`, `4: trousers`, `5: skirt`, `6: dress` şeklindedir; `outwear` yazımını koruyun. Koordinatlar görüntü genişliği/yüksekliğine bölünmüş 0–1 değerleri, en az üç farklı köşe içermelidir. Bounding box etiketleri instance maskesi değildir. Boş etiket yalnız doğrulanmış arka plan görüntüsünde kullanılmalıdır.

Üst üste duran giysileri tek raf yığını olarak etiketlemeyin; ayrı görünür sınırları kullanın. Sınıfı veya sınırı anlaşılamayan üründe etiket uydurmayın. Tahmin edilen maskeleri incelemeden gerçek eğitim etiketi saymayın.

Aynı raf, çekim serisi veya videonun benzer karelerini farklı split'lere dağıtmayın. Kontrol aracı yalnız byte düzeyinde aynı görüntülerin split sızıntısını yakalar; benzer kareleri grup bilgisiyle ayırmak gerekir. Test verisi hazırlanırsa YAML'a ayrı `test: images/test` ekleyin; test split'ini ayar seçimi için kullanmayın.

## Eğitim başlatmadan kontrol

Etiketli raf verisi hazır olduğunda Windows/CPU örneği:

```powershell
.\.venv\Scripts\python.exe scripts/check_shelf_dataset.py --data shelf.yaml
.\.venv\Scripts\python.exe scripts/preview_yolo_segmentation.py --data shelf.yaml --split val --count 7 --output runs/label_previews/shelf_val.jpg
.\.venv\Scripts\python.exe scripts/train_shelf.py --data shelf.yaml --model transfer/trained_runs/mixed_df2_fp_2epoch_20260923_110902/weights/best.pt --device cpu --batch 1 --imgsz 512 --epochs 1 --dry-run
```

Son model önceden [Drive'dan indirilmiş](DRIVE_DOWNLOAD.md) olmalıdır. Colab'da aynı araçlar `python` ile çağrılır; CUDA doğrulanınca dry-run için `--device 0` seçilebilir. Cihaz, model ve dataset'i daima açıkça belirtin. Bu örnekler eğitim başlatmaz.

Doğrulayıcı; okunabilir görüntü, eksik/yetim etiket, sınıf sırası, poligon koordinatı/alanı ve birebir split kopyalarını kontrol eder. Etiketin ürünün gerçek sınırına doğru çizildiğini, bütün self-intersection hatalarını veya benzer kareleri garantiyle doğrulamaz; görsel inceleme de gerekir.

## Modeli değerlendirme

Etiketli raf validation verisi hazırsa aynı split, çözünürlük ve cihazla başlangıç/aday checkpoint'leri ayrı değerlendirin:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_shelf.py --data shelf.yaml --model transfer/trained_runs/mixed_df2_fp_2epoch_20260923_110902/weights/best.pt --device cpu --imgsz 512 --split val
```

Kutu ve maske metriklerini, sınıf bazında sonuçları, kaçırılan ve yanlış ürünleri birlikte inceleyin. Etiketsiz fotoğraf karşılaştırması mAP ölçmez. Hazır maskeli yeni veri/model seçimi için ölçütler ve mevcut alan farkı [teknik değerlendirmededir](PROJECT_REVIEW.md#5-raf-görüntülerindeki-bulgular-ve-sonraki-karar).
