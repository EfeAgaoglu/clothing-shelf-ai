# Raf verisiyle fine-tuning

## Büyük internet dataseti: Fashionpedia

Küçük aRTF/Roboflow örnekleri yerine akademik ve yaygın Fashionpedia indirildi. Resmi veri seti 48.825 görsel, 27 ana giysi sınıfı, 19 giysi parçası ve instance-segmentation maskeleri içerir. Kaynak arşivleri `data/external/fashionpedia` altında tutulur; mevcut dosyalar silinmedi. Yedi sınıfa dönüştürülen hazır veri `data/fashionpedia_yolo_7class` altındadır.

```bash
.venv/bin/python scripts/convert_fashionpedia_to_yolo.py
.venv/bin/python scripts/check_shelf_dataset.py --data fashionpedia_7class.yaml
.venv/bin/python scripts/preview_yolo_segmentation.py --data fashionpedia_7class.yaml --split val --count 7 --output debug_fashionpedia/preview.jpg
.venv/bin/python scripts/train_shelf.py --data fashionpedia_7class.yaml --device mps --batch 1 --imgsz 640 --epochs 30 --dry-run
```

Dönüşüm sonucu train 44.937 görsel/86.093 nesne, validation 1.143 görsel/2.016 nesnedir. Yedi sınıfın tamamı vardır. Tam veri kontrolü ve eğitim dry-run başarılıdır; eğitim başlatılmadı. Fashionpedia ağırlıklı olarak insan üzerindeki günlük yaşam, sokak, etkinlik, podyum ve çevrimiçi mağaza görsellerinden oluşur. Veri hacmini ve çeşitliliği artırır fakat raf alan farkını tek başına çözmez. Nihai raf başarısı için gerçek kullanıma benzeyen raf fotoğraflarıyla ayrıca fine-tuning gerekir.

Onaydan sonra Fashionpedia ile çalıştırılabilecek komut:

```bash
.venv/bin/python scripts/train_shelf.py --data fashionpedia_7class.yaml --device mps --batch 1 --imgsz 640 --epochs 30 --execute
```

### Dengeli 10 bin görsellik hızlı deneme

`scripts/create_balanced_fashionpedia_subset.py`, büyük dönüşümden seed=42 ile tekrar üretilebilir bir train subset'i oluşturur. Nadir sınıfları önceleyen seçim sonucu `data/fashionpedia_balanced_10k` altında 10.000 train ve tam 1.143 validation görseli vardır. Yapılandırma `fashionpedia_balanced_10k.yaml` dosyasındadır.

```bash
.venv/bin/python scripts/create_balanced_fashionpedia_subset.py
.venv/bin/python scripts/check_shelf_dataset.py --data fashionpedia_balanced_10k.yaml
.venv/bin/python scripts/train_shelf.py --data fashionpedia_balanced_10k.yaml --device mps --batch 1 --imgsz 512 --epochs 1 --execute
```

## Küçük internet dataset hazırlığı

Resmi aRTF Clothes 1.0.0 veri setinin 512x256 COCO segmentasyon sürümü indirildi ve MD5 doğrulandı. `top` ve `shorts` sınıfları `data/internet_shelf_7class` altında YOLO segmentasyona dönüştürüldü. Kaynak ve lisans kaydı `data/external/README.md` ile `data/external/aRTF/SOURCE.md` dosyalarında tutulur.

```bash
.venv/bin/python scripts/convert_artf_to_yolo.py
.venv/bin/python scripts/check_shelf_dataset.py --data internet_shelf.yaml
.venv/bin/python scripts/preview_yolo_segmentation.py --data internet_shelf.yaml
.venv/bin/python scripts/train_shelf.py --data internet_shelf.yaml --device mps --batch 1 --imgsz 640 --dry-run
```

Sonuç: train 251 (168 top, 83 shorts), val 63 (42 top, 21 shorts), bağımsız test 580 (400 top, 180 shorts). Beş proje sınıfı eksiktir ve görüntüler yoğun raf yığınları değildir. Bu veri tek başına yedi sınıflı uzun eğitimin nihai dataseti olarak değerlendirilmemelidir. Eğitim başlatılmadı.

## Güncel ayarlar: shelf_dataset / MPS

Son isteğe göre `scripts/train_shelf.py` varsayılanları `shelf.yaml`, `device=mps`, `batch=1`, `imgsz=640` oldu. Başlangıç ağırlığı aynı DeepFashion2 7 sınıflı `best.pt`; varsayılan süre 30 epoch. Eski `shelf_7class.yaml` ve klasörleri korundu. Aşağıdaki eski hazırlık notlarında geçen dataset adı, 512 çözünürlük ve otomatik cihaz seçimi yerine bu bölümdeki güncel ayarlar kullanılmalı.

Yeni veri konumları:

```text
data/shelf_dataset/images/train/
data/shelf_dataset/images/val/
data/shelf_dataset/labels/train/
data/shelf_dataset/labels/val/
```

Eğitim başlatmayan kontrol komutları:

```bash
.venv/bin/python scripts/check_shelf_dataset.py --data shelf.yaml
.venv/bin/python scripts/train_shelf.py --dry-run
```

Kullanıcının onayından sonra çalıştırılacak eğitim komutu (henüz çalıştırılmadı):

```bash
.venv/bin/python scripts/train_shelf.py --data shelf.yaml --device mps --batch 1 --imgsz 640 --epochs 30 --execute
```

Klasörlere gerçek raf fotoğrafları ve eşleşen segmentasyon etiketleri eklenmeden eğitim başlatılamaz. Dry-run, MPS üzerinde eğitim işlemi yapmaz; cihazın çalışabilirliğini kanıtlamaz. Değerlendirmede aynı veri ve çözünürlük için `scripts/evaluate_shelf.py --data shelf.yaml --imgsz 640 --device mps` kullanılır.

Mevcut dosyalar, modeller ve datasetler değiştirilmedi veya silinmedi. Yeni eğitim **başlatılmadı**. Aşağıdaki komutlar proje kökünde `.venv` ile çalıştırılır. Yeni scriptler başka çalışma dizininden de çağrılabilir; göreli `--data` ve `--model` proje köküne göre, YAML içindeki `path` YAML konumuna göre çözülür. Doğrudan `yolo train` yerine aşağıdaki scripti kullanın.

## Veri hazırlama

Oluşturulan yapı:

```text
data/shelf_7class/
  images/train/   labels/train/
  images/val/     labels/val/
  images/test/    labels/test/
shelf_7class.yaml
scripts/shelf_utils.py
scripts/check_shelf_dataset.py
scripts/train_shelf.py
scripts/evaluate_shelf.py
scripts/test_shelf_workflow.py
```

Klasörler bilinçli olarak boş. Gerçek fotoğraf ve elle doğrulanmış YOLO **segmentation** etiketlerinizi yerleştirin. Görüntü ve etiketin dosya gövdesi aynı olmalı: `images/train/raf_001.jpg` ve `labels/train/raf_001.txt`. Alt klasör kullanmayın. Her görünür ürün instance'ı için bir poligon satırı gerekir:

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

Koordinatlar görüntü genişliği/yüksekliğine bölünmüş, 0–1 aralığında olmalı; en az üç farklı köşe gerekir. Beş alanlı bounding box etiketi segmentasyon için yeterli değildir. Boş `.txt` yalnızca hedef sınıflardan hiçbir ürünün bulunmadığı, doğrulanmış arka plan görüntüsü içindir. Düşük confidence tahminlerini doğrulamadan gerçek etiket kabul etmeyin.

| ID | Sınıf | DeepFashion2 category_id |
|---|---|---|
| 0 | top | 1, 2 |
| 1 | outwear | 3, 4 |
| 2 | sleeveless_top | 5, 6 |
| 3 | shorts | 7 |
| 4 | trousers | 8 |
| 5 | skirt | 9 |
| 6 | dress | 10, 11, 12, 13 |

`outwear` yazımını ve bütün ID'leri aynen koruyun. Katlı kıyafetin sınıfı görselden anlaşılamıyorsa ürün bilgisiyle doğrulayın; görünmeyen sınırları uydurmayın. Üst üste duran ayrı ürünler görünür sınırlarına göre ayrı instance olarak etiketlenmeli; bir raf yığınını tek ürün saymak farklı bir görev olur. Tüm hedef ürünleri tutarlı biçimde etiketleyin.

Aynı raf, çekim serisi, video veya neredeyse aynı fotoğrafları tek split'te tutun. Yaklaşık %70/%20/%10 train/val/test başlangıç planı olabilir; küçük veride sınıf ve çekim çeşitliliğini korumak oranlardan daha önemlidir. Benzer kareleri rastgele bölmek sonuçları olduğundan iyi gösterir. Kontrol aracı yalnızca byte düzeyinde aynı görüntülerin split sızıntısını yakalar; benzer kareler elle/grup bilgisiyle ayrılmalıdır. Her sınıfa farklı ışık, mesafe, açı, raf ve katlama biçimleri ekleyin.

Test klasörü isteğe bağlıdır. Etiketli test verisi hazır olunca YAML'daki `test: images/test` satırını etkinleştirin. Test verisini hiperparametre seçimi için kullanmayın.

## Eğitim başlatmadan kontrol

```bash
.venv/bin/python scripts/check_shelf_dataset.py
.venv/bin/python scripts/train_shelf.py --dry-run
.venv/bin/python scripts/test_shelf_workflow.py
```

Raf klasörleri boşken ilk iki komutun çıkış kodu **2** olması beklenir. Kontrol; okunabilir görüntüler, eksik/yetim etiketler, aynı gövdeli görüntüler, sınıf sırası, poligon koordinatları/alanı ve splitler arası birebir kopyaları denetler. Geometrik kontrol, etiketlerin ürüne doğru çizildiğini veya poligonların kendisiyle kesişmediğini garanti etmez; eğitim öncesi görsel kontrol gerekir. Eksik sınıflar uyarı üretir; her split'te en az bir geçerli ön plan poligonu gerekir.

## Daha sonra kullanıcı tarafından başlatılacak eğitim

Önce veri hazır olduğunda kısa bir eğitim denemesi:

```bash
.venv/bin/python scripts/train_shelf.py --execute --epochs 1
```

Bu komut **bir epoch boyunca tüm raf train verisini işler**; süre dataset boyutuna bağlıdır. Bu inceleme sırasında çalıştırılmadı. Sonraki fine-tuning örneği:

```bash
.venv/bin/python scripts/train_shelf.py --execute --epochs 30
```

Başlangıç ağırlığı `runs/segment/runs/deepfashion2_7class/test_3epoch-2/weights/best.pt`. Yeni veri üzerinde yeni optimizasyon başlar (`resume=False`). Varsayılan AdamW, `lr0=0.0001`, 512 çözünürlük, batch 1, workers 0, seed 42, patience 10 ve sınırlı augmentasyon başlangıç ayarlarıdır; raf başarısı için doğrulanmış optimum ayarlar değildir. `optimizer=auto` manuel öğrenme hızını değiştirebildiği için açıkça AdamW seçildi. Mosaic/mixup/copy-paste kapalıdır. [Ultralytics eğitim ayarları](https://docs.ultralytics.com/modes/train/).

Cihaz otomatik CUDA → MPS → CPU sırasıyla seçilir. Örneğin `--device mps` ile zorlanabilir; bu inceleme oturumunda MPS görünmedi. Bellek uygunsa `--imgsz 640` denenebilir; karşılaştırmaları aynı çözünürlükte yapın. Model dosyası bulunamazsa otomatik indirme yerine hata verir. Çıktılar zaman damgalı yeni `runs/shelf_7class/finetune_*` klasörlerine gider; çözülmüş dataset YAML'ları `runs/shelf_7class/configs/` altında tutulur. Eski `best.pt` üzerine yazılmaz.

## Önce/sonra değerlendirme

Etiketli raf verisi hazır olduğunda başlangıç modelinin raf validation metriğini kaydedin:

```bash
.venv/bin/python scripts/evaluate_shelf.py --imgsz 512
```

Fine-tuning bittikten sonra aynı split ve çözünürlükle çalıştırın; aşağıdaki model yolunu oluşan gerçek çalışma adıyla değiştirin:

```bash
.venv/bin/python scripts/evaluate_shelf.py --model runs/shelf_7class/finetune_TIMESTAMP/weights/best.pt --imgsz 512
```

Etiketli test split'i YAML'a eklendiyse son değerlendirmede `--split test` kullanın. Kutu ve maske mAP50-95, sınıf bazında precision/recall, kaçırılan ve yanlış ürünleri karşılaştırın. Confidence tek başına doğruluk ölçütü değildir. İnsan üzerindeki başarının korunması isteniyorsa DeepFashion2 validation üzerinde de aynı modelle ayrı değerlendirme yapın (`--data deepfashion2_7class.yaml`); gerekirse sonraki deneyde ayrı bir karma eğitim dataseti hazırlayın.
