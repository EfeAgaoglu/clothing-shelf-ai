# Proje incelemesi — 10 Eylül 2026

Bu dosya 10 Eylül'deki incelemenin tarihsel kaydıdır. 28 Eylül'deki temizlikte kullanılmayan eski deneme scriptleri ve 13 sınıflı yapılandırma kaldırıldı; aşağıdaki dosya listesi o günkü durumu anlatır. Güncel araçlar ve geri alma bilgileri [temizlik kaydında](PROJECT_CLEANUP.md), Colab devam adımları [COLAB_CONTINUE.md](COLAB_CONTINUE.md) içindedir.

## Doğrulanan durum

- `.venv`: Python 3.11, torch 2.14.0, ultralytics 8.4.144. Bu araç oturumunda MPS kullanılabilirliği False; önceki eğitim args.yaml kayıtlarında `device: mps`.
- Belirtilen `best.pt` başarıyla yüklendi: task `segment`, ölçek `l`, başlık `Segment26`, beklenen 7 sınıf aynı sırada.
- 7 sınıflı subset: 10.000 train, 2.000 val. Tüm görüntüler PIL verify kontrolünden geçti; eksik/yetim etiket, bozuk link, geçersiz sınıf/koordinat, dejenere poligon ve splitler arası byte düzeyinde kopya bulunmadı.
- Bütün 12.000 etiket ham JSON ve görüntü boyutları kullanılarak mevcut dönüştürücünün algoritmasıyla yeniden hesaplandı; **0 uyuşmazlık**. Bu, dönüştürme tutarlılığıdır; tüm maskelerin anlamsal/görsel doğruluğu iddiası değildir.
- Train'de 11.236, val'de 351 nesnede birden fazla geçerli kaynak poligondan yalnızca en büyüğü seçilmiş. Bu sayılar nesne sayısıdır. Diğer görünür parçalar eğitim maskesine katılmıyor. Mevcut dataset korunmuştur; gelecekte çok parçalı maskelerin dönüşümü ayrı sürümde ele alınmalı.

| Sınıf | Train instance | Val instance |
|---|---:|---:|
| top | 5577 | 1180 |
| outwear | 694 | 133 |
| sleeveless_top | 987 | 163 |
| shorts | 1885 | 249 |
| trousers | 2865 | 576 |
| skirt | 1603 | 417 |
| dress | 2620 | 556 |

`top`, `outwear` sayısının yaklaşık 8 katı. Raf verisinde sınıf ve çekim çeşitliliğini ayrıca izlemek gerekir.

## Eğitim ve tahmin

Tamamlanan koşunun `results.csv` dosyasında 3 epoch var; üçüncü epoch sonunda toplam süre 23.405,7 saniye (~6,50 saat). Kutu mAP50: 0,63920; kutu mAP50-95: 0,49244; maske mAP50: 0,62571; maske mAP50-95: 0,43425. Bunlar DeepFashion2 validation metrikleri, raf performansı değil. Metrikler üç epoch boyunca artıyor; uzun eğitimin kazanımı bu kayıtla kesinleştirilemez.

Checkpoint ile CPU üzerinde `save=False` tahminler tekrarlandı:

| Görüntü | imgsz | conf eşiği | Sonuç |
|---|---:|---:|---|
| test_images/test.jpg | 512 | 0,05 | top %89,29; trousers %74,29 |
| test_images/shelf_test.jpg | 1024 | 0,05 | top %8,73 |

İkisinde de maske üretildi. Verilen sonuçlar yeniden üretildi. İnsan üzerindeki kıyafetlerle raf üzerindeki katlı/örtüşen ürünler arasındaki görüntü farkı olası temel etkendir; tek raf fotoğrafı ve etiketsiz test ile neden kesin kanıtlanamaz. Eğitim 512, raf tahmini 1024 çözünürlükte; karşılaştırmalı ölçümlerde bunu sabitlemek gerekir. Confidence eşiğini 0,05'e indirmek düşük skorlu çıktıyı gösterir, modelin öğrendiklerini iyileştirmez.

## Mevcut dosyalar ve riskler

| Dosya/klasör | İnceleme |
|---|---|
| scripts/analyze_deepfashion2.py | Ham 13 sınıf istatistiklerini CSV'ye çıkarır; mevcut CSV üzerine yazar. |
| scripts/deepfashion2_to_yolo_subset.py | İlk 500/100 örneği dönüştürür; eski çıktıları temizlemediği için farklı ayarla tekrar çalıştırmak eski dosyaları bırakabilir. Tek sayıda koordinat ve sınıf aralığı koruması eksik. |
| scripts/build_large_subset.py | 13 sınıf 10k/2k oluşturur; çıktı klasörlerini `rmtree` ile siler. Tek sayıda koordinat koruması eksik. |
| scripts/build_7class_subset.py | 13→7 map doğru, tek sayıda koordinatı eler. Çıktıları `rmtree` ile siler; bu incelemede çalıştırılmadı. Shuffle öncesi glob sıralanmadığından seed tek başına makineler arası aynı subset garantisi vermez. En büyük poligon seçimi bilgi kaybettirir. |
| scripts/check_labels.py, check_7class_labels.py | Birkaç örneğin görsel overlay'ini üretir; bütün datasetin doğrulaması değildir. Import sırasında çalışır ve debug görüntülerini yeniden yazar. |
| scripts/train_subset.py, train_7class_test.py | Sırasıyla 13/7 sınıf 3 epoch eğitim. Import sırasında eğitim başlatırlar; doğrulamak için import edilmedi. Göreli yollar ve zorunlu MPS başka ortamda sorun çıkarabilir. |
| predict_7class.py | Doğru checkpoint, raf fotoğrafı, conf=0,05, imgsz=1024. Göreli yol/MPS sabit; import sırasında tahmin ve kayıt yapar. |
| predict_deepfashion.py | `rglob('best.pt')[-1]` en yeni veya doğru sınıflı modeli garanti etmez. `agnostic_nms=True` sınıflar arası bastırma ayarıdır; etkisi modelin NMS/end-to-end moduna bağlıdır. |
| predict_test.py, test_model.py | Temel YOLO26 modelini kullanır; 7 sınıflı checkpoint testi değildir. Import yan etkileri vardır. |
| deepfashion2.yaml | 500/100'lük `deepfashion2_yolo` datasına bağlı; `deepfashion2_yolo_large` kullanılmıyor. |
| deepfashion2_7class.yaml | Doğru dataset ve doğru sınıf sırası; göreli path çalışma dizinine/Ultralytics ayarlarına bağlı olabilir. |
| requirements.txt | Tam ortam paketlerinin sürümleri sabitlenmiş; mevcut torch/ultralytics sürümleriyle uyum gözlendi. Temiz ortam kurulumu denenmedi. |
| data/deepfashion2_raw | Train 191.959 annotation, validation 32.153 annotation. Image dizinlerindeki ek girişler görüntü sayısı olarak yorumlanmadı. |
| data/deepfashion2_yolo, deepfashion2_yolo_large | Sırasıyla 500/100 ve 10k/2k görüntü/etiket girişi; 13 sınıf geçmiş denemeler korunuyor. Tam etiket taraması yalnızca aktif 7 sınıflı datasette yapıldı. |
| data/deepfashion2_coco | Üst seviyede alt dizin görülmedi; aktif eğitim YAML'larının hedefi değil. |
| debug_labels, debug_7class | Geçmiş görsel kontrol çıktıları, korundu. |
| runs | Önceki eğitim ve tahmin çıktıları korundu. `test_3epoch` yalnızca args kaydı; tamamlanan koşu `test_3epoch-2`. |
| yolo26l-seg.pt, test_images | Temel ağırlıklar ve iki test fotoğrafı korundu. |

`runs/segment/runs/...` yolu kurulu Ultralytics'in göreli `project` önüne runs kökü ve görev adını eklemesinden oluşuyor. Yeni script mutlak project yolu kullanıyor. Projede Git deposu bulunmadığından git diff/status ile karşılaştırma yapılamadı; mevcut dosyalara edit uygulanmadı. AGENTS.md talimatı bulunmadı.

## Eklenen hazırlık ve test sonuçları

`README_SHELF.md` kullanım adımlarını, `shelf_7class.yaml` sınıf sözleşmesini içerir. `shelf_utils.py` ortak kontrol/yol/cihaz/model işlevlerini, `check_shelf_dataset.py` salt okunur veri kontrolünü, `train_shelf.py` açık `--execute` ile fine-tuning'i, `evaluate_shelf.py` checkpoint karşılaştırmasını sağlar. Boş train/val/test görüntü ve etiket klasörleri oluşturuldu. Mevcut kodların riskleri burada belgelendi; geriye dönük davranışları değiştirilmedi.

Çalıştırılan kontroller:

- Tüm mevcut ve yeni Python dosyalarında AST syntax kontrolü: başarılı.
- `.venv/bin/python scripts/check_shelf_dataset.py --data deepfashion2_7class.yaml`: başarılı, 12.000 görüntü.
- Ham kaynak → etiket karşılaştırması: 12.000/12.000 uyumlu.
- `.venv/bin/python scripts/train_shelf.py --dry-run`: checkpoint ve Ultralytics seçenek kontrolü başarılı; boş raf verisi nedeniyle beklenen çıkış kodu 2, eğitim çağrılmadı.
- `/private/tmp` çalışma dizininden mutlak script yolu ile `train_shelf.py --dry-run --data deepfashion2_7class.yaml`: başarılı, çıkış kodu 0; 12.000 örnek, checkpoint ve seçenekler kontrol edildi, eğitim çağrılmadı. Çalışma dizininden bağımsız yol çözümü doğrulandı.
- `.venv/bin/python scripts/test_shelf_workflow.py`: 7 test başarılı; bozuk poligonlar, sınıf sırası, eksik etiket, arka plan-only split, split sızıntısı, yinelenen görüntü gövdesi ve geçerli göreli yollar.
- `.venv/bin/python scripts/evaluate_shelf.py --help`: başarılı.
- İki gerçek görüntüde checkpoint CPU inference: başarılı, önceki confidence sonuçları yeniden üretildi.

Uzun eğitim, kısa eğitim ve tam model validation yeniden çalıştırılmadı. Eğitim adımının ileri/geri yayılımı henüz test edilmedi; raf etiketleri geldikten sonra README'deki bir epoch denemesi bunun içindir.
