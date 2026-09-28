# Güncel proje durumu ve teknik değerlendirme

Güncelleme: **28 Eylül 2026**. Bu sayfa aktif dosyaları, tamamlanan Colab eğitimlerini ve bugün doğrulanan durumu anlatır. Adım adım kurulum için [Colab devam rehberini](COLAB_CONTINUE.md), proje ilerleyişinin anlatımı için [Word raporunu](Proje%20Ge%C3%A7mi%C5%9Fi%20ve%20Yap%C4%B1lanlar.docx) kullanın.

## 1. Hedef ve mevcut sonuç

Amaç, raf/askılık üzerindeki her görünür kıyafeti ayrı bir instance olarak maskelemek ve yedi sınıftan birine atamaktır. Kullanılan mimari YOLO26l-seg'dir; depodaki iki referans checkpoint 31.384.201 parametre içerir.

Google Colab / Tesla T4 üzerinde Fashionpedia ile 1 epoch fine-tuning ve ardından DeepFashion2 + Fashionpedia birleşimiyle 2 epoch eğitim tamamlandı. **Son tamamlanan deney karma 2 epoch modelidir; raf kullanımına hazır olduğu doğrulanmış bir model değildir.** Raf karşılaştırmalarında yanlış geniş maskeler, yanlış sınıflar ve kaçırılan ürünler devam ediyor. Bu nedenle yalnızca daha fazla epoch çalıştırmanın problemi çözeceği söylenemez.

Sınıf sözleşmesi bütün dataset ve checkpoint'lerde aynı kalmalıdır:

| ID | Sınıf | DeepFashion2 kaynak category_id |
|---|---|---|
| 0 | top | 1, 2 |
| 1 | outwear | 3, 4 |
| 2 | sleeveless_top | 5, 6 |
| 3 | shorts | 7 |
| 4 | trousers | 8 |
| 5 | skirt | 9 |
| 6 | dress | 10, 11, 12, 13 |

`outwear` yazımı değiştirilmemelidir; isim ve ID sırası modelle birebir eşleşmelidir.

## 2. Kullanılan hazır datasetler

| Dataset | Train görüntüsü | Validation görüntüsü | Hazır veri yolu |
|---|---:|---:|---|
| DeepFashion2, 7 sınıf | 10.000 | 2.000 | `data/deepfashion2_yolo_7class` |
| Fashionpedia subset, 7 sınıf | 10.000 | 1.143 | `data/fashionpedia_balanced_10k` |
| DeepFashion2 + Fashionpedia birleşimi | 20.000 | 3.143 | `data/mixed_df2_fashionpedia_20k` |

Birleşik küme, iki dataset'in kendi train ve validation ayrımlarını koruyarak hazırlandı; aynı sınıf ID'leri kullanıldı. Datasetlerin birleştirilmesi, modellerin paralel eğitilmesi değil, tek modelin iki kaynaktan örneklerle eğitilmesidir.

Büyük datasetler GitHub'da değildir. Bu Windows çalışma alanında Fashionpedia subset'i mevcut ve yeniden doğrulandı: `ready: true`, 10.000 train, 1.143 val, `errors: []`, `warnings: []`. DeepFashion2 ve karma dataset bu yerel ortamda mevcut değil; Colab'da kullanmak için Drive arşivlerinden geri alınmalıdır. 28 Eylül'de paylaşılan Drive klasöründe iki ZIP ve checksum dosyalarının varlığı ayrıca görüldü; indirilen iki checksum kaydı sabit SHA-256 değerleriyle eşleşti. Bu son envanter kontrolünde büyük ZIP'ler indirilip yeniden açılmadı. [Seçerek indirme rehberi](DRIVE_DOWNLOAD.md), indirme sonrasında hash/CRC ve tam dataset doğrulaması yapar.

Drive aktarım klasörü `clothing-shelf-ai-transfer` altında gerekli arşivler:

- `fashionpedia_balanced_10k_windows.zip` ve `.zip.sha256`.
- `deepfashion2_7class_10k2k.zip` ve `.zip.sha256`.

Hazır subset'leri yeniden üretmeyin. SHA-256 kontrolü, açma işlemi ve karma veri hazırlığı [Colab rehberinin 5–6. bölümlerindedir](COLAB_CONTINUE.md#5-yalnız-eğitim-hazırlığı-için-hazır-datasetleri-geri-al). Dataset'in sayısal olarak geçerli olması, her maskenin ürüne anlamsal olarak doğru çizildiğini garanti etmez.

Fashionpedia subset'i nadir sınıfları önceleyen seçimle hazırlandı; eşit sınıf dağılımı değildir. Örneğin validation'da `sleeveless_top` yalnızca 22 instance içerir; küçük sınıfların sonuçlarını genel ortalamadan ayrı değerlendirmek gerekir.

## 3. Checkpoint seçimi ve saklama

Aşağıdaki Drive yolları `/content/drive/MyDrive/clothing-shelf-ai-transfer/` klasörüne göredir:

| Model | Konum | Rol |
|---|---|---|
| Karma, 2 epoch | `trained_runs/mixed_df2_fp_2epoch_20260923_110902/weights/best.pt` | Son deney; ek eğitim başlangıcı ve karşılaştırma adayı |
| Fashionpedia Colab, 1 epoch | `trained_runs/finetune_20260923_093619/weights/best.pt` | Karma modelle karşılaştırılacak model |

GitHub klonunda ayrıca iki başlangıç/referans checkpoint bulunur:

| Referans model yolu | SHA-256 |
|---|---|
| `runs/segment/runs/deepfashion2_7class/test_3epoch-2/weights/best.pt` | `3a39fbea6f9f5077c58414e02bc39f7f2a592edccf65a23165664e36b5dd72d6` |
| `runs/shelf_7class/finetune_20260910_143218/weights/best.pt` | `b08db72eb1dd5518c9113515518a850a8cfa6dbab36e48ce79fe24a7f418274d` |

Bu iki dosya bugün yerel CPU ortamında yüklendi; görev `segment`, sınıf sırası yukarıdaki sözleşmeyle aynı. **Son Colab checkpoint'leri klonla otomatik gelmez** ve bu incelemede yerel olarak yeniden yüklenmedi. Drive'dan alınan checkpoint de tahminden önce görev ve sınıf sırası açısından doğrulanmalıdır.

`predict_shelf.py` varsayılanı depodaki Fashionpedia referansı; `train_shelf.py` ve `evaluate_shelf.py` varsayılanı DeepFashion2 referansıdır. Bu yüzden güncel deneylerde **`--model` açıkça belirtilmelidir**. Benchmark ağırlıkları eğitim başlangıcı olarak kullanılmaz.

`best.pt` ile ek fine-tuning, öğrenilmiş ağırlıklardan yeni optimizer ve takvimle başlar (`resume=False`). Kesintili eğitimin tam durumunu sürdürmek farklıdır: eğitim durumunu koruyan checkpoint ve `resume=True` gerekir. Mevcut `train_shelf.py` resume seçeneği sunmaz; tamamlanan koşudaki `last.pt` dosyasının adı tek başına gerçek resume garantisi değildir.

## 4. Son Colab eğitim kayıtları

Fashionpedia Colab koşusu depodaki Fashionpedia referansından; karma koşu ise bu yeni Colab Fashionpedia checkpoint'inden başladı. Her iki koşuda `device=0`, `batch=1`, `imgsz=512`, `optimizer=AdamW`, `lr0=0.0001`, `seed=42`, `workers=0`, `amp=True`, `resume=False` kullanıldı. Mosaic, mixup ve copy-paste kapalıydı. Tam ayarlar koşuların `args.yaml` dosyalarında saklanır.

Aşağıdaki değerler ilgili `results.csv` dosyasının **son epoch satırıdır**; checkpoint'ler bu incelemede yeniden validation'a sokulmadı:

| Koşu | Epoch | Box mAP50 | Box mAP50–95 | Mask mAP50 | Mask mAP50–95 | CSV'deki birikimli süre |
|---|---:|---:|---:|---:|---:|---|
| Fashionpedia Colab | 1 | %65,92 | %53,76 | %64,20 | %50,00 | 2.089,89 sn ≈ 34 dk 50 sn |
| Karma DeepFashion2 + Fashionpedia | 2 | %67,38 | %53,86 | %65,26 | %46,81 | 9.109,81 sn ≈ 2 sa 31 dk 50 sn |

Kaynak kayıtlar:

- Fashionpedia: [results.csv](reports/colab/20260928_100522/finetune_20260923_093619/results.csv), [args.yaml](reports/colab/20260928_100522/finetune_20260923_093619/args.yaml).
- Karma: [results.csv](reports/colab/20260928_100522/mixed_df2_fp_2epoch_20260923_110902/results.csv), [args.yaml](reports/colab/20260928_100522/mixed_df2_fp_2epoch_20260923_110902/args.yaml).

**Validation kümeleri farklıdır:** ilk koşu Fashionpedia, ikinci koşu birleşik validation kullanır. Bu tablodan karma modelin Fashionpedia modelinden üstün veya kötü olduğu sonucu doğrudan çıkarılamaz. Bu metrikler raf başarısını da ölçmez. Süreler kaydedilmiş koşulara aittir; veri aktarımı/kurulum dahil yeni Colab oturumunun toplam süresini veya başka GPU'nun hızını garanti etmez.

## 5. Raf görüntülerindeki bulgular ve sonraki karar

Paylaşılan tam görüntü ve parçalı tahmin karşılaştırmalarında:

- Birden fazla kıyafet, duvar veya raf bölgesi tek geniş maskede birleşebiliyor.
- Giysi türleri karışıyor; pantolon görüntüsünde farklı sınıflar üretilebiliyor.
- Bazı görüntülerde hiç tespit yok; bazılarında yalnızca düşük confidence çıktılar var.
- Parçalı tahmin bazı skorları artırsa da görüntü parçalarına benzeyen yanlış kutu/maskeler üretiyor; tek başına çözüm olarak doğrulanmadı.

İnsan üzerindeki eğitim verisi ile raf/askılık sahneleri arasındaki alan farkı, örtüşme ve sınırlı görünürlük olası etkenlerdir; tek bir nedene kesin teşhis konulmuş değildir. Etiketsiz görsellerde daha çok kutu veya daha yüksek confidence, daha iyi instance segmentation kanıtı değildir. Mevcut model çıktıları kontrol edilmeden yeni eğitim etiketi olarak kullanılmamalıdır.

Sonraki veri/model adayında kontrol edilecek ölçütler: insan üzerinde olmayan giysiler, hedefe benzer sıkışık/örtüşen raf sahneleri, ayrı instance maskeleri, yedi sınıfa anlamlı eşleme, kullanım lisansı ve train/val ayrımı. Yalnız bounding box etiketi olan bir dataset, maskeli dataset olarak doğrudan kullanılamaz. Tek tek manuel maskeleme zorunlu bir sonraki adım olarak seçilmiş değildir; hazır maskeli veri veya önceden eğitilmiş aday önce incelenmelidir.

Yeni eğitim kararı öncesinde modeller aynı görseller, çözünürlük ve eşikte karşılaştırılmalı; sayısal raf başarısı için bağımsız, doğrulanmış maskeleri olan bir değerlendirme kümesi kullanılmalıdır. Böyle bir raf benchmark'ı şu anda doğrulanmış değil. Eğitime otomatik devam edilmez.

## 6. Aktif araçlar ve dikkat edilmesi gerekenler

| Dosya | Güncel görevi |
|---|---|
| `scripts/check_shelf_dataset.py` | Görüntü/etiket, sınıf ID'si, poligon ve splitler arası byte düzeyinde kopya kontrolü |
| `scripts/predict_shelf.py` | Seçilen checkpoint ile tahmin, sınıf/confidence özeti ve maskeli çıktı |
| `scripts/train_shelf.py` | Dataset/checkpoint kontrolü; varsayılan dry-run, eğitim için açık `--execute` gerekir |
| `scripts/evaluate_shelf.py` | Etiketli validation/test kümesinde seçilen checkpoint'in değerlendirilmesi |
| `scripts/preview_yolo_segmentation.py` | Dataset etiketlerinin görsel incelemesi |
| `scripts/shelf_utils.py` | Sınıf sözleşmesi, yol çözümü, cihaz/model ve dataset kontrolleri |
| `scripts/test_shelf_workflow.py` | Yedi sentetik doğrulama testi |
| `scripts/download_drive_assets.py` | Paylaşılan Drive dosyalarını seçerek indirme, SHA-256/CRC ve hazır dataset geri alma; varsayılan yalnız listeleme |
| `scripts/test_drive_download.py` | Çevrimdışı indirme ve güvenli arşiv açma testleri |
| `scripts/test_repo_consistency.py` | Doküman bağlantıları, Python örnekleri, CLI dataset varsayılanları ve notebook arşiv uyarısı için çevrimdışı testler |
| `notebooks/clothing_machine_learning.ipynb` | Geçmiş Colab çalışma hücrelerinin arşivi; ilk hücrede uyarı, çıktı hücreleri temizlenmiş |

Koddan doğrulanan sınırlar:

- `train_shelf.py` varsayılanları `shelf.yaml`, `device=mps`, `imgsz=640`, `epochs=30` şeklinde kalıyor. Colab veya Windows'ta argümansız çalıştırmayın; cihazı, dataset'i ve modeli açıkça seçin. Bu temizlikte eğitim cihazı ve hiperparametre varsayılanları değiştirilmedi.
- Dataset kontrolü, değerlendirme ve etiket önizlemesinin varsayılanı `shelf.yaml` raf şablonudur; hazır subset'ler için `--data` açıkça verilir. `shelf_7class.yaml` ve kullanılmayan aRTF yapılandırması kaldırıldı.
- `read_config`, YAML köküne göre dataset yolunu mutlaklaştırır. Doğrudan Ultralytics kullanırken de doğru mutlak dataset yolunu sağlayın; `/content/datasets` altına yanlış yönlenme tekrar yaşanmamalı.
- Doğrulayıcı benzer ama byte düzeyinde farklı fotoğrafları, yanlış çizilmiş maskeleri veya bütün poligon kendisiyle kesişmelerini garantiyle yakalamaz.
- `build_7class_subset.py` hedef çıktı klasörlerini silerek yeniden oluşturur ve çok parçalı instance'ta en büyük poligonu seçer. Hazır aktarım verisini geri almak için çalıştırılmaz; import gerektirmeyen kod incelemesiyle kontrol edildi.
- GitHub notebook'unda geçmiş eğitim hücreleri bulunur; **Tümünü çalıştır** eğitim başlatabilir. Yalnız gerekli kurulum/karşılaştırma hücrelerini seçin.

## 7. Cihaz ve bu incelemede doğrulanan durum

Ana eğitim akışı Google Colab üzerinden yürütülüyor; son kaydedilmiş eğitimler Tesla T4 ile gerçekleştirildi. Çalışan kayıtlı Colab ortamı Python 3.13.15 / PyTorch 2.11.0+cu128 / Ultralytics 8.4.144 idi; yeni oturumda bu değerler ve CUDA kullanılabilirliği tekrar kontrol edilmelidir. GPU tahsisi önceki oturumla aynı olmak zorunda değildir.

28 Eylül'de dokümanın kontrol edildiği yerel Windows ortamı Python 3.11.9 / PyTorch 2.14.0+cpu / Ultralytics 8.4.144. CUDA ve MPS kullanılabilirliği `False`; bu ortamda CPU yolu kullanılır. RX 6700 XT, Colab'ın CUDA cihazıyla aynı şekilde seçilemez; bu projede Windows AMD GPU eğitimi doğrulanmış değildir.

Bu güncelleme sırasında yapılanlar:

- Yerel Fashionpedia subset'inde tam salt okunur görüntü/poligon/split kontrolü başarılı; hata ve uyarı yok.
- Depodaki iki referans checkpoint, doğru segmentasyon görevi ve yedi sınıfla başarıyla yüklendi.
- `scripts/test_shelf_workflow.py`: 7 test geçti.
- Colab metrikleri ve ayarlar depodaki kayıtlarla karşılaştırıldı; yeni model validation veya raf inference çalıştırılmadı.
- Paylaşılan Drive'da iki dataset, iki Colab `best.pt`, koşu kayıtları ve üç orijinal test görseli görüldü. Yalnız iki küçük checksum dosyası indirildi; Colab modelleri yeniden yüklenmedi. İndirme aracı için 14 çevrimdışı test geçti.
- Son repo temizliğinde toplam 26 çevrimdışı test, kalan 14 Python dosyasının sözdizimi ve altı aktif CLI yardım komutu başarılı. Dokümanların yerel bağlantıları ve Python/PowerShell örneklerinin sözdizimi kontrol edildi. Fashionpedia tekrar tam kontrolden geçti; depodaki Fashionpedia referansıyla CPU dry-run başarılı. İki referans modelin SHA-256 değerleri değişmedi.
- Yeni eğitim ve benchmark başlatılmadı.

## 8. GitHub / Drive'dan devam

Görsel karşılaştırması için GitHub kodu, Drive'daki iki Colab checkpoint'i ve orijinal maskesiz test görselleri yeterlidir; dataset arşivleri gerekmez. Ek eğitim hazırlığı için hazır dataset arşivleri de geri alınır, SHA-256 ve dataset kontrolleri yapılır, ardından yalnız dry-run çalıştırılır. Kullanıcı açıkça onaylamadan `--execute` kullanılmaz.

[COLAB_CONTINUE.md](COLAB_CONTINUE.md) kişisel Drive bağlantısı ile devamı, [DRIVE_DOWNLOAD.md](DRIVE_DOWNLOAD.md) paylaşılan bağlantılardan doğrudan indirmeyi çalıştırılabilir hücrelerle anlatır. Yeni ağırlıklar ve datasetler Drive'da; kaynak kod, temizlenmiş notebook ve küçük deney kayıtları GitHub'da tutulur. Herkese açık erişim, dataset/görselleri yeniden dağıtma izni olarak doğrulanmış değildir; asıl kaynak koşulları geçerlidir. Token, `.venv`, `data/`, aktarım ZIP'leri ve yeni checkpoint'ler topluca Git'e eklenmez. Güncel olmayan araçların kaldırılması ve geri alma bilgileri [temizlik kaydındadır](PROJECT_CLEANUP.md).
