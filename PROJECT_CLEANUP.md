# Kullanılmayan dosyaların temizliği — 28 Eylül 2026

Temizlik, Colab'dan devam etme ve model karşılaştırması akışını koruyarak yapıldı. Eğitim veya benchmark başlatılmadı. Güncel kurulum, Drive'dan dosya alma ve eğitim hazırlığı adımları [COLAB_CONTINUE.md](COLAB_CONTINUE.md) içindedir.

## Kaldırılan eski araçlar

Aşağıdaki 11 dosyanın kayıtlı Colab notebook'unda ve kalan Python araçlarında kullanılmadığı kontrol edildi. Eski 13 sınıflı veri hazırlığı, sabit MPS cihazı kullanan eğitim denemeleri ve eski checkpoint'i seçebilen tahmin dosyaları kaldırıldı:

```text
deepfashion2.yaml
predict_7class.py
predict_deepfashion.py
predict_test.py
test_model.py
scripts/build_large_subset.py
scripts/check_7class_labels.py
scripts/check_labels.py
scripts/deepfashion2_to_yolo_subset.py
scripts/train_7class_test.py
scripts/train_subset.py
```

Güncel karşılıklar:

| İşlem | Kullanılacak araç |
|---|---|
| Tahmin / görsel çıktı | `scripts/predict_shelf.py` — son checkpoint için `--model` açıkça belirtilmeli |
| Dataset kontrolü | `scripts/check_shelf_dataset.py` |
| Eğitim hazırlığı / dry-run | `scripts/train_shelf.py` — eğitim yalnızca `--execute` ile başlar |
| Etiketleri görsel inceleme | `scripts/preview_yolo_segmentation.py` |
| DeepFashion2 sınıf yapılandırması | `deepfashion2_7class.yaml` |

Kaynak sınıf dönüşümünü belgeleyen 7 sınıflı dönüştürücü, Fashionpedia dönüştürme/subset araçları ve dataset dışa aktarma aracı korundu. Mevcut hazır subset yeniden üretilmedi.

## Yerel büyük dosyalar ve önbellekler

| Kaldırılan içerik | Kontrol / geri alma |
|---|---|
| `fashionpedia_balanced_10k_windows.zip` | SHA-256 yan dosyayla eşleşti; arşivdeki 22.287 dosyanın açılmış dataset içeriğiyle boyut ve SHA-256 eşleşmesi doğrulandı. Orijinal arşiv Drive'dan tekrar alınabilir. Açılmış dataset ve `.sha256` dosyası korundu. |
| `transfer/clothing-shelf-ai-0d7c381.zip` | 50 dosyanın eski Git commit'indeki içerikle aynı olduğu doğrulandı; metinlerde yalnızca Windows CRLF satır sonu farkları vardı. Kaynak içerik Git geçmişinde bulunur. |
| `.git/objects/82/tmp_obj_FVuXaD` ve `tmp_obj_qinVXp` | Git'in `garbage` olarak raporladığı, 23 Eylül'den kalan yarım geçici dosyalardı. Aktif Git işlemi ve kilit olmadığı kontrol edilerek kaldırıldı. Geçerli Git nesneleri ve Codex geri alma kayıtları korunuyor. |
| `runs/portable_predictions/predict-2/shelf_test.jpg` | Korunan `predict/shelf_test.jpg` ile SHA-256 değeri aynı olan kopya çıktıydı. |
| Benchmark Matplotlib font önbelleği ve Python `__pycache__` dosyası | Yeniden oluşturulabilen önbelleklerdi. |

Bu yerel arşiv/kopya/önbellek/geçici dosyalarının toplamı **1.857.759.450 bayt**, yaklaşık **1,86 GB (1,73 GiB)** idi. Bunlar kalıcı olarak kaldırıldı; iki arşiv için yukarıdaki kaynaklardan geri alma yolu mevcut. Git geçmişi yeniden yazılmadı; bu sayı GitHub'dan indirilen repo boyutunda aynı miktarda azalma anlamına gelmez.

`runs/shelf_7class/benchmark_cpu_20x20_retry/weights/` altındaki kullanılmayan `best.pt` ve `last.pt`, toplam 126.824.586 bayt, **Windows Çöp Kutusu'na taşındı**. Bu modeller yalnızca 20 train / 20 val örnekli CPU denemesine aitti; gerçek eğitim checkpoint'i olarak kullanılmamalı. Gerekirse Çöp Kutusu'ndan geri yüklenebilir. Çöp Kutusu boşaltılmadığı için bu yaklaşık 127 MB, hemen boşalan disk alanına dahil değildir. Benchmark'ın `args.yaml` ve `results.csv` kayıtları korundu.

Yalnızca yukarıdaki işlemlerden sonra boş kalan geçici/çıktı klasörleri kaldırıldı. Dataset ve aktif eğitim klasörlerinin yapısı değiştirilmedi. Kök dizindeki `*.zip` ve `*.zip.sha256` aktarım dosyaları `.gitignore` kapsamına alındı.

## Korunan önemli dosyalar

- `.venv` ve çalışan Python ortamı.
- `data/fashionpedia_balanced_10k`: 10.000 train ve 1.143 validation görüntüsü; etiketler ve subset manifesti.
- DeepFashion2 referans modeli: `runs/segment/runs/deepfashion2_7class/test_3epoch-2/weights/best.pt`.
- Fashionpedia referans modeli: `runs/shelf_7class/finetune_20260910_143218/weights/best.pt`.
- Kayıtlı Colab notebook'u, Colab metrik/ayar raporları ve yerel Word proje raporu.
- Drive'daki son Fashionpedia ve karma eğitim checkpoint'leri; Drive üzerinde hiçbir silme işlemi yapılmadı.
- Geçerli Git commit'leri, nesneleri ve geri alma kayıtları.

Eski scriptlerden birine gerçekten ihtiyaç olursa Git geçmişinden tek dosya geri alınabilir. Örneğin, aşağıdaki komut eski tahmin dosyasını geri getirir; güncel tahminlerde yine `scripts/predict_shelf.py` kullanılmalıdır:

```bash
git restore --source=d2c1865 -- predict_7class.py
```

## Doğrulama

Kaldırılan araçlara kalan Python dosyalarından veya kayıtlı notebook'tan bağımlılık bulunmadı. Yedi sentetik workflow testi geçti; kalan 13 Python dosyasının sözdizimi ve tahmin/eğitim/önizleme araçlarının yardım komutları kontrol edildi. İki referans checkpoint'in SHA-256 değerleri değişmedi. Git'in geçici dosya kontrolünde `garbage: 0` elde edildi. Eğitim başlatılmadı.
