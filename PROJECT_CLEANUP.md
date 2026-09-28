# Repo temizliği ve geri alma

28 Eylül 2026. Güncel giriş [README.md](README.md); dosya indirme ve devam akışı [DRIVE_DOWNLOAD.md](DRIVE_DOWNLOAD.md) içindedir. Bu temizlik eğitim veya benchmark başlatmaz; Git geçmişini yeniden yazmaz.

## Son temizlik kapsamı

| Kaldırılan dosya | Neden |
|---|---|
| `README_GITHUB.md` | Colab/Drive/Windows rehberlerini yineliyor, eski model ve dataset yeniden üretme komutlarına yönlendiriyordu. |
| `requirements.txt` | Eski ortamın gereksiz uygulama paketleriyle dolu tam dökümüydü; kurulumda `requirements-portable.txt` ve isteğe bağlı `requirements-download.txt` kullanılıyor. |
| `internet_shelf.yaml`, `scripts/convert_artf_to_yolo.py`, `data/external/aRTF/SOURCE.md` | Aktif DeepFashion2/Fashionpedia/karma akışında kullanılmayan iki sınıflı aRTF denemesine aitti. |
| `shelf_7class.yaml` | Kullanılmayan eski raf klasörüne yönlendiriyordu; tek raf şablonu `shelf.yaml` olarak bırakıldı. |
| `scripts/analyze_deepfashion2.py`, `deepfashion2_class_distribution.csv` | Aktif araçlar/notebook tarafından kullanılmayan ham 13 sınıflı analiz aracı ve çıktısıydı. |
| `test_images/test.jpg` | Kalan kod ve notebook'ta kullanılmayan eski test görseliydi; aktif `shelf_test.jpg` korundu. |

Ana README, Windows ve raf verisi rehberleri güncel modele/hazır subset'lere göre sadeleştirildi. Eski MPS eğitim örnekleri ve otomatik `--execute` yönlendirmeleri çıkarıldı. Kontrol, değerlendirme ve etiket önizleme araçlarının varsayılan dataset'i, kaldırılan YAML'lar yerine `shelf.yaml` olarak birleştirildi; eğitim hiperparametreleri değiştirilmedi.

## Korunanlar

- Çalışan ortam ve hazır datasetler; yerel `.venv` / `data` üzerinde toplu silme yapılmadı.
- İki Git referans checkpoint'i, son Colab checkpoint'leri ve Drive dosyaları.
- Word proje geçmişi, geçmiş Colab notebook'u ve değiştirilmeyen deney `args.yaml` / `results.csv` kayıtları.
- Sınıf eşlemelerini/dataset üretim yöntemini belgeleyen DeepFashion2 ve Fashionpedia dönüşüm/subset araçları; güncel devam için çalıştırılmazlar.
- Git commit'leri ve geri alma kayıtları.

Notebook'un başına arşiv uyarısı eklendi; gerçek çalışma hücreleri korundu. Bu dosya güncel kurulum rehberi değildir ve **Tümünü çalıştır** eğitim başlatabilir. Word raporu da proje geçmişidir; güncel teknik durum [PROJECT_REVIEW.md](PROJECT_REVIEW.md) içindedir.

## Geri alma

Bağlantılar/kod örnekleri ve kalan CLI varsayılanları `scripts/test_repo_consistency.py` ile; dataset ve indirme güvenliği mevcut iki test dosyasıyla çevrimdışı kontrol edilir. Modeller, gerçek veri veya eğitim kullanılmaz.

Silinen bu dosyalar temizlik öncesi `547d466` commit'inde bulunur; Git geçmişinden geri getirilebilir. Örneğin gerçekten ihtiyaç varsa:

```bash
git restore --source=547d466 -- README_GITHUB.md
```

Önceki temizliğin ayrıntılı kayıtları da Git geçmişindedir. Repo geçmişi korunmuştur; çalışma ağacından silme eski dosyaları geçmiş commit'lerden kaldırmaz ve klon boyutunda aynı oranda küçülme garantilemez.
