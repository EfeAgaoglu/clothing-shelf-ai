# Clothing Shelf AI

YOLO26l-seg ile raf/askılık üzerindeki kıyafetleri ayrı instance maskeleriyle tanıma projesi. Sınıf sırası: `top`, `outwear`, `sleeveless_top`, `shorts`, `trousers`, `skirt`, `dress`.

## Mevcut durum

DeepFashion2 başlangıç eğitimi, Fashionpedia fine-tuning ve Google Colab / Tesla T4 üzerinde DeepFashion2 + Fashionpedia ile 2 epoch karma eğitim tamamlandı. Son deney karma modeldir; gerçek raflarda güvenilir instance ayrımı henüz doğrulanmış değildir.

GitHub kodu ve küçük deney kayıtlarını; [paylaşılan Google Drive klasörü](https://drive.google.com/drive/folders/1HUMyWWL6uGnDtQXR5A6O_B6uwSTcDKcL) hazır datasetleri, son Colab modellerini ve orijinal test görsellerini tutar. Klonla gelen iki checkpoint eski referanslardır; son modeli indirip `--model` ile açıkça seçin.

## Nereden devam etmeliyim?

| İşlem | Rehber |
|---|---|
| Paylaşılan bağlantıdan model/görsel indirme, Colab karşılaştırması ve isteğe bağlı dataset hazırlığı | [DRIVE_DOWNLOAD.md](DRIVE_DOWNLOAD.md) |
| Kendi Drive hesabını bağlayarak Colab'da devam | [COLAB_CONTINUE.md](COLAB_CONTINUE.md) |
| Windows'ta güvenilir CPU kurulumu ve tahmin | [WINDOWS_CONTINUE.md](WINDOWS_CONTINUE.md) |
| Segmentasyon etiketi formatı ve raf verisini kontrol etme | [README_SHELF.md](README_SHELF.md) |
| Model yolları, sınıf eşlemeleri, deney metrikleri ve teknik sınırlar | [PROJECT_REVIEW.md](PROJECT_REVIEW.md) |

Görsel karşılaştırması için yalnız modeller ve test görselleri gerekir; büyük datasetleri indirmeyin. Ek eğitim hazırlığında hazır subset'leri checksum kontrolüyle geri alın; yeniden üretmeyin. İndirme ve dry-run eğitim başlatmaz. Eğitim için ayrıca açık onay ve `--execute` gerekir.

Windows RX 6700 XT için bu projede AMD GPU eğitimi doğrulanmış değil; CPU yolunu kullanın. Colab'da GPU'yu PyTorch ile kontrol edin. Düşük confidence, daha çok tespit veya daha yeni checkpoint tek başına doğruluk kanıtı değildir.

## Proje kayıtları

- [Proje geçmişi ve yapılanlar — Word raporu](Proje%20Ge%C3%A7mi%C5%9Fi%20ve%20Yap%C4%B1lanlar.docx).
- [Colab deney ayarları ve metrik kayıtları](reports/colab/20260928_100522/README.md).
- [Geçmiş çalışma notebook'u](notebooks/clothing_machine_learning.ipynb) — arşivdir, güncel kurulum rehberi değildir; **Tümünü çalıştır** kullanmayın.
- [Temizlik kapsamı ve Git'ten geri alma](PROJECT_CLEANUP.md).

Dataset ve fotoğraf bağlantılarına erişim, yeniden dağıtım izni anlamına gelmez; [kaynak koşulları ve indirme sınırlarını](DRIVE_DOWNLOAD.md#5-doğrulama-erişim-ve-lisans-sınırları) okuyun. Token, `.venv`, `data/`, `transfer/` ve yeni eğitim ağırlıklarını Git'e eklemeyin.
