# Clothing Shelf AI

YOLO26l-seg ile yedi giysi sınıfı (`top`, `outwear`, `sleeveless_top`, `shorts`, `trousers`, `skirt`, `dress`) için segmentasyon projesi. DeepFashion2 başlangıç modeli ve Fashionpedia 10k/1 epoch deneme ağırlıkları depoda bulunur. Son Colab Fashionpedia ve karma DeepFashion2 + Fashionpedia (2 epoch) checkpoint'leri Google Drive'dadır. Büyük eğitim datasetleri depoya dahil değildir.

- [GitHub ve Google Drive'dan Colab'da devam: kurulum, karşılaştırma ve eğitim hazırlığı](COLAB_CONTINUE.md)
- [Paylaşılan Drive bağlantıları ve seçerek otomatik indirme](DRIVE_DOWNLOAD.md) — [dosya klasörü](https://drive.google.com/drive/folders/1HUMyWWL6uGnDtQXR5A6O_B6uwSTcDKcL)
- [Başka cihazda kurulum ve devam etme](README_GITHUB.md)
- [Raf dataseti ve eğitim rehberi](README_SHELF.md)
- [Güncel proje durumu ve teknik değerlendirme](PROJECT_REVIEW.md)
- [Kullanılmayan dosyaların temizliği ve geri alma bilgileri](PROJECT_CLEANUP.md)

Depodaki eski Fashionpedia referans modeliyle yerel test:

```bash
.venv/bin/python scripts/predict_shelf.py --source test_images/shelf_test.jpg --device auto --conf 0.05 --imgsz 1024
```

Son Colab modelini kullanmak için `--model` ile checkpoint yolunu açıkça belirtin. Kendi Drive'ını bağlamak için [Colab rehberini](COLAB_CONTINUE.md), paylaşılan bağlantıdan doğrudan dosya almak için [indirme rehberini](DRIVE_DOWNLOAD.md) izleyin. Görsel karşılaştırması için eğitim datasetleri gerekmez. Ek eğitim için hazır subset'ler ve dataset YAML'ı ayrıca geri alınmalıdır. İndirme aracı eğitim başlatmaz; dataset/görsel erişimi, kaynakların yeniden dağıtım izni anlamına gelmez.

