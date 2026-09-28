# Clothing Shelf AI

YOLO26l-seg ile yedi giysi sınıfı (`top`, `outwear`, `sleeveless_top`, `shorts`, `trousers`, `skirt`, `dress`) için segmentasyon projesi. DeepFashion2 başlangıç modeli ve Fashionpedia 10k/1 epoch deneme ağırlıkları depoda bulunur. Son Colab Fashionpedia ve karma DeepFashion2 + Fashionpedia (2 epoch) checkpoint'leri Google Drive'dadır. Büyük eğitim datasetleri depoya dahil değildir.

- [GitHub ve Google Drive'dan Colab'da devam: kurulum, karşılaştırma ve eğitim hazırlığı](COLAB_CONTINUE.md)
- [Başka cihazda kurulum ve devam etme](README_GITHUB.md)
- [Raf dataseti ve eğitim rehberi](README_SHELF.md)
- [Proje incelemesi](PROJECT_REVIEW.md)
- [Kullanılmayan dosyaların temizliği ve geri alma bilgileri](PROJECT_CLEANUP.md)

Depodaki eski Fashionpedia referans modeliyle yerel test:

```bash
.venv/bin/python scripts/predict_shelf.py --source test_images/shelf_test.jpg --device auto --conf 0.05 --imgsz 1024
```

Son Colab modelini kullanmak için `--model` ile checkpoint yolunu açıkça belirtin; model ve test görsellerini Drive'dan alma adımları [Colab rehberinde](COLAB_CONTINUE.md) yer alır. Görsel karşılaştırması için eğitim datasetleri gerekmez. Ek eğitim için hazır subset'ler ve dataset YAML'ı ayrıca geri alınmalıdır.

Bu test görselinde model bir gerçek üst giysiyi ve bir yanlış geniş bölgeyi algılıyor. Gerçek raf fotoğraflarında güvenilir sayım için ayrıca raf alanına ait etiketli veri gerekir.
