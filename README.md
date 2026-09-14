# Clothing Shelf AI

YOLO26l-seg ile yedi giysi sınıfı (`top`, `outwear`, `sleeveless_top`, `shorts`, `trousers`, `skirt`, `dress`) için segmentasyon projesi. DeepFashion2 başlangıç modeli ve Fashionpedia 10k/1 epoch deneme ağırlıkları depoda bulunur. Büyük eğitim datasetleri depoya dahil değildir.

- [Başka cihazda kurulum ve devam etme](README_GITHUB.md)
- [Raf dataseti ve eğitim rehberi](README_SHELF.md)
- [Proje incelemesi](PROJECT_REVIEW.md)

En yeni ağırlıklarla yerel test:

```bash
.venv/bin/python scripts/predict_shelf.py --source test_images/shelf_test.jpg --device auto --conf 0.05 --imgsz 1024
```

Bu test görselinde model bir gerçek üst giysiyi ve bir yanlış geniş bölgeyi algılıyor. Gerçek raf fotoğraflarında güvenilir sayım için ayrıca raf alanına ait etiketli veri gerekir.
