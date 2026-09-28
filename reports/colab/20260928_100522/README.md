# Colab ilerleme kaydı

- Model: YOLO26l-seg, instance segmentation.
- Cihaz: Google Colab / Tesla T4.
- Sınıf sırası: top, outwear, sleeveless_top, shorts, trousers, skirt, dress.
- Fashionpedia: 10.000 train, 1.143 validation görüntüsü.
- Birleşik DeepFashion2 + Fashionpedia: 20.000 train, 3.143 validation görüntüsü.
- Birleşik veri üzerinde 2 epoch eğitim gerçekleştirildi.
- Gerçek raf görüntülerindeki karşılaştırmalar yeterli instance ayrımı göstermedi.
- Büyük yanlış maskeler ve sınıf karışıklıkları devam ediyor.
- Checkpoint dosyaları Drive'daki clothing-shelf-ai-transfer/trained_runs altında.
- results.csv eğitim metriklerini, args.yaml kullanılan ayarları içerir.
- args.yaml içindeki Colab yolları başka cihazda düzenlenmelidir.
- Notebook çıktıları GitHub kopyasından kaldırıldı; Drive kopyası korunuyor.
