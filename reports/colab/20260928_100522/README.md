# Colab deney kayıtları

Bu klasör Tesla T4 üzerindeki iki tamamlanmış deneyin ayarlarını ve metriklerini saklar:

| Koşu | Veri | Epoch | Kayıtlar |
|---|---|---:|---|
| Fashionpedia Colab | 10.000 train / 1.143 val | 1 | [args.yaml](finetune_20260923_093619/args.yaml), [results.csv](finetune_20260923_093619/results.csv) |
| Karma DeepFashion2 + Fashionpedia | 20.000 train / 3.143 val | 2 | [args.yaml](mixed_df2_fp_2epoch_20260923_110902/args.yaml), [results.csv](mixed_df2_fp_2epoch_20260923_110902/results.csv) |

Kayıtlar değiştirilmez: `args.yaml` geçmiş koşunun yollarını/ayarlarını, `results.csv` o koşunun metriklerini içerir. Validation kümeleri farklı olduğundan doğrudan model üstünlüğü karşılaştırması değildir; raf performansını da ölçmez. Yorum ve süreler [PROJECT_REVIEW.md](../../../PROJECT_REVIEW.md#4-son-colab-eğitim-kayıtları) içindedir.

Son modeller Drive'dadır; [indirme rehberi](../../../DRIVE_DOWNLOAD.md) bunları güncel yerel yollarına alır. Geçmiş `args.yaml` dosyasını yeni oturumda düzenleyip kayıtların üzerine yazmayın; [Colab hazırlığı](../../../COLAB_CONTINUE.md#6-birleşik-20k-düzenini-geri-kur-ve-dry-run-yap) yeni YAML ve yalnız dry-run oluşturur.
