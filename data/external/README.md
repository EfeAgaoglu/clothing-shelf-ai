# Dataset kaynakları

Aktif akış iki hazır 7 sınıflı subset'i kullanır: DeepFashion2 10.000 train / 2.000 val ve Fashionpedia 10.000 train / 1.143 val. [Paylaşılan Drive rehberi](../../DRIVE_DOWNLOAD.md) arşivleri, sabit checksum'ları ve geri alma adımlarını içerir. Klasörü ham verilerle doldurmak veya subset'i yeniden üretmek devam etmek için gerekli değildir.

[Fashionpedia kaynak kaydı](fashionpedia/SOURCE.md), daha önceki dönüşümün kaynak dosyalarını, sınıf eşlemesini ve hash'lerini belgeler; bu dosyaların her yeni klonda bulunduğu anlamına gelmez. DeepFashion2 eşlemesi [proje değerlendirmesindedir](../../PROJECT_REVIEW.md#1-hedef-ve-mevcut-sonuç).

Kaynak dönüşümünü belgeleyen DeepFashion2/Fashionpedia araçları ve aktarım ZIP'i oluşturma aracı korunur. Bunlar güncel devam akışında çalıştırılmaz. Özellikle `scripts/build_7class_subset.py` mevcut hedefi yeniden oluşturabilir; hazır veriyi geri almak için kullanmayın.

Dataset/görsel erişimi yeniden dağıtım izni değildir; [kaynak koşullarını](../../DRIVE_DOWNLOAD.md#5-doğrulama-erişim-ve-lisans-sınırları) okuyun.
