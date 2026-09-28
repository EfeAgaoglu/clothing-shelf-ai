# GitHub ve Google Drive'dan Colab'da devam etme

Güncelleme: 28 Eylül 2026. Bu rehber yeni bir Colab oturumunda kodu, kayıtlı modelleri ve hazır veriyi geri almayı anlatır. Kurulum, görsel karşılaştırması ve dry-run hücreleri eğitim başlatmaz. Notebook'ta **Tümünü çalıştır** kullanmayın; geçmiş eğitim hücrelerini yalnızca yeni eğitim kararı verildiğinde çalıştırın.

## 1. Hangi dosya nerede?

GitHub kaynak kodunu ve yapılandırmaları saklar. Colab kaydı 28 Eylül 2026'da `main` dalına birleştirildi: `notebooks/clothing_machine_learning.ipynb` ve `reports/colab/20260928_100522/` altında notebook, `results.csv`, `args.yaml` ve deney özeti bulunur. Bu kayıt için yeni klonda `main` dalını kullanabilirsiniz. Daha sonra ayrı bir ilerleme dalı açılıp henüz birleştirilmediyse o dalı ayrıca seçin.

Drive'daki ana aktarım klasörü:

```text
/content/drive/MyDrive/clothing-shelf-ai-transfer
```

Aşağıdaki yollar bu klasöre göredir:

| Dosya / klasör | Kullanım |
|---|---|
| `trained_runs/mixed_df2_fp_2epoch_20260923_110902/weights/best.pt` | Son karma deney: DeepFashion2 + Fashionpedia, 2 epoch. Ek eğitim için başlangıç veya karşılaştırma adayı. |
| `trained_runs/finetune_20260923_093619/weights/best.pt` | Colab'daki Fashionpedia deneyi; karşılaştırma modeli. |
| Her iki koşunun `results.csv` ve `args.yaml` dosyaları | Metrikler ve kullanılan ayarların kaydı. `args.yaml` içindeki eski `/content` yolları yeni oturumda yeniden hazırlanır. |
| Koşunun `weights/last.pt` dosyası, varsa | Kesintili eğitimi sürdürmek için ancak gerekli eğitim durumu da checkpoint'te korunmuşsa kullanılabilir. Son kontrolde `last.pt` yeniden doğrulanmadı. |
| `fashionpedia_balanced_10k_windows.zip` ve `.zip.sha256` | Hazır Fashionpedia subset'i: 10.000 train, 1.143 val. Eğitim için gerekir. |
| `deepfashion2_7class_10k2k.zip` ve `.zip.sha256` | Hazır DeepFashion2 subset'i: 10.000 train, 2.000 val; görüntü + etiket toplamı 24.000 dosya. Dosya adı kaydedilen notebook'tan doğrulandı. |
| `shelf_eval_images/` | Orijinal, maskesiz raf/askılık test görselleri. Karşılaştırma için kullanılır; eğitim verisi değildir. |

Asıl notebook ayrıca `/content/drive/MyDrive/Colab Notebooks/clothing_machine_learning.ipynb` yolunda doğrulandı. Drive'da bu notebook'u Colab ile açabilirsiniz. GitHub kopyasının temizlenmiş çıktıları, asıl notebook'un Drive'daki çıktılarının silindiği anlamına gelmez.

Repo klonunda bulunan iki **eski referans** model:

```text
runs/segment/runs/deepfashion2_7class/test_3epoch-2/weights/best.pt
runs/shelf_7class/finetune_20260910_143218/weights/best.pt
```

Son karma model GitHub klonuyla kendiliğinden gelmez. `train_shelf.py` varsayılanı eski DeepFashion2, `predict_shelf.py` varsayılanı eski Fashionpedia checkpoint'idir. Son deney için her zaman modeli açıkça seçin.

## 2. Yeni oturumda GitHub ve Drive bağlantısı

Colab'da sol paneldeki **Secrets / anahtar** bölümüne `GITHUB_TOKEN` adında token ekleyin ve **Notebook access** seçeneğini açın. Token'ın yalnız `EfeAgaoglu/clothing-shelf-ai` deposuna erişmesi yeterlidir; push için `Contents: Read and write` gerekir. Token'ı hücreye veya Git remote URL'sine yazmayın.

Eğitim hazırlığı için **Çalışma zamanı → Çalışma zamanı türünü değiştir → GPU** seçin; Tesla T4 sunuluyorsa onu kullanabilirsiniz. Tahmin CPU üzerinde de çalışır. Seçilen GPU'yu 3. bölümde PyTorch ile ayrıca doğrulayacağız.

Bu hücreyi çalıştırın. `BRANCH` değerini, ilerleme kaydını içeren dala göre seçin. Klasör zaten varsa silinmez veya üzerine klon yapılmaz.

```python
from pathlib import Path
from google.colab import drive, userdata
import os
import shutil
import subprocess

drive.mount("/content/drive")
os.environ["GH_TOKEN"] = userdata.get("GITHUB_TOKEN")

if shutil.which("gh") is None:
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(["apt-get", "install", "-y", "gh"], check=True)
subprocess.run(["gh", "auth", "setup-git"], check=True)

BRANCH = "main"  # Birleştirilmediyse gerçek codex/colab-progress-... dalını yazın.
ROOT = Path("/content/clothing-shelf-ai")
TRANSFER = Path("/content/drive/MyDrive/clothing-shelf-ai-transfer")

if not ROOT.exists():
    subprocess.run([
        "git", "clone", "--branch", BRANCH,
        "https://github.com/EfeAgaoglu/clothing-shelf-ai.git", str(ROOT)
    ], check=True)
elif not (ROOT / ".git").exists():
    raise RuntimeError("Klasör mevcut fakat Git kaydı yok; dosyalar korunuyor.")

active_branch = subprocess.check_output(
    ["git", "-C", str(ROOT), "branch", "--show-current"], text=True
).strip()
print("Proje:", ROOT, "Dal:", active_branch)
if active_branch != BRANCH:
    raise RuntimeError("Mevcut dal farklı. BRANCH seçimini ve git status'u kontrol edin.")
assert TRANSFER.is_dir(), f"Drive aktarım klasörü bulunamadı: {TRANSFER}"
print("Drive aktarım dosyaları:")
for path in sorted(TRANSFER.iterdir()):
    print(path.name)
```

`Drive already mounted` mesajı hata değildir. `/content` altındaki çalışma dosyaları oturum sonunda kaybolabilir; Drive'da tutulan kayıtları her yeni oturumda yeniden bağlayın. Büyük veriyi Drive üzerinde binlerce küçük dosya olarak okumak yerine arşivi yerel Colab diskine alıp açacağız. [Colab saklama ve dosya aktarımı açıklaması](https://research.google.com/colaboratory/faq.html).

## 3. Paketler, cihaz ve son checkpoint

Colab'ın hazır PyTorch kurulumunu önce koruyun. Aşağıdaki kurulum dosyası Ultralytics'i geçmiş koşularda kullanılan `8.4.144` sürümüne sabitler; rastgele sürüm güncellemesi yapmaz.

```python
import sys
import subprocess

subprocess.run([
    sys.executable, "-m", "pip", "install", "-r",
    str(ROOT / "requirements-portable.txt")
], check=True)
subprocess.run([sys.executable, "-m", "pip", "check"], check=True)

import torch
import ultralytics
import yaml
from PIL import Image
from ultralytics import YOLO

DEVICE = 0 if torch.cuda.is_available() else "cpu"
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)
print("CUDA kullanılabilir:", torch.cuda.is_available())
print("GPU'lar:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("Tahmin cihazı:", DEVICE)

RUN_NEW = TRANSFER / "trained_runs/mixed_df2_fp_2epoch_20260923_110902"
MODEL = RUN_NEW / "weights/best.pt"
NAMES = ["top", "outwear", "sleeveless_top", "shorts", "trousers", "skirt", "dress"]
assert MODEL.is_file(), f"Son checkpoint bulunamadı: {MODEL}"
model = YOLO(str(MODEL))
assert model.task == "segment", "Checkpoint segmentasyon modeli değil."
assert model.names == dict(enumerate(NAMES)), "Checkpoint sınıf sırası farklı."
print("Checkpoint:", MODEL, "Boyut (MB):", round(MODEL.stat().st_size / 1_000_000, 1))
print("Sınıflar:", model.names)
del model
```

Önceki çalışan Colab ortamı Python 3.13.15 / PyTorch 2.11.0+cu128 / Tesla T4 idi; yeni oturum aynı donanımı veya paketleri garanti etmez. `pip check` ya da import hata verirse gerçek çıktıyı inceleyin. Yalnız önceki `ipython requires jedi` hatası tekrar ederse `python -m pip install jedi` ile eksik bağımlılığı tamamlayın. GPU seçili görünse bile `torch.cuda.is_available()` sonucunu doğrulayın. Windows RX 6700 XT için bu CUDA yolu kullanılmaz; [Windows CPU rehberini](WINDOWS_CONTINUE.md) izleyin.

## 4. Eğitim olmadan görsel karşılaştırması

Bu işlem dataset ZIP'lerini gerektirmez. Orijinal JPG/PNG/AVIF görsellerini Drive'daki `shelf_eval_images` klasörüne koyun. Önceki maskeli çıktıları girdi olarak kullanmayın. AVIF, Pillow tarafından bu ortamda açılamıyorsa aynı orijinal görselin JPG/PNG kopyasını ekleyin; asıl dosyayı silmeyin.

Aşağıdaki hücre iki Colab checkpoint'ini aynı görseller, aynı çözünürlük ve aynı eşikle karşılaştırır. Görseller yoksa repo'daki `test_images/shelf_test.jpg` kullanılır. Maske çıktıları ve yan yana karşılaştırmalar doğrudan yeni bir Drive klasörüne yazılır.

```python
from datetime import datetime, timezone
from PIL import Image, ImageOps, ImageDraw
import cv2

models = {
    "fashionpedia_colab": TRANSFER / "trained_runs/finetune_20260923_093619/weights/best.pt",
    "karma_2epoch": MODEL,
}
for checkpoint in models.values():
    assert checkpoint.is_file(), f"Checkpoint bulunamadı: {checkpoint}"

image_dir = TRANSFER / "shelf_eval_images"
suffixes = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".tif", ".tiff"}
images = sorted(p for p in image_dir.glob("*") if p.is_file() and p.suffix.lower() in suffixes)
if not images:
    images = [ROOT / "test_images/shelf_test.jpg"]
assert all(p.is_file() for p in images), "Test görseli bulunamadı."

loaded = {label: YOLO(str(path)) for label, path in models.items()}
for candidate in loaded.values():
    assert candidate.task == "segment" and candidate.names == dict(enumerate(NAMES))

stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
output = TRANSFER / f"shelf_comparison_{stamp}"
output.mkdir(exist_ok=False)
summary = []

for image_path in images:
    with Image.open(image_path) as original:
        rgb = ImageOps.exif_transpose(original).convert("RGB")
    panels = []
    tag = f"{image_path.stem}_{image_path.suffix.lower().lstrip('.')}"
    for label, candidate in loaded.items():
        result = candidate.predict(
            source=rgb, device=DEVICE, conf=0.05, imgsz=1024,
            iou=0.50, max_det=300, save=False
        )[0]
        plotted = result.plot()
        target = output / f"{tag}__{label}.jpg"
        assert cv2.imwrite(str(target), plotted), f"Görsel kaydedilemedi: {target}"
        masks = 0 if result.masks is None else len(result.masks.data)
        lines = [f"{image_path.name} / {label}: {len(result.boxes)} tespit, {masks} maske"]
        for box in result.boxes:
            lines.append(f"  {result.names[int(box.cls[0])]}: {float(box.conf[0]):.2%}")
        print("\n".join(lines))
        summary.extend(lines)
        panel = Image.fromarray(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB))
        panel = panel.resize((640, max(1, round(panel.height * 640 / panel.width))))
        panels.append((label, panel))
    canvas = Image.new("RGB", (640 * len(panels), max(p.height for _, p in panels) + 30), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, panel) in enumerate(panels):
        draw.text((index * 640 + 8, 8), label, fill="black")
        canvas.paste(panel, (index * 640, 30))
    canvas.save(output / f"{tag}__comparison.jpg", quality=95)

(output / "tespit_ozeti.txt").write_text("\n".join(summary), encoding="utf-8")
print("Tüm çıktılar:", output)
print("Eğitim başlatılmadı.")
```

`conf=0.05` düşük skorlu tahminleri de gösteren teşhis eşiğidir. Tespit sayısının veya confidence'ın artması doğruluğun arttığını göstermez. Etiketsiz raf görüntülerinde bu karşılaştırma görseldir; mAP ölçümü değildir. Son karma model raf örneklerinde henüz güvenilir instance ayrımı göstermedi; daha yeni olması, raf görevinde daha iyi olduğu anlamına gelmez.

## 5. Yalnız eğitim hazırlığı için: hazır datasetleri geri al

Dataseti ham kaynaktan dönüştürmeyin, yeniden subset seçmeyin. Taşınan iki hazır subset'i kullanın. Doğrulanmış ZIP SHA-256 değerleri:

```text
Fashionpedia: b2b4179c1a135aefa21e20efc4defe16b2adf810551d8d1a45796c1b0ff44add
DeepFashion2: 766710e3de61e5f094ad261a64674cfaea395001089c836fee0857eabb95968a
```

Bu hücre, mevcut dataset klasörünü korur. Eksik dataset için arşivi Drive'dan yerel `transfer/` klasörüne kopyalar, hash ve ZIP CRC kontrolünden geçirir, yalnız `data/<dataset>/` içeriğini proje köküne çıkarır. Mevcut arşiv farklıysa üzerine yazmaz. ZIP'leri checksum dosyalarıyla birlikte Drive'daki ana aktarım klasöründe tutun. Büyük arşivlerin doğrulanması zaman alabilir; bu eğitim veya benchmark değildir.

```python
import hashlib
import shutil
from pathlib import PurePosixPath
from zipfile import ZipFile

FP_HASH = "b2b4179c1a135aefa21e20efc4defe16b2adf810551d8d1a45796c1b0ff44add"
DF2_HASH = "766710e3de61e5f094ad261a64674cfaea395001089c836fee0857eabb95968a"

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def restore_dataset(source, dataset, expected_hash):
    destination = ROOT / "data" / dataset
    if destination.exists():
        print("Mevcut dataset korunuyor; aşağıda ayrıca doğrulanacak:", destination)
        return
    assert source.is_file(), f"Drive arşivi bulunamadı: {source}"
    sidecar = source.with_suffix(".zip.sha256")
    if sidecar.is_file():
        recorded = sidecar.read_text(encoding="utf-8-sig").split()
        assert recorded and recorded[0].lower() == expected_hash, "Checksum kaydı farklı."
    local = ROOT / "transfer" / source.name
    local.parent.mkdir(parents=True, exist_ok=True)
    if not local.exists():
        with source.open("rb") as src, local.open("xb") as dst:
            shutil.copyfileobj(src, dst)
    actual = sha256(local)
    assert actual == expected_hash, f"ZIP hash'i farklı: {local}"
    with ZipFile(local) as archive:
        entries = []
        for entry in archive.infolist():
            parts = PurePosixPath(entry.filename).parts
            if parts[:2] != ("data", dataset):
                continue
            assert ".." not in parts and "\\" not in entry.filename, "Geçersiz ZIP yolu."
            entries.append(entry)
        assert entries, f"ZIP içinde data/{dataset}/ bulunamadı."
        assert archive.testzip() is None, "ZIP CRC kontrolü başarısız."
        archive.extractall(ROOT, members=entries)
    print("Hash ve ZIP kontrolü geçti; dataset açıldı:", destination)

restore_dataset(
    TRANSFER / "fashionpedia_balanced_10k_windows.zip",
    "fashionpedia_balanced_10k", FP_HASH
)
restore_dataset(
    TRANSFER / "deepfashion2_7class_10k2k.zip",
    "deepfashion2_yolo_7class", DF2_HASH
)

for config in ("fashionpedia_balanced_10k.yaml", "deepfashion2_7class.yaml"):
    subprocess.run([
        sys.executable, str(ROOT / "scripts/check_shelf_dataset.py"),
        "--data", str(ROOT / config)
    ], check=True)
```

Beklenen sonuçlar: Fashionpedia **10.000 / 1.143**, DeepFashion2 **10.000 / 2.000** train/val; `ready: true`, `errors: []`, `warnings: []`. Tam kontrol her görüntüyü ve etiketi inceler. Beklenen sayılar veya kontrol sonucu farklıysa eğitime geçmeyin. Adında ` 2` bulunan kopya YAML dosyalarını kullanmayın.

## 6. Birleşik 20k düzenini geri kur ve dry-run yap

İki subset ayrı ayrı doğrulandıktan sonra bu hücreyi çalıştırın. Yeni subset seçimi, sınıf dönüşümü veya maske üretimi yapmaz; hazır görüntü ve etiketleri `df2_` / `fp_` önekleriyle birleştirir. Yerel Colab diskinde hard link kullanır; kaynak dosyalar değişmez ve görüntüler ikinci kez yer kaplamaz. Hard link desteği olmayan bir ortamda gerçek hatayı inceleyin; bu hücre Windows için değildir. Mevcut birleşik dataset varsa korunup yeniden kontrol edilir.

```python
mixed = ROOT / "data/mixed_df2_fashionpedia_20k"
suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
sources = [
    ("df2", ROOT / "data/deepfashion2_yolo_7class", {"train": 10000, "val": 2000}),
    ("fp", ROOT / "data/fashionpedia_balanced_10k", {"train": 10000, "val": 1143}),
]

if not mixed.exists():
    plans = []
    for prefix, source, counts in sources:
        for split, expected in counts.items():
            source_images = sorted(
                p for p in (source / "images" / split).glob("*")
                if p.is_file() and p.suffix.lower() in suffixes
            )
            assert len(source_images) == expected, f"{prefix}/{split}: görüntü sayısı farklı."
            assert len({p.stem for p in source_images}) == expected, "Tekrarlanan görüntü adı."
            labels = source / "labels" / split
            assert {p.stem for p in labels.glob("*.txt")} == {p.stem for p in source_images}
            for image in source_images:
                label = labels / f"{image.stem}.txt"
                assert label.is_file(), f"Etiket bulunamadı: {label}"
                plans.extend([
                    (image, mixed / "images" / split / f"{prefix}_{image.name}"),
                    (label, mixed / "labels" / split / f"{prefix}_{label.name}"),
                ])
    for source, destination in plans:
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)
else:
    print("Mevcut birleşik dataset korunuyor:", mixed)

sys.path.insert(0, str(ROOT / "scripts"))
from shelf_utils import read_config, validate_dataset

config = {
    "path": str(mixed.resolve()), "train": "images/train", "val": "images/val",
    "names": dict(enumerate(NAMES))
}
DATA_YAML = ROOT / "runs/shelf_7class/configs/colab_mixed_df2_fashionpedia_20k.yaml"
if DATA_YAML.exists():
    assert read_config(DATA_YAML) == {
        "path": str(mixed.resolve()), "names": dict(enumerate(NAMES)),
        "train": str((mixed / "images/train").resolve()),
        "val": str((mixed / "images/val").resolve()),
    }, "Mevcut YAML farklı; üzerine yazılmadı."
else:
    DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    with DATA_YAML.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

report = validate_dataset(read_config(DATA_YAML))
print(report)
assert report["ready"] and not report["errors"] and not report["warnings"]
assert report["summary"]["train"]["images"] == 20000
assert report["summary"]["val"]["images"] == 3143

assert torch.cuda.is_available(), "GPU eğitim hazırlığı için Colab CUDA cihazı gerekli."
subprocess.run([
    sys.executable, str(ROOT / "scripts/train_shelf.py"),
    "--data", str(DATA_YAML), "--model", str(MODEL),
    "--device", "0", "--batch", "1", "--imgsz", "512",
    "--epochs", "2", "--dry-run"
], check=True)
print("Birleşik dataset ve checkpoint doğrulandı. Eğitim başlatılmadı.")
```

Eğitim hazırlığı için 4. bölümü çalıştırmak zorunlu değildir; 2, 3, 5 ve 6. bölümlerin hücreleri yeterlidir.

Dry-run'daki `batch=1`, `imgsz=512`, `epochs=2` değerleri önceki karma koşunun ana ayarlarıyla aynıdır. Bu dry-run hız ölçümü veya bütün eski ayarların birebir tekrarlandığının doğrulaması değildir. Önceki karma koşu Fashionpedia Colab checkpoint'inden başlamıştı; bu örnek ek eğitim hazırlığı için son karma checkpoint'i seçer. Gerçek eğitim öncesinde `args.yaml`, GPU belleği ve deney hedefi dikkate alınarak ayarlar seçilir. `train_shelf.py` yalnız `--execute` verildiğinde eğitim başlatır. Kullanıcının açık eğitim onayı olmadan `--execute` kullanmayın.

## 7. Ek fine-tuning, gerçek resume ve yeni kayıtların saklanması

- Tamamlanmış 2 epoch'tan sonra ek öğrenme: `best.pt` başlangıç ağırlıkları olarak kullanılır. Modelin öğrendiği ağırlıklar korunur; `train_shelf.py` yeni optimizer ve eğitim takvimiyle `resume=False` çalışır. Yeni koşuya `epochs=2` verilmesi iki **ek** epoch demektir; eski epoch sayaçları sürdürülmez.
- Yarım kalmış koşuyu aynı durumdan sürdürme: eğitim durumunu içeren checkpoint ve `resume=True` gerekir. Genellikle `last.pt` kullanılır; yalnız dosya adının `last.pt` olması yeterli değildir. Biten Ultralytics koşusunda optimizer checkpoint'ten çıkarılmış olabilir. Resume uygunluğunu ve yeni oturumdaki dataset yollarını önce doğrulayın. Mevcut `train_shelf.py` bir resume seçeneği sunmaz. [Ultralytics resume açıklaması](https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings).
- `args.yaml` eğitim ayarlarını kaydeder; model ağırlıklarını veya optimizer durumunu taşımaz. `results.csv` metrik kaydıdır; eğitimi sürdürmek için tek başına yeterli değildir.
- Yeni eğitim onaylanıp yapıldığında koşu klasörünü (`weights/best.pt`, mevcut `last.pt`, `args.yaml`, `results.csv`) Drive'daki `trained_runs/<yeni-koşu-adı>/` içine kopyalayın. Yeni isim kullanın; önceki kayıtların üzerine yazmayın. Kesinti sırasında gerçek resume gerekiyorsa epoch checkpoint'leri, optimizer durumu temizlenmeden Drive'da ayrıca korunmalıdır; eğitim sonundaki kopya aynı garanti değildir.
- Notebook'u Drive'a kaydedin. GitHub'a kaynak kod, notebook'un çıktıları temizlenmiş kopyası ve küçük deney kayıtlarını ekleyin. Dataset ZIP'lerini, `data/`, `.venv/` ve yeni `.pt` ağırlıklarını `git add .` ile topluca eklemeyin.

## 8. Drive'dan Windows'a dosya alma

Drive web arayüzünde `Drive'ım / clothing-shelf-ai-transfer` klasörünü açın. Karşılaştırma için iki `trained_runs/<koşu-adı>` klasörünü ve `shelf_eval_images` klasörünü indirin. Klasör indirmesi ZIP olarak gelir; içteki `weights/best.pt` düzenini koruyarak proje içindeki `transfer/` klasörüne çıkarın. Tarayıcı indirme konumuna göre oluşan ek dış ZIP klasörünü kontrol edin; model yolunu varsaymayın.

Eğitim veya dataset incelemesi gerekiyorsa hazır Fashionpedia ve DeepFashion2 ZIP'lerini checksum dosyalarıyla birlikte ayrıca indirin. Yalnız görsel karşılaştırması için büyük datasetleri indirmeniz gerekmez. Windows'ta paket kurulumu için [WINDOWS_CONTINUE.md](WINDOWS_CONTINUE.md) dosyasını kullanın.

Modeli `transfer/trained_runs/mixed_df2_fp_2epoch_20260923_110902/weights/best.pt` konumuna yerleştirdiyseniz PowerShell örneği:

```powershell
.\.venv\Scripts\python.exe scripts/predict_shelf.py --source test_images/shelf_test.jpg --model transfer/trained_runs/mixed_df2_fp_2epoch_20260923_110902/weights/best.pt --device cpu --conf 0.05 --imgsz 1024
```

Maskeli çıktı scriptin yazdırdığı `runs/portable_predictions/...` klasörüne kaydedilir. Bu komut eğitim yapmaz. Windows AMD kartını CUDA cihazı gibi seçmeyin.
