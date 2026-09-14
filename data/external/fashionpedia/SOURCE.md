# Fashionpedia

- Official project: https://fashionpedia.github.io/home/
- Official repository: https://github.com/cvdfoundation/fashionpedia
- Dataset size: 48,825 images
- Annotation format: COCO instance segmentation with localized attributes
- Annotation/ontology license: CC BY 4.0
- Image rights: retained by the original image sources; see the official terms page

## Local source files

| File | Bytes | SHA-256 |
|---|---:|---|
| `train2020.zip` | 3,344,364,592 | `d896792104a4c35efa7859a18cf605fddae779f5c2372e87e08d7b45949244e2` |
| `val_test2020.zip` | 236,499,034 | `44eb1685364b6ed6bda4595fe0dc07a252e9b923112b97b24ff4d440751a2acc` |
| `instances_attributes_train2020.json` | 542,193,045 | `f6af94fb8712bd8170a49c857e842e1227e6806c4f33f32340db0c849133e006` |
| `instances_attributes_val2020.json` | 14,533,475 | `f831f415edad12d52acde8b138fe7ebd1592b57bf0dcd6a3b63ba8cabd208223` |

Official downloads:

- `https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip`
- `https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip`
- `https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json`
- `https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json`

The archive integrity test (`unzip -tq`) passed for both image archives. The `val_test2020.zip` archive stores both labeled validation and unlabeled test images under `test/`; filenames from the validation JSON are used to select the labeled subset.

## Project class mapping

| Fashionpedia category | Project category |
|---|---|
| shirt, blouse | top |
| top, t-shirt, sweatshirt | top |
| sweater | top |
| cardigan | outwear |
| jacket | outwear |
| coat | outwear |
| cape | outwear |
| vest | sleeveless_top |
| shorts | shorts |
| pants | trousers |
| skirt | skirt |
| dress | dress |

Jumpsuits, accessories and garment-part annotations are excluded. Conversion is implemented in `scripts/convert_fashionpedia_to_yolo.py`. If one garment has disconnected polygon parts, the converter joins them using Ultralytics' standard COCO-to-YOLO segment merge so that small visible garment regions are retained.

Converted output: 44,937 train images with 86,093 instances and 1,143 validation images with 2,016 instances. The output uses symbolic links to the extracted source images and contains all seven project classes.
