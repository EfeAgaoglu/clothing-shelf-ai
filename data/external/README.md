# External dataset inventory

## Ready

### aRTF Clothes 1.0.0

- Official source: https://zenodo.org/records/10875542
- License: CC BY 4.0
- Verified download: `aRTFClothes-resized-paper-splits.zip`
- Converted config: `../../internet_shelf.yaml`
- Converted counts: train 251, val 63, test 580
- Covered project classes: `top`, `shorts`
- Limitation: household scenes with mostly spread/deformed garments, not dense retail shelf stacks

## Selected large dataset: Fashionpedia

- Official project: https://fashionpedia.github.io/home/
- Official download host: CVDF/S3
- Academic dataset: 48,825 images with exhaustive instance segmentation
- Taxonomy: 27 main apparel categories, 19 apparel parts, 294 attributes
- Annotation and ontology license: CC BY 4.0; image use remains subject to each image source's terms
- Training image archive: about 3.34 GB (`train2020.zip`)
- Validation/test image archive: about 236 MB
- Train annotation JSON: about 542 MB; validation annotation JSON: 14,533,475 bytes
- Local mapping script: `../../scripts/convert_fashionpedia_to_yolo.py`
- Converted config: `../../fashionpedia_7class.yaml`
- Converted counts: train 44,937 images / 86,093 instances; val 1,143 images / 2,016 instances
- Covered project classes: all seven
- Full image/label integrity validation: passed

Project mapping:

- `shirt, blouse`, `top, t-shirt, sweatshirt`, `sweater` -> `top`
- `cardigan`, `jacket`, `coat`, `cape` -> `outwear`
- `vest` -> `sleeveless_top`
- `shorts` -> `shorts`
- `pants` -> `trousers`
- `skirt` -> `skirt`
- `dress` -> `dress`
- `jumpsuit`, garment parts and accessories are excluded

Fashionpedia improves category diversity and mask volume, but most images still depict garments on people. It is not a substitute for shelf-domain validation data.

## Awaiting authenticated export

### Roboflow shirts v10

- Source: https://universe.roboflow.com/noah-kuntner-umodm/shirts-tc9pn/dataset/10
- License shown by publisher: CC BY 4.0
- Task: object detection, 495 generated images in v10
- Relevant labels include `folded shirt` and `folded pants`
- Download requires signing in to Roboflow. No files from this source have been downloaded.
- Bounding boxes cannot be used directly as YOLO segmentation masks. After export, relevant original images should be isolated, masks generated, and every mask visually reviewed.
