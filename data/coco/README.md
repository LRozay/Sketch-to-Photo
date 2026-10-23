# COCO 2017 subset for synthetic pairs

Photos and captions come from a COCO 2017 subset of about 2,000 to 5,000 images.
Captions are the text prompts. Each photograph is paired with a Canny or HED edge
map built by `src/prepare_synthetic_edges.py`.

Expected layout after preparation:

```text
data/coco/
  images/          resized 512 x 512 photographs
  prompts/         one caption file per image stem
  canny/           Canny edge maps
  hed/             HED edge maps
  splits/
    train.txt      image stems used for synthetic pairing
    fid_real.txt   held-out photograph stems for FID
```

Large image files are not committed. Run `src/prepare_coco_subset.py` on a machine
that already has the COCO 2017 train images and annotations.
