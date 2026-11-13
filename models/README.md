# Scout decision models

Only project-trained inference models and their matching configurations live here.
No pretrained YOLO/S3D/MobileNet weights, CPU detector exports, footage, GT or training
checkpoints are shipped in this folder.

- `scout-reviewed-precision/0.5.0/classifier.json`: default HGB plus visual-only
  boundary-window scoring, exported as validated numeric JSON.
- `scout-reviewed-precision/0.4.0/classifier.json`: prior HGB classifier, retained
  for rollback and the Person Transformer detector/feature dependency.
  Its learned tree bytes are unchanged from accepted version 0.3.0. Version 0.4.0
  changes packaging to a decision-only pack with a pinned, lazy YOLO dependency.
- `person-transformer/1.0.0/head.onnx`: best tested person-feature Transformer;
  retained as a selectable experimental alternative, not the quality default.
- `catalog.json`: explicit file identities and the default/alternative recipes.

General YOLO26n weights come from the official Ultralytics asset release, with a
fixed SHA-256. They are fetched on first detection and cached in OS application
cache. CPU OpenVINO/ONNX exports are generated once in that cache. First use needs
internet and the reference/export runtime; subsequent use reuses verified files.
The normal reference installer includes that runtime, not pretrained weights.
The experimental inference-only build needs an imported ready-to-run CPU pack.

Classifier inputs depend on the exact detector, sampler, derived features and
postprocessing settings. Copying the JSON alone into an arbitrary older pipeline
does not reproduce the reported quality. See docs/model-inventory.md.
