# Next model comparison: sequence context and efficient video encoders

Research decision, 7 November 2026. This is an experiment proposal, not a new
trained model or a claim of improved accuracy. The installed default remains
`scout-reviewed-precision@0.3.0`.

The objective remains maximum play precision with at least 95% play recall,
including per-game checks, useful boundary accuracy, CPU throughput of 5–10x and
GPU throughput around 15x. The current repeated validation already reaches about
92% precision and 98% recall; full-game native measurements reached 6.59x CPU and
15.86x GPU. These games are not a fresh blind test. See
[the existing study](model-improvement.md).

## Recommendation

A custom temporal head is worth comparing. Training an entire visual transformer
from scratch is a weaker first investment with nine reviewed games: frames from
the same game do not provide independent examples of new teams, cameras or fields.
This is an engineering assessment, not an experimental result.

| Candidate | Why compare it | Main limitation |
| --- | --- | --- |
| Current numeric HGB head | Established baseline with identical cached inputs | Fixed temporal feature recipe |
| Small temporal convolutional head | Learn activity transitions and suppress short false proposals | Adds a new learned runtime to validate |
| Small temporal transformer over cached features | Learn which surrounding moments matter | Does not remove the cost of person detection |
| Pretrained compact video encoder, initially frozen | Test scene/motion evidence independent of person boxes | Full decode/encoder cost may exceed the CPU budget |

Suggested first comparison: a small dilated temporal convolutional head versus a
two-layer, 64-dimensional temporal transformer, using the same per-time-step
numeric features. These sizes are proposed starting points, not optimized choices.
Keep the detector, frame sampling and interval evaluation fixed to isolate the
effect of the head. Such a head could improve precision; it cannot be assumed to
make the complete pipeline substantially faster when visual inference dominates.

## Literature informing the comparison

- [MS-TCN, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html)
  uses temporal convolutions and a smoothing loss for temporal action segmentation.
  Its treatment of fragmented predictions motivates a compact temporal baseline.
- [MoViNets, CVPR 2021](https://arxiv.org/abs/2103.11511) targets efficient streaming
  video recognition. It motivates a compact video alternative; it does not prove
  football play segmentation accuracy or throughput on this machine.
- [VideoMAE, NeurIPS 2022](https://arxiv.org/abs/2203.12602) studies self-supervised
  video pretraining. A pretrained encoder could be a research reference or teacher.
  Efficient masked pretraining does not establish fast full-video CPU inference.

## Evaluation before adoption

1. Keep footage, existing Scout state and the approved model immutable. Put new
   training outputs, caches and environments in an isolated experiment directory.
2. Select architecture and thresholds with game-grouped folds inside the five
   development games. Fit normalization only on each training fold; preserve game
   boundaries in sequence windows. Do not select settings from validation labels.
3. Freeze the complete pipeline, then report all four existing validation games
   together and separately. Explicitly label this repeated validation. New games
   from independent recording sessions are needed for a fresh blind confirmation.
4. Report one-to-one play precision/recall, temporal overlap and start/end errors,
   plus misses and false detections. Export pre/post padding is a viewing setting;
   it must not inflate model quality metrics or modify the evaluation GT.
5. Measure startup, decoding, feature extraction, head inference and review writing
   end to end on CPU and GPU. Report export encoding separately. Any lower
   resolution, quantization or cascade is a separate controlled comparison.
6. Replace the default only after the precision improvement, minimum recall and
   runtime targets hold. Keep the current model as a rollback option.

No new weights have been downloaded or trained for this proposal.
