# CVPR 2027 Paper Draft

This directory contains an implementation-aligned bilingual paper draft for the current DINO Dense-Teacher / Sparse-Student branch.

## Files

- `main.tex`: English CVPR-style submission draft.
- `main_zh.md`: synchronized Chinese draft for internal review.
- `references.bib`: BibTeX entries used by the English draft.

`SAPT` is a provisional method name. Check for naming conflicts before submission.

## Compile

The source uses the official CVPR style when `cvpr.sty` is present in this directory. Without it, the document falls back to a two-column `article` layout for local drafting.

```bash
cd paper/cvpr2027
latexmk -pdf main.tex
```

For an official submission build, copy the CVPR-provided `cvpr.sty` and `ieeenat_fullname.bst` into this directory and compile again. Do not submit the fallback layout.

## Draft Status

- Architecture text matches the current `DinoSharedViT`, `DinoTSStage1Adapt`, `DinoTSDense`, `DinoTSDenseEMA`, and `DinoTSSparse` implementations.
- Current concrete configuration: DINOv3 ViT-B/16, `480×640`, `R=3`, `N=1200`, and final `K=600`.
- Current checked-in training configs target MFNet. FMB, SemanticRT, PST900, LLVIP, and KAIST are planned protocol items, not completed experiments.
- No accepted logs or benchmark numbers were found. Every empirical value remains `TBD`.
- Unlabeled semantic pseudo-labels are deliberately excluded.
- Open-vocabulary and universal raw-sensor claims are deliberately excluded.

## Submission-Critical Checklist

- [ ] Resolve the provisional title and acronym after a collision search.
- [ ] Replace the architecture placeholder with a publication-quality vector figure.
- [ ] Build a deduplicated Stage-1 paired dataset; SemanticRT overlaps with LLVIP/OSU/INO sources.
- [ ] Run RGB-only, Thermal-only, dense dual-modal, Anchor-only, and full sparse baselines.
- [ ] Evaluate clean, 13 corruption types across fixed severities, RGB-missing, Thermal-missing, and real adverse subsets.
- [ ] Hold out corruption families, not merely random seeds, for unseen-corruption evaluation.
- [ ] Measure hard-gather FLOPs, synchronized latency, throughput, and peak memory on fixed hardware.
- [ ] Complete budget, fusion-depth, adapter, loss, and training-stage ablations.
- [ ] Save configs, checkpoints, seeds, logs, and environment metadata for every table entry.
- [ ] Recheck all 2026 arXiv citations for updated titles, author lists, and conference publication records.
- [ ] Remove all `TBD`, `TODO`, draft-status text, and red placeholders before submission.
- [ ] Verify anonymous CVPR formatting, page limit, supplementary structure, and reproducibility checklist.

## Claims That Require Evidence

The final paper should only make the following claims if the corresponding tests support them:

1. Anchor preservation improves dense segmentation relative to unconstrained or direct fusion at matched compute.
2. Utility Top-K outperforms random Extra selection at the same `K`.
3. Dense robust distillation improves corrupted and missing-modality performance without unacceptable clean degradation.
4. Hard gathering reduces end-to-end latency, not only theoretical attention FLOPs.
5. The method transfers beyond MFNet when trained and evaluated with dataset-specific heads.

Shared DINO weights, separate modality stems, and modality adapters are not sufficient novelty because SpectraDINO and related work already cover closely related adaptation designs.
