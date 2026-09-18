# Frozen single-stage validation on GenEval-553

This directory contains the direct PyTorch Phase-2 comparison between fixed PSP
and the schedule selected in `../single_stage_calibration/FROZEN_SCHEDULE.json`.
The schedule is copied with a matching SHA-256 checksum and is never edited here.

Both methods use SD1.5 fp16, DDIM with 64 steps and `eta=0`, guidance 7.5,
ImageReward-only online pruning/final selection, and a fresh candidate seed base
of `20260919`. HPS v2.1 and official GenEval score only final winner images.

