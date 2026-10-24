# Decisions

Running log of the main choices made while building this: options weighed, what was picked, and the evidence. Numbers are copied from the output files named in each entry, never worked out by hand. "To be tested" means it hadn't been checked yet when the entry was written. Later entries say what happened.

---

## D01: Reading the lesion labels

`lesion` in `FP.csv` is a list stored as text, e.g. `"[1, 1, 1, 0, 0, 0, 0]"`. Options: split on separators, a regular expression, or `ast.literal_eval`. Went with `ast.literal_eval`, since it reads the list as written. Strict checks after: exactly 7 values, each an integer 0 or 1. A bad row raises `ValueError` naming the image. No silent skipping.

Other rules:
- header must be exactly `type,image,grade,lesion,lr`
- image id = filename stem of the `image` field
- split from the prefix: `tr` = `train`, `ts` = `test`, anything else raises

Format checked against the header and first two rows:

```
type,image,grade,lesion,lr
FP,img/tr000001.jpg,0,"[0, 0, 0, 0, 0, 0, 0]",0
FP,img/tr000002.jpg,3,"[1, 1, 1, 0, 0, 0, 0]",1
```

Tests in `tests/test_lesion_parsing.py`: good rows, wrong length, out-of-range value, malformed string, prefix mapping, duplicate ids. Not yet run on the full file at this point; `scripts/audit.py` does that. Expected counts from the dataset paper: 11,118 images (8,893 train, 2,225 test). The audit records the real counts. A clean audit means every row fits. Labels never changed.

## D02: Cropping the black border

Black border around the retina, so most of it needs cropping. Options: fixed greyscale threshold plus bounding box, Otsu, largest connected component of the foreground, fitted circle. Started with the simplest. Greyscale above 10 (strictly) counts as fundus. Box around those pixels, plus a margin of 2% of the longer side, clipped to the image.

Crop not trusted if nothing is above 10, or if the box covers less than 30% of the image. Full image kept and flagged `crop_failed=True`. Settings in `configs/data.yaml`.

First choice, not yet tested on real images. To check by eye using the crop columns in `outputs/audit/images.csv`, plus the `crop_failed` count in `summary.json`. Side note: re-saving one synthetic disc as JPEG moved the crop box 2 px on one edge. One synthetic image only, so not a rate.

Matters because the crop box feeds the vessel-pattern signature (D03), so an unstable crop affects near-duplicate detection too. Crop failures get reported, not hidden.

## D03: Finding near-duplicates (vessel pattern, not pHash)

Options: pHash on the cropped image (64-bit `imagehash`, Hamming distance), a vessel-pattern signature, CNN embeddings (not tried).

pHash tried first and dropped. On the real data it gave 85,272 pairs at Hamming distance 6 or less and one cluster of 7,472 images. The boundary contact sheet (distances 4 to 6) showed different eyes, including a left and a right eye. Fundus photos all share the same round shape, so a whole-image hash mostly measures that shape.

Went with the vessel-pattern signature: high-pass green channel inside the fundus mask and a central ring, box-downsampled, compared by cosine similarity. Computed in `src/drgrade/vessel_signature.py`, compared in `src/drgrade/neardup.py`. Pairs at or above `vsig_threshold` (0.93, in `configs/data.yaml`) are reported. Connected components of those pairs give the groups, and the groups are the unit for the validation split (D08). pHash and its config keys removed.

What the exploratory run showed (final numbers come from `scripts/audit.py` into `outputs/audit/summary.json`):
- all 197 images in exact-duplicate groups (same MD5) scored 1.0 with their copy
- median nearest-neighbour similarity over all images: 0.36
- contact sheets: pairs at or above about 0.93 were the same eye captured twice (repeat shots seconds apart, or re-saved copies). Pairs between 0.80 and 0.93 were different eyes, often low-detail or blurred
- thresholds 0.93 and 0.95 gave 149 and 145 groups, and 50 test images with a copy in train both times. Not sensitive to the exact threshold

Limits. Catches repeat images of the same eye, not other images of the same patient (the other eye, later visits). Patient-level leakage can't be measured without patient ids, so it's a stated limitation. Boundary sheet `outputs/audit/contact_threshold_boundary_pairs.png` still to check against the 0.80 to 0.93 judgement before the threshold is treated as settled. Two quirks from development, to check on real data: a plain disc on black is not a zero signature (the blur carries the rim into the ring, so two discs of the same shape can look alike); a featureless full-frame image gives the zero vector.

## D04: Python 3.10 as the minimum

Options: 3.10 (what the audit VM runs), 3.11 (the original plan), 3.12. Picked 3.10 for stage 1 code and its pins, because the audit runs on a 3.10 machine. Stage 1 code uses nothing 3.11-only (no `tomllib`, `typing.Self` or `ExceptionGroup`).

Stage 1 pins (`numpy==2.2.6`, `pandas==2.2.3`, `Pillow==11.3.0`, `matplotlib==3.10.3`, `pytest==8.3.5`, `ImageHash==4.3.2`, `PyYAML==6.0.2`) installed on CPython 3.10.20, all tests passed. torch and timm pins on 3.10 to be tested at stage 4.

No effect on results. Only changes which interpreter the code is tested on.

## D05: Audit cache, partial runs, output folders

Options: recompute every run, cache keyed on file contents, cache with a coverage check. `outputs/audit/images.csv` is the cache for the per-image pass. Reused only with `--reuse-cache` and only when it covers every audited image id. A `--limit N` run writes to `outputs/audit/limit_<N>/` and never touches the cache, so a smoke test can't overwrite a full run.

Tested in `tests/test_audit_smoke.py` (second run with the cache gives identical duplicate results; a limited run writes to its own subfolder).

Catch: the per-image cache doesn't check crop settings or file contents. After changing `configs/data.yaml` or the images, run without `--reuse-cache`. The vessel-signature cache (`vessel_signatures.npz`) is reused only with `--reuse-cache` and only when image ids and signature settings match exactly.

## D06: Which pairs go on the contact sheets

Options: random sample, closest pairs, pairs spread across the distance range. Cross-split sheet shows the 12 closest cross-split pairs (sorted by distance, then id). Boundary sheet shows up to 12 pairs spread evenly from (threshold minus 2) to the threshold. Both deterministic. Method choice, no numbers needed.

The cross-split sheet shows the most convincing pairs, not a random sample. To judge how many pairs are true duplicates, use `near_duplicate_pairs.csv`, not the sheet.

## D07: Repeat captures with different grades

Options: drop groups with mixed grades, relabel to a consensus grade, or report and leave labels alone. Report only. Labels never changed or dropped. Groups with more than one grade go to `outputs/audit/grade_disagreement_groups.csv`, pairs with different grades are drawn on `outputs/audit/contact_grade_disagreement_pairs.png`, counts in `summary.json` (`duplicates.near`).

Counts to be read from `outputs/audit/summary.json` after the real audit (`n_groups_with_grade_disagreement`, `pair_grade_difference_counts`). The synthetic audit test (`tests/test_audit_smoke.py`) checks that a copy with a different grade gets listed.

Two grades for one eye means label noise or a grading-process difference, but not which grade is right. Copies across the official split are kept, so test still contains them; the sensitivity analysis (D22) is where they come out. Within-split groups stay together in the validation split.

Real-data result (`outputs/audit/summary.json`, `contact_grade_disagreement_pairs.png`): 5 groups with mixed grades. Checked by eye: 4 are genuinely the same eye, each graded one step apart (0 vs 1 twice, 1 vs 2 twice, one of the 1 vs 2 pairs spanning train and test). The fifth, tr004850 (grade 4) with ts001140 (grade 0) at similarity 0.930, is a false match: two different low-detail eyes right at the threshold. Kept in the grouping (a false pair only keeps two extra images on the same side of the split), but left out when reporting grade disagreement and train/test copies. So 50, not 51, test images have a genuine copy in train.

## D08: How the validation split is made

Options: random stratified split of images, `StratifiedGroupKFold` grouped by near-duplicate cluster, or `StratifiedKFold` over the groups with each group labelled by its most common grade. Went with the last: 7 folds, shuffled, seed 42, fold 0 as validation. Groups from `outputs/audit/dup_clusters.csv`; images in no group are their own group.

Why: on a synthetic 2,000-image set, `StratifiedGroupKFold` left a train/validation gap of 3.3 percentage points for grades 2 and 3 under scikit-learn 1.6.1 and 1.7.2, and gave a different split under 1.9.1. Group-level `StratifiedKFold` had a largest gap of 0.18 points and the same split under all three versions. Real-data gaps in `splits/split_v1.summary.json`.

Result: repeat captures of one eye never sit on both sides of train/validation, so validation scores aren't inflated by them. Repeats across the official train/test boundary can't be fixed without changing the official test set, so they're reported instead (D03).

## D09: Resize to 512 px with area averaging, JPEG quality 95

Options: 384 px only, 512 px with bilinear or bicubic, 512 px with area (BOX), PNG instead of JPEG. Padded square resized to `target_size` (512) with Pillow `Image.Resampling.BOX` (averages source pixels over each output pixel). Saved as JPEG at `jpeg_quality` 95 with chroma subsampling off (4:4:4, Pillow `subsampling=0`). Pillow's default 4:2:0 halves colour resolution, and the smallest lesions (microaneurysms, small haemorrhages) are small red dots where colour is the signal. Output in `data/processed/s512/`. The 384 px baseline (E1) resizes from these 512 px files in the data pipeline.

Prior choice, to be tested. 512 because small lesions like microaneurysms stay visible while the model still fits a T4; also the planned value. The 512 vs 384 effect is only testable as far as E1 (384) and E2 (512) differ. No number yet.

Stored JPEGs are a lossy copy. Quality measures (D10) are computed before encoding, so they describe the resized pixels, not the file. Quality 95 with full chroma keeps the extra loss small; files are bigger than with default subsampling.

## D10: Quality measures and the fundus mask

Options: measures on the whole image (border included), on the fundus mask, on the mask with a one-pixel erosion, or a separate mask threshold in the config. Three measures on the resized image (before Ben Graham, D11), inside the fundus mask. Mask = greyscale strictly above `crop_threshold`, same as the crop, so no new config key.
- `brightness`: mean of HSV V (max of R, G, B), 0 to 255
- `contrast`: RMS contrast, std of greyscale inside the mask, 0 to 255
- `sharpness`: variance of a 3x3 Laplacian, kernel [[0,1,0],[1,-4,1],[0,1,0]], on greyscale, only where the full 3x3 neighbourhood is inside the mask. The erosion drops the hard black-to-fundus edge, which would otherwise swamp the variance. Image border pixels never counted.

Empty mask gives 0.0 for all three.

Prior choice, to be tested. Synthetic tests (`tests/test_preprocess.py`) check the direction of each: blurred copy less sharp, darkened copy less bright, flat disc zero contrast, flat disc with one hard edge zero sharpness after erosion. No real-data values yet.

These are image-level proxies. Brightness and contrast partly reflect camera and operator, so they're quality indicators, not clinical findings. Sharpness depends on output size, so only comparable at the same `target_size`.

## D11: Ben Graham normalisation off by default

Options: always on, on by default with an ablation, or off with the `ben_graham` flag for an ablation only. Off (`ben_graham: false`). When on: 4 x image minus 4 x its Gaussian blur (sigma `ben_graham_sigma`, 10 px at 512 px), plus 128, clipped to 0 to 255, set to 0 outside the fundus mask. Quality measures still computed before this step (D10).

Prior choice, to be tested. The method strips low-frequency colour and brightness variation, and here colour is clinical (exudates and haemorrhages are coloured), so off. An ablation needs its own training run, and E1 to E3 are the whole compute budget, so not done. Smoke test checks the flag changes the saved pixels and leaves the manifest measures alone.

No reported result uses it. If it's ever used, the run is an ablation and gets compared against the same model without it.

## D12: Resumable preprocessing, settings guard, limit runs

Options: one long run (fails under the VM's 170 s command limit), a resume flag, automatic resume with a progress file checked against the outputs, a settings check on resume.

`scripts/preprocess.py` saves progress after each batch of 400 images to `manifest_partial.csv`. Rerunning the same command skips images with a row there and an existing JPEG, so no `--resume` flag. `--max-seconds` stops after the first batch that ends at or past the limit and exits with status 3; just run it again. Output-changing settings (`target_size`, `jpeg_quality`, crop settings, `ben_graham`, `ben_graham_sigma`) are stored in `preprocess_info.json`. Resuming with different settings stops with status 2 and asks for `--overwrite`. JPEGs go to a temp name, then `os.replace`. A `--limit N` run writes to `s<target_size>_limit<N>/`, with no zip, full manifest or figure. Default 4 workers (the VM has 4 cores).

`tests/test_preprocess_smoke.py`: interrupted run plus resume gives a byte-identical `manifest.csv`, a missing JPEG is regenerated, changed settings refuse to resume, no `.tmp` files left. Resumed floats are read with round-trip precision, which the byte-identical check needed.

No effect on results. The guard covers the listed keys only; changing anything else, like the images on disk, still needs `--overwrite`.

## D13: Grey camera backgrounds, corner masking, camera as a confounder

Options: keep the fixed threshold of 10, raise it for every image, read the background from the whole image edge (tried first, see below), or read it from the four corners only, blacken edge-connected pixels near that level, and mask the corners after resizing.

The first preprocessing run showed fundus discs inside a grey frame. Those photos use dark grey (about 16 to 20) instead of black, so the threshold of 10 left the crop as the whole image.

First fix read the level from the whole image edge. Wrong: when the disc touches the frame edge, the "edge level" is fundus (45 to 108), and the rule erased parts of the fundus (e.g. tr002199, tr000400). Five photos with a white border (all grade 4) were blacked out entirely. Corners sit outside the disc even when it touches the edges, so switched to corners. Third check on the same problem images (`outputs/preprocess/check_v3.png`): intact and cropped. Slightly lighter grey halos left in some corners, which the circular mask removes.

Final version: `border_level` = median of four small corner patches. When they agree (spread at most `bg_corner_spread`, 12) on a level of at least `bg_detect_min` (8), `blacken_background` blacks out pixels within `bg_tolerance` (10) of that level that can be reached in a straight line from the image edge. If cropping the cleaned image fails, the original is used. After resizing, `apply_circle_mask` blacks out everything outside the inscribed circle (`circle_mask: true`). Same cleaning inside `crop_fundus`, so audit, vessel signature and preprocessing agree. Each original's corner level stored in the manifest as `border_level`.

Synthetic tests (`tests/test_background.py`): grey and white borders, discs touching edges, uneven corners, a dark spot inside the disc darker than the border.

Important: non-black backgrounds aren't spread evenly across grades. The grey-border camera was used mainly on eyes with DR, and all five white-border images are grade 4, so camera traits are a possible shortcut. Cleaning and masking remove the obvious cues, not colour response or optics. So performance is reported by `border_level` group in the error analysis, and the report names the camera as a confounder. Group counts from `data/processed/s512/manifest.csv`.

---

## D14: Augmentation (torchvision v2)

Library options: torchvision.transforms.v2, albumentations, torchvision v1. Augmentation options: full 360 degree rotation or a limited angle, flips, wide or mild crop range, colour jitter with or without hue and saturation. Picked v2: ships with torchvision (pinned at 0.20.1), so no extra dependency, and handles tensors and PIL directly.

Training: rotation uniform in [-180, 180] with black fill, horizontal and vertical flips at 0.5, random resized crop scale 0.85 to 1.0 and ratio 0.95 to 1.05, brightness and contrast jitter 0.2, gamma in [0.8, 1.2]. No hue or saturation change. Evaluation: resize only if input size isn't the stored 512 px, then ImageNet normalisation. All values in the `aug` section of each run config.

Prior choice, to be tested. Ranges as planned, not tuned. Rotation test (`tests/test_data.py`) checks the black corners of a rotated disc stay black. No accuracy yet.

Full rotation because fundus orientation varies between photos. No hue or saturation change because colour carries the signal (exudates, haemorrhages). Any gain or loss from augmentation is inside the E1 to E3 comparison, not measured separately.

## D15: Class weights (square-root inverse frequency) vs plain cross-entropy

Options: plain cross-entropy (ce), inverse-frequency weights, square-root inverse-frequency weights (wce), oversampling. wce weights = (N / n_c) ** 0.5, normalised to mean 1 over the five classes, from the training subset only (`src/drgrade/losses.py`). E3 uses wce, E2 uses ce, same model and input size, so the pair isolates the loss. Label smoothing off.

Validation loss uses the same criterion as training, so for wce it's the weighted loss. Val loss therefore can't be compared directly between a wce and a ce run. Selection uses QWK, which doesn't depend on the loss (D16).

Prior choice, to be tested. Square root was the planned form, a softer version of full inverse frequency. No validation numbers yet; results go to `runs/<run>/summary.json` after the Colab runs.

Any E2 vs E3 difference mixes the effect of the weights with run-to-run noise, so one seed can't separate them with confidence. The stability run (D34) is where that gets checked, on validation only.

## D16: Model selection rule, fixed before training

Options: validation loss, accuracy, macro-F1, or quadratic weighted kappa (QWK) with a macro-F1 tie-break. Keep the epoch with the highest validation QWK (`best.pt`); ties go to higher validation macro-F1. Early stopping counts epochs without a new best on this rule, patience 5. Rule written into `summary.json`. Nothing selected on test, which `train.py` never loads (`load_run_table` refuses it, D21).

Fixed before any training run. No numbers involved.

QWK treats a two-grade error as worse than a one-grade error, which fits an ordered scale and the cost of missing referable disease. Doesn't guarantee the best referable sensitivity; that's reported at evaluation, not used for selection.

## D17: Mixed precision on CUDA only

Options: mixed precision on CUDA, MPS and CPU; CUDA only; none. `torch.autocast` with float16 and `torch.amp.GradScaler` on CUDA only. On MPS and CPU the scaler is off and the forward pass is float32. Logits cast to float32 before the loss and softmax, so probabilities in `val_predictions.csv` are computed in float64.

Prior choice, to be tested. Picked to fit the planned T4 batch sizes. Effect on accuracy not measured, none reported.

A CUDA run won't be bit-for-bit identical to a CPU or MPS run of the same config. `env.json` records the device, and the README names it.

## D18: Determinism settings, and the CUDA vs CPU caveat

Options: nothing, cuDNN deterministic only, full deterministic algorithms with warnings for unsupported ops, or deterministic algorithms that raise. Went with warnings. `seed_everything` seeds Python, NumPy, torch (CPU, CUDA, MPS) and `PYTHONHASHSEED`, sets `torch.backends.cudnn.benchmark` to False, calls `torch.use_deterministic_algorithms(True, warn_only=True)`, and sets `CUBLAS_WORKSPACE_CONFIG` to `:4096:8` before any CUDA call. Warn-only means an op with no deterministic version warns instead of crashing; the README says such a warning may appear. DataLoader workers seeded from torch's per-worker seed (`src/drgrade/data.py`).

On CPU with no workers, `test_resume_matches_uninterrupted_run` (tests/test_train_smoke.py) shows an interrupted-and-resumed run ends with weights equal, tensor by tensor, to an uninterrupted one. CPU only. Says nothing about CUDA or MPS, which aren't tested here.

Reported numbers come from the device in `env.json`. CUDA and MPS may differ slightly from CPU. Repeat runs on the same CUDA device and seed should match, but not tested.

## D19: Colab clones the private repo with a read-only token

Options: make the repo public now, keep a Drive copy of the code, clone with a token stored as a Colab secret, or write a token into the notebook. Repo stays private until submission. The notebook reads a fine-grained, read-only token (Contents only) from the Colab secret `GITHUB_TOKEN` and clones over HTTPS. Token never printed: error text is cleaned before it's shown. Fresh clone each run instead of `git pull`, so the token isn't left in the remote config. Remote reset to the plain HTTPS URL after the clone. Token never written into the notebook.

Method choice, no numbers. Notebook structure checked with nbformat validation. Clone itself not yet run against GitHub at this point.

No effect on results. Repo has to be public (or the token kept) before anyone can reproduce from the README.

## D20: Checkpoints, resume, smoke test

Options: final weights only; a per-epoch checkpoint with the model only; a checkpoint with optimiser, scheduler, AMP scaler, history and RNG states; resume from the last best model. Went with the full checkpoint.

`last.pt` written every epoch: model, AdamW state, LambdaLR state, AMP scaler state, history, best metrics, early-stopping counter, RNG states (Python, NumPy, torch, CUDA where present, DataLoader shuffle generator) and the run config. `best.pt` written when the selection rule (D16) improves. Writes go via a temp file and `os.replace`. Resume loads `last.pt`, refuses if any setting changed other than `max_epochs` and `num_workers`, and for a finished run only regenerates a missing `val_predictions.csv`. Without `--resume`, a folder that already has `last.pt` is refused, so a finished run can't be overwritten by mistake. `--smoke` trains one epoch, stops, resumes in the same process for epoch two, and fails unless `history.csv` has two rows and the resume started from epoch 1.

Evidence: CPU resume test in `tests/test_train_smoke.py`, plus the smoke test on a 512 px synthetic subset built from the real manifest labels (CPU, 2 cores, not a result). Exact equality only holds on CPU with no workers. With workers, the persistent worker processes don't restart in the same state, so augmentation draws after a resume aren't guaranteed to match. Stated in the README.

So a Colab run that's interrupted and resumed matches an uninterrupted one in model state on CPU with no workers. On Colab with two workers, same training procedure, not the same exact random draws.

## D21: The loader refuses the test subset

Options: a comment in the training script, a check in `train.py` only, or a check inside the shared loader every script goes through. The loader. `load_run_table` in `src/drgrade/data.py` raises ValueError for `test` unless `allow_test=True` is passed. Only evaluation passes it. `train.py` asks for `train` and `val` only. A test in `tests/test_data.py` checks the refusal.

`test_test_subset_is_refused_unless_allowed` passes. No real test data loaded during stage 3 or 4; the synthetic fixtures only include test rows to exercise the refusal.

Test set untouched during training, validation and the smoke test, which backs up single use of the test set.

---

## D22: Test images with a copy in training, and the known false match

Options: drop nothing (headline test metrics only), drop every test image with any near-duplicate pair in official training (false match included), or drop genuine links only, with the known false match removed first. The last.

`scripts/make_leakage_list.py` (logic in `src/drgrade/leakage.py`) builds `splits/test_with_train_copy.csv` (image_id, linked_train_ids, max_similarity, exact). A test image is listed when it connects, through genuine edges only, to an official training image (train or validation subset). Pairs in `splits/false_matches.csv` are removed as edges before forming components. The one entry is tr004850 with ts001140, different eyes at the threshold (D07). Linked ids = every official training image in the component. `max_similarity` and `exact` describe the direct genuine edges from the test image to training; an image linked only through a chain has an empty `max_similarity`.

Prior choice, to be tested on the real audit outputs. The script logs the count; not written here. D07 says 50 test images have a genuine copy in train once the false pair is out, and the log is the check. Synthetic tests in `tests/test_leakage.py`: the false-match case, a chain through a test image, a false edge that would otherwise link a test image, a test-only cluster.

`scripts/evaluate.py` reports the headline test metrics with and without the listed images. The list depends on the vessel-signature threshold (D03). Only finds repeat captures of one eye, so doesn't measure patient-level leakage. File committed since it's ids only.

## D23: Test-set lock with open and closed status

Options: one lock written on the first test prediction and never changed; a lock that refuses every later test prediction without an override; an open lock that lets the plain and TTA test outputs of one checkpoint each be made, then an explicit close. The last.

`scripts/predict.py --subset test` needs `--final`. First complete test output for a run writes `<run>/TEST_LOCK.json` with status `open`, UTC time, SHA-256 of `best.pt`, git commit and dirty flag (none here, as this directory isn't a git repo), the command, the TTA flag, and each output written. While open, the plain and TTA outputs can each be made once without an override. `scripts/predict.py --close-lock --run DIR` sets status `closed` with a UTC time. After that, and for a repeated output while open, a test prediction needs `--force` and `--reason TEXT`. Every forced rerun is appended to `reruns` with time, reason, command and checkpoint hash. A changed `best.pt` hash also needs the override. The lock check runs before the test subset is loaded. An incomplete test prediction (exit status 3) writes no lock.

Prior choice, to be tested. Rules tested in `tests/test_predict.py` (`test_lock_decision_rules`, `test_cli_test_lock_flow`, `test_cli_partial_test_run_writes_no_lock_until_complete`).

The lock records every test output made, so the number of test evaluations is auditable. Won't stop someone determined from copying a file, and can't catch a hand edit to `best.pt` until the hash check on the next run (D38 added more). Every forced rerun in `reruns` must be named in the report.

## D24: Test-time augmentation views and when to use them

Options: no TTA, four flip views, eight views with 90 degree rotations, or TTA chosen on test (rejected, that uses test for a decision). Four views of the normalised batch, averaged in softmax probability: identity, horizontal flip, vertical flip, both flips (the 180 degree rotation). Flips are tensor ops on the normalised batch, so exact. In a TTA file the logit columns are the log of the averaged probabilities; a plain file keeps the raw logits. Decided on validation only, comparing `val.csv` with `val_tta.csv` through `scripts/evaluate.py`. Test results reported plain first, then TTA if chosen.

Prior choice, to be tested on validation. The four views are the symmetry set of horizontal and vertical flips: no interpolation, and two of them are already training augmentations. The plan first said flips and 90 degree rotations. Cut to these four, because 90 degree rotations add four more forward passes per image on a 4-core CPU for little expected gain.

The result depends on the view set. Adding 90 degree rotations later would be a new variant and need its own validation check before any test run.

## D25: Majority-class baseline M0

Options: most common grade in all processed images, in the test set (not allowed, uses test labels), in the training subset only, or a uniform random predictor. Training subset only (`train` rows of the split file, not validation). Ties go to the lower grade (`majority_grade` in `src/drgrade/metrics.py`). `scripts/evaluate.py` scores M0 with the same functions as the models. No probability columns, so referable AUROC is null, with a note saying so. Grade can be passed with `--train-majority-grade`.

Method choice; training grades from the manifest and split file. The value is written to `metrics.json` by the script. `tests/test_metrics.py` checks the tie-break and M0 metrics on a hand example.

M0 is the floor to beat. Referable sensitivity is 0 whenever the majority grade is below 2, and QWK is null when every prediction is the same grade.

## D26: Bootstrap intervals and undefined values

Options: normal-approximation intervals, percentile bootstrap over images with a fixed seed, bias-corrected intervals, no intervals. Percentile bootstrap: 95%, 1,000 resamples with replacement, seed 42, `numpy.random.default_rng`. For accuracy, macro-F1, QWK, referable sensitivity and referable miss rate. A resample where a metric is undefined is left out of that metric's percentiles, and `n_valid` says how many are left. Undefined main metrics (no positives, no grade 4 images, constant prediction) are null with a note, never an error. Macro-F1 and weighted-F1 count an undefined per-class F1 as zero, same as `validation_metrics` in training, so the two agree.

Prior choice, to be tested. Matches scikit-learn on accuracy, macro-F1 and QWK in `tests/test_metrics.py`; intervals checked to be deterministic and to contain the point estimate. No widths quoted here.

The bootstrap treats test images as independent. It ignores repeat captures and patient-level dependence (D03), so intervals are too narrow if that dependence is strong. Report should say so.

## D27: Subgroups by border level and quality quartile

Options: no subgroups; split by camera background (border level below 8 vs 8 or above, the `bg_detect_min` threshold in configs/data.yaml); split by observed grade; quartiles of the three quality measures by rank. Two sensitivity analyses. `metrics.json` reports headline metrics for `border_level` below 8 (black background) and 8 or above (non-black), with n in each, because D13 names the camera background as a confounder. `quality_stratified.csv` reports headline metrics per quartile of brightness, contrast and sharpness, by rank, so each quartile is about the same size. Test metrics without the listed train copies (D22) are in `metrics.json` under `excluding_train_copies`.

Prior choice, to be tested. Threshold 8 = `bg_detect_min` in the data config. Depends on the processed manifest, which isn't on this machine, so no group sizes here.

A gap between groups could be camera, lesion mix or grade mix, and this can't separate them. Groups aren't balanced, and D13 says the non-black photos cluster in some grades. Small subgroups are reported with their n and may be null.

---

## D28: Temperature scaling on validation logits

Options: no recalibration, histogram binning, isotonic regression, Platt scaling, temperature scaling (one scalar T). Temperature scaling: `softmax(logits / T)`, T fitted by minimising NLL on validation logits only (`src/drgrade/calibration.py`, `fit_temperature`). Optimises log T so T stays positive. LBFGS in float64, strong Wolfe line search, at most 100 iterations, starting at T = 1. A test file, if given, is only read to report metrics before and after (`scripts/calibrate.py`). Calibrated CSVs keep the input columns: probabilities recalibrated, logit columns divided by T, `pred_grade` unchanged (checked in code). T saved in `<run>/calibration/calibration.json`.

Method choice. Synthetic tests (`tests/test_calibration.py`): perfectly calibrated data gives T near 1 and ECE near 0, logits scaled by 3 give T near 3 and lower NLL after scaling, argmax unchanged for any T, test file doesn't change T (shuffled test labels). Real T values are in each run's `calibration.json`, not here.

Scaling changes the confidence used for referral, not the predicted grade, so accuracy, QWK and referable miss rate don't move. For a TTA validation file the logits are log averaged probabilities; T then rescales log-probabilities, still a monotone recalibration, and `calibration.json` records `tta: true`.

## D29: ECE with 15 equal-width bins

Options: 10, 15 or 20 bins, equal-mass bins, a kernel estimate. 15 equal-width bins on confidence (largest probability), as planned. Each bin is [lower, upper), last bin also holds 1.0. ECE = bin-count-weighted mean of |accuracy - mean confidence| over non-empty bins (`ece` in `src/drgrade/calibration.py`). NLL = mean of -log p(true grade), probabilities floored at 1e-12. Multi-class Brier = mean over images of the sum over five classes of (p - onehot)^2. Per-bin table in `reliability_bins.csv`.

Method choice. Hand calculation on four images in 10 bins gives ECE 0.4, which the unit test checks (`tests/test_calibration.py`). No real values here.

With 1,268 validation images (`splits/split_v1.summary.json`), the top and bottom bins hold few images, so their accuracy is noisy and ECE has sampling error. That's why the reliability diagrams show the count under each bin.

## D30: Referral rules, fixed before any test use

Options: a referral fraction chosen on test (rejected, test used for a decision), a fraction picked after looking at validation curves (rejected, not declared in advance), an operating point declared in advance on validation and applied as a fixed threshold to test, a probability cut at 0.5 (not used, scale not calibrated). The declared operating point.

Primary: refer the most uncertain 20% of validation cases, score = calibrated max probability. The 20% becomes a confidence threshold on validation (`threshold_for_count`), and that fixed threshold is applied to test (at or below it = referred).

Secondary: the lowest referral fraction, in 1% steps, at which the retained referable miss rate on validation is 5% or below. If nothing up to 50% gets there, the output says "not reached" and no threshold is set. Also converted to a fixed threshold. Sets are nested, so the retained miss rate is computed on exactly the cases the threshold keeps. In `src/drgrade/referral.py` and `scripts/referral.py`.

Written here before `scripts/referral.py` touched any real test predictions. Before that, the pipeline was run once on synthetic validation and test files in a scratch folder outside the project; no real data, no rule set from it. Validation numbers go to `runs/<run>/referral/referral.json`, none quoted here.

The referral fraction on test is whatever the fixed threshold refers, so it can drift from 20% if confidence shifts. Report test results with the actual fraction. 20% is a workload choice; it doesn't show 20% is clinically right.

## D31: Confidence vs entropy, compared on validation

Options: confidence only, entropy only, both with the primary rule already fixed to confidence, or pick whichever wins on validation and use it on test (rejected: a choice made from the numbers after the fact). Primary score stays calibrated max probability (D30). AUROC of each uncertainty score (1 minus confidence, and entropy) reported on validation, for separating wrong predictions and for separating referable misses (true 2 or above, predicted 0 or 1). `referral.json` names the better one for errors (`better_for_errors`). That's a report, not a rule, and doesn't change the operating point.

To be tested. Numbers in `runs/<run>/referral/referral.json`, field `uncertainty_validation`. None quoted here.

If entropy ranks errors better, the report says so and the rule stays on confidence. Switching the score would be a new rule with its own declaration and validation check before any test use.

## D32: Referable threshold matched to E3's validation sensitivity

Options: keep E2's hard decision (pred 2 or above), a 90% sensitivity target for every run, match E3's validation sensitivity by moving E2's threshold on the referable score, or compare E2 and E3 at their own hard decisions only. Matching, when `--match-sensitivity-of` is given.

Referable score s = calibrated p(2) + p(3) + p(4). On validation, pick the highest threshold whose sensitivity reaches the target, where the target is E3's validation sensitivity from its hard predictions (pred 2 or above). No reference run = 90% target. Threshold applied as-is to test (`scripts/referral.py`, `referable_threshold.json`).

Numbers read at the time from `runs/*/eval_val_predictions/metrics.json`. Referable AUROC on validation 0.981 for both E2 and E3, so the two scores rank cases equally well. E2 hard decision: sensitivity 0.843, specificity 0.964. E3: sensitivity 0.919, specificity 0.932. The real question is whether E2 can reach E3's sensitivity by moving its threshold, and at what specificity.

Equal AUROC means any sensitivity gap at a fixed decision comes from the operating point, not ranking. If E2 matches E3's sensitivity at similar specificity, then E2 vs E3 for referral is about the threshold, not one model ranking better. Threshold chosen on validation; the test result is one fixed-threshold outcome.

## D33: Image-quality stress test

Options: no stress test, perturb test (rejected, test is single use), perturb validation with Gaussian blur and darkening at three levels each, additive noise, JPEG re-compression, contrast reduction. Validation, blur and darkening. Blur radius 1, 2 and 4 px at processed size (Pillow `GaussianBlur` on the RGB image, before the evaluation transform). Darkening multiplies RGB by 0.8, 0.6 and 0.4. After blurring, everything outside the inscribed circle is set back to black so the corners stay black (same geometry as `apply_circle_mask`, D13). Baseline ("none") goes through the same code. `best.pt`, no TTA, each level resumes from its own partial file (`scripts/quality_stress.py`, `predict_subset`).

Prior choice, to be tested. Levels aren't calibrated to any measured camera or operator variation, so they show sensitivity to a known change, not to what's actually in the MMRDR photos.

Shows whether confidence drops when quality drops, and whether QWK drops with it. Doesn't measure natural quality variation; that's the quartile breakdown in the error analysis (`errors_<name>/quality_stratified.csv`). Blur radius is in pixels at processed size, so levels mean something different at another `target_size`.

## D34: Extra runs (stability, replica, resolution): validation only

Options: treat the extra runs as more selection candidates, or as stability checks and an ablation only (as planned before they ran). The latter. Selection rule applies to E1, E2 and E3 only (`outputs/compare/val_comparison.md`, which picks E2). Extra runs compared in `outputs/compare_extras/` with `scripts/evaluate.py --compare --no-selection`, so that table makes no selection. E2 stays the selected model.

Validation results (`outputs/compare_extras/val_comparison.csv` and `runs/*/eval_val_predictions/metrics.json`). Three runs of the E2 recipe: E2 on a Colab T4; E5 (seed 43) and E2r (seed 42) on an RTX 6000 Ada with torch 2.8 and 5 data-loader workers.
- QWK 0.929, 0.926, 0.932
- macro-F1 0.754, 0.762, 0.764
- referable sensitivity 0.843, 0.854, 0.878
- grade 3 recall 0.54, 0.62, 0.56
- referable AUROC 0.981, 0.977, 0.981

E3 (weighted loss): QWK 0.924, referable sensitivity 0.919, grade 3 recall 0.83, referable AUROC 0.981. E4 (E2 at 384 px): QWK 0.929, macro-F1 0.743.

What it means:
- The spread across the three E2-recipe runs is the run-to-run noise on validation. QWK gaps of a few thousandths between E1 to E4 sit inside it, so neither class weighting (E2 vs E3) nor 384 vs 512 px (E2 vs E4) reliably changes overall agreement.
- E3's referable sensitivity and grade 3 recall are outside the E2-recipe range, so class weighting's effect on the operating point and on severe NPDR is bigger than noise. Only one E3 run, which the report says.
- E2r isn't a bit-exact reproduction: different GPU, PyTorch version and number of workers (which changes augmentation order). Shows the result holds in another environment, not that it's identical.
- The pod runs record `git.dirty: true`. Cause: an untracked `logs/` folder created by `scripts/pod/run_parallel.sh` in the clone; tracked code was commit `c16e121`, unchanged. `logs/` now in `.gitignore`.

## D35: Decisions locked before any test prediction

Options: pick E3 for its validation sensitivity (rejected, choosing by a metric picked after seeing results), test only the selected model, test the selected model plus one declared secondary, test every run (rejected, turns test into a leaderboard).

Committed before any test prediction:
1. Selected model: E2 (`runs/e2_effb0_512_ce`), by the fixed rule (highest validation QWK among E1 to E3, D16). Extra runs don't change it (D34).
2. Declared secondary on test: E3 (`runs/e3_effb0_512_wce`), to check on unseen data whether class weighting's validation effect on the referable operating point and on grade 3 holds. Reported as a comparison, not the selected model. No other run goes on test.
3. E2's referable sensitivity handled by thresholds fixed on validation (D30 to D32): the confidence-based referral rules and a referable-score threshold matched to E3's validation sensitivity. Thresholds and temperature fitted on validation only, applied unchanged to test.
4. TTA (D24) adopted for E2 only if, on validation, it raises QWK by at least 0.005 (about the run-to-run spread, D34) without raising the referable miss rate. Otherwise plain only on test. Plain test results always first. E3 plain only.
5. Each run's test predictions under its own TEST_LOCK.json, each lock closed once its declared outputs exist.

Validation results in `outputs/compare/`, `outputs/compare_extras/` and `runs/*/eval_val_predictions/` (D34). No test prediction existed when this was committed (checked: no TEST_LOCK.json and no predictions folder in any run).

Any test result for E3, or for E2 with TTA, is a declared, pre-planned comparison. Test set not used to choose a model, threshold, temperature or TTA.

## D36: channels_last layout for CPU inference

Options: default (NCHW) or channels_last on CPU. channels_last for the model and every input batch when predicting on CPU (`src/drgrade/predict.py`). Only changes memory layout, not the maths.

On my local Linux VM (4 cores), a random-weight EfficientNet-B0 batch of 8 at 512 px took 2.69 s default vs 1.57 s channels_last, max logit difference 0.0. Trained E2 on the first 144 validation images: every grade identical, largest logit difference 3.1e-5 (rounding order). Validation predictions made before this were set aside (renamed `*_pre_channels_last*`). Every local prediction (`runs/<run>/predictions/*.csv`) uses this code. The training-time validation files (`runs/<run>/val_predictions.csv`, written on the GPU) were only used for model selection and the comparison tables (D16, D34); calibration, referral and every test result use `predictions/*.csv` (D38).

No effect on results. Roughly halves CPU time for prediction, TTA and the stress test.

## D37: No test-time augmentation (decided on validation)

Options: plain predictions or four-view TTA (D24). Plain only on test, by the rule in D35 item 4 (adopt TTA only if validation QWK rises by at least 0.005 without a higher referable miss rate).

E2 on validation, both predicted locally with the same code (`runs/e2_effb0_512_ce/predictions/eval_val/metrics.json` and `eval_val_tta/metrics.json`):
- QWK 0.9294 plain vs 0.9278 TTA
- accuracy 0.8533 vs 0.8557
- macro-F1 0.7554 vs 0.7562
- referable miss rate 0.1568 vs 0.1649
- referable AUROC 0.9810 vs 0.9824

QWK fell and the miss rate rose, so the rule wasn't met. Small moves either way, all within run-to-run spread (D34). Reported as a negative finding. No TTA on test.

## D38: Pre-test audit and the controls it led to

Options: run test predictions with the lock as it was, or fix the findings of an independent review first. Fixed them first. Before any test prediction, the pipeline was reviewed against the plan by a reviewer who hadn't written the code. Findings and what was done:
1. Calibration and referral defaulted to `<run>/val_predictions.csv`, the training-time GPU file, while test predictions come from the CPU path (`predictions/*.csv`). Fitting temperature and thresholds on one path and applying them to another mixes two numerical paths. Defaults now `<run>/predictions/val.csv`, and both scripts refuse validation and test files that don't name the same run and checkpoint. The existing E2 and E3 calibration and referral outputs had already been fitted on `predictions/val.csv` (recorded by hash in `calibration.json`), so unchanged.
2. The written plan said only the selected model goes on test, which contradicted D35 item 2. Plan updated to name E3 as the declared secondary, as D35 had decided before any test prediction.
3. The per-folder TEST_LOCK.json could be bypassed: only written when a run finished, a copy of the run folder had no lock, and the lock held no hash of the output. Three controls added (`src/drgrade/provenance.py`): a committed allowlist of runs allowed on test (`configs/test_allowlist.yaml`: E2 and E3 only); a central append-only access log (`results/test_access_log.jsonl`, committed) with an "attempt" event before any test image is loaded and a "complete" event with the SHA-256 of the output CSV; and the CSV hash in each TEST_LOCK.json entry. Repeating a completed run and variant is refused in any folder unless `--force` and `--reason` are given. Unknown lock status refused. Locks copied to `results/` and committed after the test predictions.
4. Analysis scripts didn't check that input was complete or from the declared run. Every predictions CSV from `scripts/predict.py` now has a sidecar (`<file>.meta.json`: subset, run, checkpoint hash, TTA flag, rows, CSV hash). `evaluate.py` and `error_analysis.py` check the row count against the split and refuse a test file whose hash isn't in the access log and the lock. `calibrate.py` and `referral.py` need sidecars for both files. The existing E2 and E3 `predictions/val.csv` files predate sidecars; theirs were written afterwards and say so (`backfilled: true`), with the checkpoint hash read from `best.pt`.
5. Explainability stage, results collection script and some README steps not yet written. Not a blocker for test predictions; done afterwards.

Controls tested in `tests/test_provenance.py`, `tests/test_predict.py` (allowlist, log events, hashes, a copied run folder refused, unknown lock status), `tests/test_evaluate_smoke.py` (unregistered and truncated files refused), `tests/test_calibration.py` and `tests/test_referral.py`. Tests use a temporary allowlist and log (`tests/conftest.py`), so the suite never writes to the real log. Confirmed correct by the review: test isolation in training and selection, the split, the metrics, the calibration arithmetic, referral directions, the selection rule, the evaluation transform, channels_last and TTA.

Test set read once per declared output. Every test number in the report traces from the access log to a CSV hash and a checkpoint hash. The GPU and CPU validation files differ slightly; how much is in `results/val_path_differences.json`.

## D39: Grad-CAM layer for EfficientNet-B0 is bn2, not conv_head

Options: `conv_head` (the first idea: last convolution, before BatchNorm and SiLU), `bn2` (timm `BatchNormAct2d`, BatchNorm plus SiLU on the `conv_head` output), an earlier block. `bn2` (`src/drgrade/models.py`, `gradcam_target_layer`). Its output is exactly what global average pooling hands to the linear classifier. So the gradient of a class logit with respect to it is the classifier weight divided by the number of positions, and Grad-CAM there equals CAM (Zhou et al., 2016): the classifier-weighted sum of the final feature maps. At `conv_head`, BatchNorm and SiLU sit between the map and the score, so channel weights would be averages of position-dependent gradients. ResNet-18 keeps `layer4[-1]`, which also feeds the pooling directly.

`tests/test_gradcam_shapes.py::test_cam_equivalence_with_classifier_weights` checks, for both architectures, that the Grad-CAM map equals ReLU(sum_k W[c, k] A_k) / (h w) from the classifier weights. On trained E2, the Grad-CAM forward pass reproduces the logits in `runs/e2_effb0_512_ce/predictions/val.csv` and `test.csv` exactly (max absolute difference 0, in `explain/maps_val.json` and `maps_test.json`).

Maps are 16 by 16 for a 512 px input (one cell per 32 by 32 pixels), so they locate regions, not single microaneurysms. Upsampling is bilinear and adds nothing.

## D40: Explainability checks and what they found (E2, validation unless stated)

Options: heatmaps of selected cases only, heatmaps plus the planned sanity and faithfulness checks, a second attribution method (Integrated Gradients, optional in the plan; not done). Heatmaps plus checks (`scripts/explain.py`, `configs/explain.yaml`):
- Maps for every validation image (`explain/maps_val.npz`). On test, only the 20 distinct images among the 30 representative cases in `errors_test/cases.csv`. Test images read for explanation only, after the allowlist and provenance check on the locked test predictions, and each read logged as an "explain" event in `results/test_access_log.jsonl` (D38). Nothing chosen or changed from them.
- Randomisation test (Adebayo et al., 2018) on 50 sampled validation images: cascading re-initialisation from the classifier down to the stem, Spearman inside the fundus and SSIM on maps normalised by their max at 128 px. Reference: the same measures between trained maps of different images.
- Deletion test on 200 sampled validation images (the 50 are a subset): the most salient 5, 10, 20 and 40% of fundus pixels replaced by a Gaussian blur (radius 10 px), against the same share of random pixels and against the salient mask rotated about the centre by a random angle in [90, 270) degrees. The rotated mask keeps shape and size, so it separates "where" from "how clumped". Probability is the uncalibrated softmax of the predicted class.
- Shortcut check on all validation images: share of heatmap mass in the outer 5% of the fundus radius and in a disc of 10% of the radius around the brightest smoothed region (optic-disc proxy; lands on the disc in all 12 images of `explain/stats/disc_check.png`), each against its share of the area.
- Lesions: spread (share of fundus holding 50% of the mass) and peak, with and without each lesion flag, Mann-Whitney U, on all validation images and within true grade 2 or above.
- Case panels: image, predicted-class map and, for errors, the true-grade map, with the Spearman between the two.

Results (files under `runs/e2_effb0_512_ce/explain/`, follow-up in D41):
- Randomisation (`randomisation/summary.csv`): re-initialising only the classifier gives mean Spearman -0.20 with the trained map (negative in 36 of 50 images) and SSIM 0.29. Every later stage stays between -0.08 and 0.12 (Spearman) and 0.22 and 0.30 (SSIM). Trained maps of two different images: 0.10 and 0.41.
- Deletion (`deletion/summary.csv`, n = 200, of which 138 predicted grade 0): mean drop in predicted-class probability at 5, 10, 20, 40%: salient 0.17, 0.30, 0.61, 0.87; rotated 0.06, 0.14, 0.30, 0.71; random 0.01, 0.04, 0.09, 0.22. Paired salient minus rotated: 0.11, 0.16, 0.30, 0.16 (standard errors 0.02 to 0.03). Salient beats rotated in 61, 70, 85 and 90% of images (D41).
- Shortcut (`stats/shortcut_summary.json`, n = 1,268): edge band holds 6.5% of the mass on 9.8% of the area. Disc proxy holds 3.7% of the mass on 1.0% of the area; 4.3% when predicted grade is 0, 1.8% when it's 4.
- Lesions (`stats/lesion_comparison.csv`): over all validation images, maps are tighter (smaller spread) when each lesion is present. But lesions go with grade, so this is mostly a grade effect. Within each true grade, no lesion difference survives multiple-testing correction (D41).
- Case review (`results/explain_reviewer_notes.csv`, the `referable_miss` cases): in 4 of 6 test cases (true grade 2, predicted 0 or 1; a fifth borderline) and 2 of 6 validation cases (a third has a smaller spot there), the predicted-class map peaked over the fovea, often with the optic disc. The true-grade map mostly peaked elsewhere, with some overlap (mean Spearman between the two 0.29 on test, 0.11 on validation, 6 cases each). By eye, not measured.

What it means:
- Maps pass the deletion check: blurring the highlighted pixels drops the predicted-class probability more than blurring a same-shaped region elsewhere or random pixels. That's evidence about the model, not the retina; it doesn't show the highlighted regions are lesions. Blurred regions are outside the training distribution (the ROAR-style critique), probabilities are uncalibrated and many start near 1, so the 40% values sit near the ceiling.
- Randomisation shows the maps depend on the classifier weights: once those are random, maps are no closer to the trained ones than maps of unrelated images. Expected, since the map is the classifier-weighted feature map (D39), so the cascade alone can't show whether the learned features matter.
- A map explains the predicted class. For a grade 0 prediction it shows what raises the grade 0 score, which in the reviewed cases sat over the fovea or disc, so it can't show the lesions that were missed. For underestimation errors, the true-grade map is the more useful view for a reviewer. Main limitation of Grad-CAM here.
- Disc attention could be a shortcut or could be anatomy: neovascularisation at the disc shows up there, and disc appearance differs between cameras. Can't separate these. The proxy lands on the disc in the 12 checked images, mostly grade 0; not checked separately on severe grades, where haemorrhages and exudates can be brighter than the disc.
- 16 by 16 cells over 512 px (D39) is too coarse for microaneurysms, and lesion labels are image-level, so overlap with lesions can't be measured.

## D41: Review of the explainability stage, and the follow-up

Options: add every suggested extra check (per-layer randomisation, several seeds, SSIM inside the fundus, deletion controls for pixel change and mask overlap), or fix the claims and add only what the existing outputs allow. The second. Explainability is one part of a 2,000-word report, and the existing checks already go beyond what the task asks.

A reviewer who hadn't written the code checked it, the test-set handling and every number in D39 and D40. Confirmed: the Grad-CAM implementation, the bn2 argument, the independence of the CAM-equivalence test, the test-set handling (allowlist, provenance check and access event before any test image is read; no explain command changes a model, threshold or decision) and almost every number. Found: one wrong claim (a lesion difference "within grade 2 or above" that was really grade 4 vs grades 2 to 3), a case description that mixed grades, two rounding errors, and wording that went past the evidence. D40 rewritten from that. `scripts/explain_followup.py` adds, from existing outputs only:
- `results/explain_lesions_by_grade.csv`: spread, edge share and disc share with and without each lesion within each true grade, Mann-Whitney U, Benjamini-Hochberg and Bonferroni adjusted across the 39 tests with at least 5 images per group. Peak left out: the raw map max follows the predicted-class logit, so it measures confidence, not map shape.
- `results/explain_deletion_by_class.csv` and `results/explain_followup.json`: deletion by predicted class, and the share of images where salient beats each baseline.

Results (`results/explain_followup.json`, `results/explain_deletion_by_class.csv`):
- Lesions within grade: none of the 39 differences below 0.05 after Benjamini-Hochberg or Bonferroni.
- Deletion by predicted class at 5% (salient, rotated, random): grade 0 (n = 138) 0.07, 0.07, 0.00; grade 1 (n = 17) 0.51, 0.18, 0.10; grade 2 (n = 27) 0.48, 0.01, 0.01; grade 3 (n = 10) 0.24, -0.03, -0.02; grade 4 (n = 8) 0.09, -0.01, -0.02. At 20%: grade 0 0.64, 0.37, 0.06.
- Randomisation, classifier stage: Spearman negative in 72% of the 50 images, from one random draw of the classifier.

What it means:
- For grade 1 to 3 predictions, a small highlighted region carries the decision: blurring 5% of the fundus there drops the probability far more than the same shape elsewhere. For grade 0 the evidence is spread out: at 5%, salient and rotated matter equally, and the gap only appears at larger fractions. Grade 4 predictions are so confident that small deletions barely move them. The pooled curve hides all this, so the report quotes the breakdown.
- Once grade is fixed, lesion flags add nothing measurable to the heatmap statistics, at this map resolution and sample size.
- Known limits left as they are, stated in the report: SSIM averaged over the whole square including black corners, which raises it for every comparison; random deletion masks drawn fresh at each fraction instead of nested; the deletion fundus is the inscribed circle, which includes black bands in images cut flat at top and bottom; validation images not fully independent (repeat captures grouped, but one patient's two eyes may both be in).

## D42: Publishing predictions, metadata and models for reproduction

Options: keep `runs/` private and ask a reader to retrain (results would differ slightly, D34), commit predictions and metric files only, or also commit the two models evaluated on test. The last.

`scripts/collect_results.py` copies the report's inputs into `results/`: predictions with provenance sidecars, metric, calibration and referral files, the audit summary, explainability summaries, and `manifest_meta.csv` (per-image metadata, no pixels). Writes `results/MANIFEST.json` with SHA-256 hashes and sources. `scripts/reproduce_report.py` reruns the project's own evaluation, calibration and referral code on those files and compares each result with the value printed in the report (`results/report_expected.json`). E2 and E3 models committed in `models/` (16 MB each), so anyone with the dataset can regenerate the predictions, checked by `scripts/check_predictions.py`.

The original plan was to commit nothing derived from the images. Exception for these files only: MMRDR is CC BY 4.0 (figshare record 29423747), which allows redistribution with attribution. No images committed. `decide_test_access` now lets `--force --reason` through when a folder has no lock, so a reproduction from a fresh clone is allowed and logged rather than blocked by the committed access log.

`scripts/reproduce_report.py` reproduces 95 of 95 report numbers from `results/` alone. It also caught an error in the report draft: "102 of them as grade 1" should have been 113 (102 is the grade 2 to grade 1 count; 11 grade 3 and 4 images were also graded 1). Fixed. Re-predicting 272 validation images with `models/e2_effb0_512_ce` on my local CPU gave identical grades and a largest logit difference of 0.0.

Anyone can check every reported number in about a minute without the data, and regenerate the predictions with it. Retraining only reproduces results to within the run-to-run spread in D34.
