# aps490-capstone-kite

## Batch Direction Detection

This repo includes a small CLI (`videonamefixer.VideoDirectionBatch`) that runs the OpenCV-based `DirectionDetector` over all participant videos and writes a CSV summary.

Expected directory layout:

```
videos/
  <participant_id>/
    *.mp4
```

Each video is analyzed and the result is written with columns:

- participant_id
- file_name
- direction (LEFT | RIGHT | UNDETERMINED)
  - If a video throws an exception or cannot be opened, the value is "error processing".

### Run

From the repo root, either of the following:

- `cd VideoNameFixer && ./gradlew runBatchDirection`
- `./VideoNameFixer/gradlew :app:runBatchDirection`

By default, it scans `<repo>/videos` and writes `<repo>/direction_results.csv` using a `sampleStep` of 10. It also gives extra weight to optical-flow direction measured in the first 2 seconds to reduce misclassification from brief slips later in the clip.

### Options

You can override inputs/outputs via Gradle properties:

```
./VideoNameFixer/gradlew :app:runBatchDirection \
  -PvideosDir=/absolute/or/relative/path/to/videos \
  -PoutCsv=/absolute/or/relative/path/to/output.csv \
  -PsampleStep=10 \
  -PnoMotionThreshold=0.03 \
  -PearlySeconds=2.0 \
  -PearlyWeight=2.0
```

Notes:

- Only `.mp4` files directly inside each participant folder are processed.
- If a video fails to process, the direction defaults to `UNDETERMINED` and processing continues.
 - You can lower `-PnoMotionThreshold` (default 0.05) if some true directional videos are marked `UNDETERMINED`.
 - To bias toward the initial movement direction, adjust `-PearlySeconds` (window length) and `-PearlyWeight` (multiplier). Set `-PearlyWeight=1.0` to disable.
