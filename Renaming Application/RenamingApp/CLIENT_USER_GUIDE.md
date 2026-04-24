# SlopeSense - Client User Guide

## 1. What This App Does
SlopeSense supports two related workflows:

1. `Run`: matches GoPro videos to MATLAB tipper files, applies any needed human review, and creates renamed video outputs.
2. `Validate`: runs the integrated computer-vision inference pipeline on videos and compares model predictions against the tipper label in the filename.

The validation pipeline classifies each video as `Pass` or `Fail` using:
- YOLO pose extraction
- CTR-GCN classification

At the end of each run, the app writes reports to your selected `Reports directory`.

## 2. Before You Start
Make sure your data is organized by date folders.

Expected structure:

```text
Videos/
  2024-06-01/
    sub001/
      GX010001.MP4
      GX010002.MP4
      ...
  2024-06-02/
    sub002/
      ...

Tipper/
  2024-06-01/
    idapt811_sub001_DP_GP1_14-01-48.mat
    idapt811_sub001_DP_0_GP1_14-02-10.mat
    ...
  2024-06-02/
    ...
```

Supported tipper filename formats include:
- `shoe_sub_DP_GP1_HH-MM-SS`
- `shoe_sub_DP_angle_GP1_HH-MM-SS`
- `shoe_sub_DP_angle_HH-MM-SS`
- `shoe_sub_DP_HH-MM-SS`

If angle is missing, the app reads it from MATLAB data and injects it into the filename as needed.

For validation to work, the packaged app must include these model files:
- `models/classifier.onnx`
- `models/yolo26x-pose.onnx`

## 3. Main Screen Overview
The app has two tabs:

### Run Tab
This is the main working screen. It contains:
- `Videos directory`: source videos root
- `Tipper directory`: source tipper root
- `Output directory`: where renamed videos are copied
- `Write log to file`: optional toggle
- `Log file path`: auto-managed path inside the Reports directory
- `Reports directory`: where CSV/XLSX reports are saved
- `Dry run`: preview mode with no actual copy or rename operations
- `Run` button: starts video-to-tipper matching and renaming
- `Validate` button: runs CV inference on videos
- `Cancel` button: cancels an active run or validation job
- Progress panel and progress bar

### Tipper Preview Tab
Shows the current date's tipper rows and live status updates:
- `Pending`
- `Current`
- `Matched`
- `Skipped`
- `Corrected`
- `Unmatched`

## 4. Recommended User Flow
The current app is best used in this order:

1. Set all folder paths.
2. Run a `Dry run` first.
3. Review the renaming reports and any corrections.
4. Run again with `Dry run` off if the preview looks correct.
5. After the renaming run completes, click `Validate`.
6. Review the validation Excel report in the Reports directory.

You can also click `Validate` without doing a renaming run in the current session. In that case, the app scans the `Videos directory` directly and validates all supported video files it finds.

## 5. How To Run The Renaming Workflow
1. Open the app.
2. Set `Videos directory`.
3. Set `Tipper directory`.
4. Set `Output directory`.
5. Set `Reports directory`.
6. Optional: enable `Write log to file`.
7. Optional: enable `Dry run`.
8. Click `Run`.
9. Respond to any HITL dialogs during processing.
10. Wait for the completion popup.
11. Review outputs in:
- Output folder, if `Dry run` was off
- Reports folder

When a renaming run finishes successfully, the app may tell you how many videos are available for validation and prompt you to click `Validate`.

## 6. How To Run Validation
Validation runs the inference pipeline on videos and produces a separate Excel report.

### Validation After A Renaming Run
If you click `Validate` immediately after a run, the app validates the renamed video outputs from that session.

### Validation Without A Prior Run
If you click `Validate` without first running the renaming workflow in the current session, the app scans the `Videos directory` directly and validates all supported video files it finds.

Supported video extensions for validation include:
- `.mp4`
- `.mov`
- `.m4v`
- `.avi`
- `.mkv`

### What Validation Does
For each video, the app:
- extracts poses with YOLO
- preprocesses the skeleton sequence
- classifies the video as `Pass` or `Fail` using CTR-GCN
- compares the model prediction to the tipper label encoded in the renamed filename when available

Validation does not rename files or edit tipper files.

## 7. HITL Dialogs
### A) Direction Needed
Appears when video direction is undecided.

You will see:
- video name
- thumbnail preview
- `Open in Player` button

Actions:
- `Down (D)`: force video direction to `D` and continue
- `Up (U)`: force video direction to `U` and continue
- `Skip`: skip this video
- `Abort Run` or closing the dialog: aborts the run

### B) Tipper Angle Review
Appears when tipper angle is `0` and result is undecided (`U`).

Actions:
- `Keep Undecided (U)`
- `Mark Pass (P)`
- `Mark Fail (F)`
- `Delete/Skip`
- `Abort Run` or closing the dialog: aborts the run

### C) Direction Conflict
Appears when a video direction and current tipper direction do not match.

Actions:
- `Fix Video Direction`: changes the video direction in memory for matching
- `Fix Tipper Direction`: updates the tipper direction and, if not in dry run, applies the tipper rename
- `Skip Video`
- `Skip Tipper`
- `Abort Run` or closing the dialog: aborts the run

## 8. Dry Run vs Normal Run
### Dry Run ON
- No actual copy or rename operations
- Reports and logs still show what would happen
- Best for a first pass

### Dry Run OFF
- Videos are copied to the output folder with matched renamed filenames
- Tipper corrections that require renaming are applied

## 9. Reports Explained
All reports are written to your selected `Reports directory`.

### Renaming Reports
`run.log`
- Optional
- Written only if `Write log to file` is enabled

`tipper_corrections.csv`
- Tracks tipper file corrections
- Columns: `date`, `sub`, `original_name`, `new_name`, `reason`

`failed_folders.csv`
- Folders/date-sub entries that failed or were skipped
- Column: `folder`

`video_mappings.xlsx`
- One row per matched video/tipper mapping
- Includes:
- `tipper_filename`
- `date`
- `time`
- `sub`
- `shoe`
- `Trial Number`
- `Angle`
- `Ice Temperature degC`
- `Air Humidity RH`
- `Air Temperature degC`
- `Ultrasonic Distance (m)`
- `Total Trial Time`
- `original_video`
- `renamed_video`
- `dry_run`

`MAA/`
- One workbook per participant per date
- Path format: `Reports directory/MAA/<date>/<sub>.xlsx`
- Each shoe gets its own worksheet
- Includes uphill/downhill MAA summaries, angle counts, and trial details

### Validation Report
`validation_results_YYYYMMDD_HHMMSS.xlsx`
- Written each time validation is run
- Contains one row per validated video
- Includes:
- `Original Video`
- `Renamed Video`
- `Tipper Classification`
- `Model Classification`
- `Model Confidence`
- `Match`
- `Error`

The validation sheet also includes a summary with:
- total videos
- number of matches
- agreement rate
- number of errors

## 10. Completion and Cancellation
### Renaming Completion Popup
The completion popup reports:
- whether processing completed or was cancelled
- skipped folder count
- unmatched video/tipper counts
- reports directory
- whether videos are available for validation

### Validation Completion Popup
The validation popup reports:
- total number of classified videos
- agreement with tipper labels, when comparable labels exist
- error count

### Cancel Button
`Cancel` requests a safe stop for either:
- a renaming run
- a validation run

### Important
Closing or cancelling any HITL dialog aborts the renaming run.

## 11. Common Problems and What To Check
### No Files Processed During Run
- Verify `Videos directory` and `Tipper directory`
- Confirm both use matching date folder names

### Validation Says Model Files Are Missing
- Confirm the packaged app includes:
- `models/classifier.onnx`
- `models/yolo26x-pose.onnx`

### No Videos Found To Validate
- Check that the `Videos directory` is set correctly
- If validating after a run, confirm the run produced mapped video outputs

### Many Unmatched Files
- Check date/sub alignment
- Review the `Tipper Preview` tab to see where matching diverged

### Missing Outputs
- If `Dry run` was on, no copied or renamed files are expected
- Otherwise, review `run.log` and the reports folder

### Aborted Run
- Check whether a HITL dialog was closed
- Review `failed_folders.csv` and `run.log`

## 12. Quick Reference Checklist
- Set `Videos directory`
- Set `Tipper directory`
- Set `Output directory`
- Set `Reports directory`
- Decide whether to use `Dry run`
- Click `Run`
- Respond to HITL dialogs if prompted
- Review renaming reports
- Click `Validate`
- Review `validation_results_YYYYMMDD_HHMMSS.xlsx`
