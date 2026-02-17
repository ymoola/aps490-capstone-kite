# Video Name Fixer - Client User Guide

## 1. What This App Does
Video Name Fixer matches GoPro videos to MATLAB tipper files and creates renamed video outputs based on the matched tipper filenames.

The app also supports Human-in-the-Loop (HITL) decisions when:
- video direction cannot be auto-detected,
- tipper angle/result needs manual review,
- video and tipper directions conflict.

At the end of each run, the app writes run reports to your selected reports folder.

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

## 3. Main Screen Overview
The **Run** tab contains:
- `Videos directory`: source videos root
- `Tipper directory`: source tipper root
- `Output directory`: where renamed videos are copied
- `Write log to file`: optional toggle
- `Log file path`: auto-managed path (inside Reports directory)
- `Reports directory`: where CSV/XLSX reports are saved
- `Dry run`: simulate actions without copying/renaming files
- `Run` and `Cancel` buttons
- Progress panel and progress bar

The **Tipper Preview** tab shows current date tipper rows and live status updates:
- `Pending`, `Current`, `Matched`, `Skipped`, `Corrected`, `Unmatched`

## 4. How To Run (Step-by-Step)
1. Open the app.
2. Set `Videos directory`.
3. Set `Tipper directory`.
4. Set `Output directory`.
5. Set `Reports directory`.
6. Optional: enable `Write log to file`.
7. Optional: enable `Dry run` for a simulation pass.
8. Click `Run`.
9. Respond to any HITL dialogs during processing.
10. Wait for the completion popup.
11. Review outputs in:
- Output folder (or simulated actions if dry run),
- Reports folder (`run.log`, `tipper_corrections.csv`, `failed_folders.csv`, `video_mappings.xlsx`).

## 5. HITL Dialogs 
### A) Direction Needed
Appears when video direction is undecided.

You will see:
- video name,
- thumbnail preview,
- `Open in Player` button.

Actions:
- `Down (D)`: force video direction to D and continue.
- `Up (U)`: force video direction to U and continue.
- `Skip`: skip this video.
- `Abort Run` or closing dialog: aborts the run.

### B) Tipper Angle Review
Appears when tipper angle is 0 and result is undecided (`U`).

Actions:
- `Keep Undecided (U)`: keep as U.
- `Mark Pass (P)`: update result to P.
- `Mark Fail (F)`: update result to F.
- `Delete/Skip`: remove this tipper from matching sequence.
- `Abort Run` or closing dialog: aborts the run.

### C) Direction Conflict
Appears when a video direction and current tipper direction do not match.

You can choose correction direction from dropdowns and apply:
- `Fix Video Direction`: changes video direction in-memory for matching.
- `Fix Tipper Direction`: updates tipper direction (and name if not dry run), then accepts match.
- `Skip Video`: skip this video entry.
- `Skip Tipper`: skip this tipper entry.
- `Abort Run` or closing dialog: aborts the run.

## 6. Dry Run vs Normal Run
### Dry Run ON
- No actual copy/rename file operations.
- App logs and reports what would happen.
- Useful for validation before committing changes.

### Dry Run OFF
- Videos are copied to output with matched renamed filenames.
- Tipper corrections that require renames are applied.

## 7. Reports Explained
All reports are written to your selected `Reports directory`.

### 1) `run.log` (optional, if enabled)
Detailed runtime logs and warnings.

### 2) `tipper_corrections.csv`
Tracks tipper file corrections:
- `date`, `sub`, `original_name`, `new_name`, `reason`

### 3) `failed_folders.csv`
Folders/date-sub entries that failed or were skipped:
- `folder`

### 4) `video_mappings.xlsx`
Per matched mapping details:
- `tipper_filename`
- `date`
- `time`
- `sub`
- `shoe`
- `Trial Number`
- `Angle` (stored as integer, half-up rounding)
- `Ice Temperature degC`
- `Air Humidity RH`
- `Air Temperature degC`
- `Ultrasonic Distance (m)`
- `Total Trial Time`
- `original_video`
- `renamed_video`
- `dry_run`

## 8. Completion and Cancellation
### Completion Popup
At end of run, app shows summary:
- completed vs cancelled,
- skipped folder count,
- unmatched counts,
- report directory.

### Cancel Button
`Cancel` requests a safe stop. It may take time if current video analysis step is still running.

### Important
Closing/aborting any HITL dialog aborts the full run by design.

## 9. Common Problems and What To Check
### No files processed
- Verify `Videos directory` and `Tipper directory` are correct.
- Confirm both use matching date folder names.

### Many unmatched files
- Check date/sub folder alignment.
- Review `Tipper Preview` statuses for where mismatches started.

### Missing outputs
- If `Dry run` was ON, this is expected (simulation only).
- If OFF, check `run.log` for copy/rename warnings.

### Aborted run
- Check if any HITL dialog was closed.
- Review `failed_folders.csv` and `run.log`.

## 10. Recommended Operational Workflow
1. First pass: run with `Dry run` ON.
2. Review `video_mappings.xlsx` and `tipper_corrections.csv`.
3. Resolve any data layout issues.
4. Second pass: run with `Dry run` OFF.
5. Archive reports with the delivered output set.

## 11. Quick Reference Checklist
- Set all four directories.
- Decide dry run ON/OFF.
- Click Run.
- Respond to HITL dialogs.
- Wait for completion popup.
- Review reports.

