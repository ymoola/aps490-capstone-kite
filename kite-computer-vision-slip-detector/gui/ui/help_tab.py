from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextBrowser


HELP_HTML = """
<h2>SlopeSense Help</h2>
<p>
This app turns raw rehabilitation or gait videos into skeleton data, uses that skeleton
data to train a slip-detection model, and then saves a final production-ready model.
</p>

<h3>End-to-End Workflow</h3>
<ol>
  <li><b>Config</b>: choose where your videos live, where outputs should be saved, and how much preprocessing and training you want.</li>
  <li><b>Videos</b>: scan the selected video folder, preview clips, and run pose extraction. This creates skeleton files from each video.</li>
  <li><b>Poses</b>: inspect the saved skeleton files, preview overlays, and build the machine-learning dataset.</li>
  <li><b>Training</b>: run hyperparameter search. The app tries combinations of training settings and compares their validation performance.</li>
  <li><b>Production</b>: train one final model using the best settings found during HPO and save the final checkpoint.</li>
</ol>

<h3>How To Use Each Tab</h3>

<h4>1. Config Tab</h4>
<ol>
  <li>Choose the folder that contains the raw videos you want to process.</li>
  <li>Choose a <b>Pose Output Folder</b>. This is where extracted skeleton files and project checkpoint state will live.</li>
  <li>Choose a <b>Runs Output Folder</b>. This stores all HPO experiments and training charts.</li>
  <li>Choose a <b>Production Output Folder</b>. This stores the final production model and its summaries.</li>
  <li>Point the app to the YOLO pose model file and the CTR-GCN repo folder.</li>
  <li>Adjust preprocessing and training settings only if you need to. For first tests, keep them small and simple.</li>
  <li>Click <b>Save Project</b> once the paths look right.</li>
</ol>

<h4>2. Videos Tab</h4>
<ol>
  <li>Click <b>Scan Videos</b> to load supported videos from the configured video folder.</li>
  <li>Check the status column. New videos have not been processed yet. Stale videos were processed before but need to be regenerated.</li>
  <li>Preview a few videos to confirm you selected the right folder.</li>
  <li>Start pose extraction. This converts each video into a skeleton-based pose file.</li>
  <li>If you stop halfway through, you can reopen the project later and continue from the saved checkpoint state.</li>
</ol>

<h4>3. Poses Tab</h4>
<ol>
  <li>Click <b>Scan Poses</b> to load extracted <code>.npz</code> pose files.</li>
  <li>Single-click a pose file to see a still skeleton preview and metadata.</li>
  <li>Double-click a pose file to render and open the full skeleton animation.</li>
  <li>Once the pose files look correct, click <b>Build Dataset</b> to create the CTR-GCN train/validation/test fold files.</li>
</ol>

<h4>4. Training Tab</h4>
<ol>
  <li>Click <b>Scan Existing Runs</b> if you want to load earlier HPO experiments from disk.</li>
  <li>Click <b>Start HPO</b> to begin training across the configured hyperparameter grid.</li>
  <li>Watch the live training curve and confusion matrix update as each run progresses.</li>
  <li>Use the run list on the left to inspect completed or interrupted runs.</li>
  <li>If a run is incomplete, the orange status banner will appear and the <b>Restart Run</b> button becomes available when training is idle.</li>
  <li>The bottom summary ranks the best hyperparameter combinations across folds.</li>
</ol>

<h4>5. Production Tab</h4>
<ol>
  <li>Open this tab only after HPO has produced a usable summary of best hyperparameters.</li>
  <li>Review the best hyperparameter block at the top.</li>
  <li>Click <b>Train Production Model</b> to train one final model using the best HPO settings.</li>
  <li>Use the live chart to monitor training and the model path field to find the final saved checkpoint.</li>
  <li>Copy the path if you need to share or load the final model elsewhere.</li>
</ol>

<h3>What Each Stage Produces</h3>
<ul>
  <li><b>Pose extraction</b>: compressed <code>.npz</code> skeleton files for each video.</li>
  <li><b>Dataset building</b>: CTR-GCN-ready train/validation/test fold files.</li>
  <li><b>Training</b>: per-run histories, validation metrics, test metrics, and HPO summaries.</li>
  <li><b>Production</b>: a final best model checkpoint plus split and training summaries.</li>
</ul>

<h3>How Restart / Resume Works</h3>
<p>
The app stores checkpoint information inside <code>&lt;Pose Output Folder&gt;/.slopesense</code>.
When you reopen the project, it compares the saved state with the files currently on disk.
</p>
<ul>
  <li><b>New</b>: the video was found in the video folder but has never been processed in this project.</li>
  <li><b>Stale</b>: the video was processed before, but the saved pose output is missing or your pose settings changed.</li>
  <li><b>Completed</b>: the current saved result matches the current pose settings.</li>
  <li><b>Failed</b>: a previous extraction attempt ended with an error.</li>
</ul>

<h3>Recommended First Test</h3>
<ul>
  <li>Use a fresh empty pose output folder, runs folder, and production folder.</li>
  <li>Start with a small set of videos.</li>
  <li>Use a tiny HPO grid such as one batch size, one learning rate, one weight decay, and 2 epochs for a quick smoke test.</li>
</ul>

<h3>Practical Tips</h3>
<ul>
  <li>If <b>Scan Videos</b> finds zero videos, check that the Video Folder is correct and contains supported files such as <code>.mp4</code>, <code>.avi</code>, <code>.mov</code>, <code>.mkv</code>, or <code>.wmv</code>.</li>
  <li>If you want a fully clean run, point the config at brand-new empty output folders.</li>
  <li>If training is too slow, reduce folds, epochs, and the size of the hyperparameter grid for testing.</li>
  <li>If results look noisy, try leaving interpolation and smoothing enabled before changing model settings.</li>
</ul>
"""


class HelpTab(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        viewer = QTextBrowser()
        viewer.setOpenExternalLinks(True)
        viewer.setHtml(HELP_HTML)
        layout.addWidget(viewer)
