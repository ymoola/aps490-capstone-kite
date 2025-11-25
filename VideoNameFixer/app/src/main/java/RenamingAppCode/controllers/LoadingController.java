package RenamingAppCode.controllers;

import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.ProgressBar;
import javafx.stage.Stage;
import videonamefixer.DirectionDetector;
import videonamefixer.video_processing.VideoBatchRunner;
import videonamefixer.video_processing.MovementVideoProcessor;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class LoadingController {

    @FXML
    private ProgressBar ProgessIndictor;

    private String videoFolderPath;
    private String maaFolderPath;
    private String tipperFolderPath;

    /**
     * Initialize method called after FXML is loaded
     */
    @FXML
    public void initialize() {
        // Set initial progress to 0
        if (ProgessIndictor != null) {
            ProgessIndictor.setProgress(0.0);
        }
    }

    /**
     * Set the folder paths from the previous screen
     */
    public void setFolderPaths(String videoPath, String maaPath, String tipperPath) {
        this.videoFolderPath = videoPath;
        this.maaFolderPath = maaPath;
        this.tipperFolderPath = tipperPath;
    }

    /**
     * Start the file renaming process
     */
    public void startProcessing() {
        Task<Void> processingTask = new Task<Void>() {
            @Override
            protected Void call() throws Exception {
                try {
                    // Update progress: Starting
                    updateProgress(0.1, 1.0);

                    // Process video files
                    if (videoFolderPath != null && !videoFolderPath.isEmpty()) {
                        processVideoFolder();
                        updateProgress(0.5, 1.0);
                    }

                    // Process MAA files (placeholder for future implementation)
                    if (maaFolderPath != null && !maaFolderPath.isEmpty()) {
                        // TODO: Implement MAA file processing
                        updateProgress(0.75, 1.0);
                    }

                    // Process Tipper files (placeholder for future implementation)
                    if (tipperFolderPath != null && !tipperFolderPath.isEmpty()) {
                        // TODO: Implement Tipper file processing
                        updateProgress(0.9, 1.0);
                    }

                    // Complete
                    updateProgress(1.0, 1.0);

                } catch (Exception e) {
                    e.printStackTrace();
                    throw e;
                }
                return null;
            }

            @Override
            protected void succeeded() {
                super.succeeded();
                navigateToResults();
            }

            @Override
            protected void failed() {
                super.failed();
                System.err.println("Processing failed: " + getException().getMessage());
                getException().printStackTrace();
            }
        };

        // Bind progress bar to task progress
        ProgessIndictor.progressProperty().bind(processingTask.progressProperty());

        // Run task in background thread
        Thread thread = new Thread(processingTask);
        thread.setDaemon(true);
        thread.start();
    }

    /**
     * Process video folder using VideoBatchRunner
     */
    private void processVideoFolder() throws IOException {
        Path videoDir = Paths.get(videoFolderPath);

        // Create output file for results
        Path outputFile = videoDir.getParent().resolve("video_processing_results.csv");

        try (BufferedWriter writer = Files.newBufferedWriter(outputFile)) {
            // Write CSV header
            writer.write("ParticipantID,FileName,Direction,DeltaSeconds");
            writer.newLine();

            // Create processor and batch runner with default parameters
            MovementVideoProcessor processor = new MovementVideoProcessor(
                10, // sampleStep
                DirectionDetector.DEFAULT_NO_MOTION_THRESHOLD,
                DirectionDetector.DEFAULT_EARLY_SECONDS,
                DirectionDetector.DEFAULT_EARLY_WEIGHT
            );
            VideoBatchRunner batchRunner = new VideoBatchRunner(processor);

            // Run batch processing
            batchRunner.runBatch(videoDir, writer);
        }

        System.out.println("Video processing complete. Results saved to: " + outputFile);
    }

    /**
     * Navigate to the Results page
     */
    private void navigateToResults() {
        Platform.runLater(() -> {
            try {
                FXMLLoader loader = new FXMLLoader(getClass().getResource("/fxml/Results.fxml"));
                Parent root = loader.load();

                // Get the Results controller and pass data if needed
                ResultsController resultsController = loader.getController();
                if (resultsController != null) {
                    resultsController.setFolderPaths(videoFolderPath, maaFolderPath, tipperFolderPath);
                }

                // Get current stage and switch scene
                Stage stage = (Stage) ProgessIndictor.getScene().getWindow();
                Scene scene = new Scene(root);
                stage.setScene(scene);
                stage.show();

            } catch (IOException e) {
                e.printStackTrace();
            }
        });
    }
}
