package RenamingAppCode.controllers;

import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.stage.Stage;
import java.awt.Desktop;
import java.io.File;
import java.io.IOException;

public class ResultsController {

    @FXML
    private Button VideoFolderBrowseButtonFinal;

    @FXML
    private Button MAAFolderBrowseButtonFinal;

    @FXML
    private Button TipperFolderBrowseButtonFinal;

    @FXML
    private Button FinishButton;

    @FXML
    private Button RestartButton;

    private String videoFolderPath;
    private String maaFolderPath;
    private String tipperFolderPath;

    /**
     * Set the folder paths from the loading screen
     */
    public void setFolderPaths(String videoPath, String maaPath, String tipperPath) {
        this.videoFolderPath = videoPath;
        this.maaFolderPath = maaPath;
        this.tipperFolderPath = tipperPath;
    }

    /**
     * Open video folder in file explorer
     */
    @FXML
    private void openVideoFolder() {
        openFolderInExplorer(videoFolderPath);
    }

    /**
     * Open MAA folder in file explorer
     */
    @FXML
    private void openMAAFolder() {
        openFolderInExplorer(maaFolderPath);
    }

    /**
     * Open Tipper folder in file explorer
     */
    @FXML
    private void openTipperFolder() {
        openFolderInExplorer(tipperFolderPath);
    }

    /**
     * Helper method to open a folder in the system file explorer
     */
    private void openFolderInExplorer(String folderPath) {
        if (folderPath == null || folderPath.isEmpty()) {
            System.out.println("No folder path specified");
            return;
        }

        try {
            File folder = new File(folderPath);
            if (folder.exists() && folder.isDirectory()) {
                Desktop.getDesktop().open(folder);
            } else {
                System.err.println("Folder does not exist: " + folderPath);
            }
        } catch (IOException e) {
            System.err.println("Failed to open folder: " + folderPath);
            e.printStackTrace();
        }
    }

    /**
     * Handle Finish button - close the application
     */
    @FXML
    private void handleFinish() {
        Stage stage = (Stage) FinishButton.getScene().getWindow();
        stage.close();
    }

    /**
     * Handle Restart button - go back to welcome screen
     */
    @FXML
    private void handleRestart() {
        try {
            javafx.fxml.FXMLLoader loader = new javafx.fxml.FXMLLoader(
                getClass().getResource("/fxml/Welcome.fxml")
            );
            javafx.scene.Parent root = loader.load();

            Stage stage = (Stage) RestartButton.getScene().getWindow();
            javafx.scene.Scene scene = new javafx.scene.Scene(root);
            stage.setScene(scene);
            stage.show();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
