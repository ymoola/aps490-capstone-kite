package RenamingAppCode.controllers;

import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;
import javafx.stage.DirectoryChooser;
import javafx.stage.Stage;

import java.io.File;
import java.io.IOException;

public class FolderSelectionController {

    @FXML
    private TextField videoPathField;

    @FXML
    private TextField maaPathField;

    @FXML
    private TextField tipperPathField;

    @FXML
    private Button renameButton;

    private String selectedVideoPath;
    private String selectedMAAPath;
    private String selectedTipperPath;

    /**
     * Browse for video folder
     */
    @FXML
    private void browseVideoFolder() {
        selectedVideoPath = browseForFolder("Select Video Folder");
        if (selectedVideoPath != null) {
            videoPathField.setText(selectedVideoPath);
        }
    }

    /**
     * Browse for MAA folder
     */
    @FXML
    private void browseMAAFolder() {
        selectedMAAPath = browseForFolder("Select MAA Folder");
        if (selectedMAAPath != null) {
            maaPathField.setText(selectedMAAPath);
        }
    }

    /**
     * Browse for Tipper folder
     */
    @FXML
    private void browseTipperFolder() {
        selectedTipperPath = browseForFolder("Select Tipper Folder");
        if (selectedTipperPath != null) {
            tipperPathField.setText(selectedTipperPath);
        }
    }

    /**
     * Helper method to open directory chooser
     */
    private String browseForFolder(String title) {
        DirectoryChooser directoryChooser = new DirectoryChooser();
        directoryChooser.setTitle(title);

        Stage stage = (Stage) renameButton.getScene().getWindow();
        File selectedDirectory = directoryChooser.showDialog(stage);

        if (selectedDirectory != null) {
            return selectedDirectory.getAbsolutePath();
        }
        return null;
    }

    /**
     * Handle rename button click - navigate to Loading page
     */
    @FXML
    private void handleRenameFiles() {
        try {
            // Get the paths from text fields
            selectedVideoPath = videoPathField.getText();
            selectedMAAPath = maaPathField.getText();
            selectedTipperPath = tipperPathField.getText();

            // Load the Loading screen
            FXMLLoader loader = new FXMLLoader(getClass().getResource("/fxml/Loading.fxml"));
            Parent root = loader.load();

            // Get the Loading controller and pass the folder paths
            LoadingController loadingController = loader.getController();
            if (loadingController != null) {
                loadingController.setFolderPaths(selectedVideoPath, selectedMAAPath, selectedTipperPath);
                // Start processing after scene is set
            }

            // Switch to Loading scene
            Stage stage = (Stage) renameButton.getScene().getWindow();
            Scene scene = new Scene(root);
            stage.setScene(scene);
            stage.show();

            // Start processing after the scene is displayed
            if (loadingController != null) {
                loadingController.startProcessing();
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
