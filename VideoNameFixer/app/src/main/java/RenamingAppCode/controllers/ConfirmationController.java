package RenamingAppCode.controllers;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.stage.Stage;

import java.io.IOException;

public class ConfirmationController {

    @FXML
    private Label videoPathLabel;

    @FXML
    private Label maaPathLabel;

    @FXML
    private Label tipperPathLabel;

    @FXML
    private Button confirmButton;

    @FXML
    private Button editButton;

    // Keep current paths so we can send them back if user clicks "Edit"
    private String videoPath;
    private String maaPath;
    private String tipperPath;

    /** Called by the previous controller to pass in the paths */
    public void setPaths(String videoPath, String maaPath, String tipperPath) {
        this.videoPath = videoPath;
        this.maaPath = maaPath;
        this.tipperPath = tipperPath;

        videoPathLabel.setText(videoPath);
        maaPathLabel.setText(maaPath);
        tipperPathLabel.setText(tipperPath);
    }

    private Stage getStage() {
        return (Stage) confirmButton.getScene().getWindow();
    }

    // Go to Loading.fxml when user confirms
    @FXML
    private void handleConfirm(ActionEvent event) throws IOException {
        FXMLLoader loader =
                new FXMLLoader(getClass().getResource("/fxml/Loading.fxml"));
        Parent root = loader.load();

        Stage stage = getStage();
        stage.setScene(new Scene(root));
        stage.show();
    }

    // Go back to the folder selection page and pre-fill the paths
    @FXML
    private void handleEdit(ActionEvent event) throws IOException {
        // If your first page FXML is actually named something else, change this:
        FXMLLoader loader =
                new FXMLLoader(getClass().getResource("/fxml/FolderSelection.fxml"));
        Parent root = loader.load();

        FolderSelectionController controller = loader.getController();
        controller.setInitialPaths(videoPath, maaPath, tipperPath);

        Stage stage = getStage();
        stage.setScene(new Scene(root));
        stage.show();
    }

}
