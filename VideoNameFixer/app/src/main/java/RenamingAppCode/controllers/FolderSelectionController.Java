package RenamingAppCode.controllers;

import javafx.event.ActionEvent;
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

    // --- Utility to get the current Stage ---
    private Stage getStage() {
        return (Stage) renameButton.getScene().getWindow();
    }

    // --- Browse buttons ---

    @FXML
    private void handleBrowseVideo(ActionEvent event) {
        chooseDirectoryIntoField(videoPathField);
    }

    @FXML
    private void handleBrowseMaa(ActionEvent event) {
        chooseDirectoryIntoField(maaPathField);
    }

    @FXML
    private void handleBrowseTipper(ActionEvent event) {
        chooseDirectoryIntoField(tipperPathField);
    }

    private void chooseDirectoryIntoField(TextField targetField) {
        DirectoryChooser chooser = new DirectoryChooser();
        chooser.setTitle("Select folder");
        File selected = chooser.showDialog(getStage());
        if (selected != null) {
            targetField.setText(selected.getAbsolutePath());
        }
    }

    // --- Rename Files button → go to Confirmation page ---

    @FXML
    private void handleRenameFiles(ActionEvent event) throws IOException {
        FXMLLoader loader = new FXMLLoader(
                getClass().getResource("/fxml/Confirmation.fxml")
        );
        Parent root = loader.load();

        // Get the confirmation controller and pass the paths
        ConfirmationController controller = loader.getController();
        controller.setPaths(
                videoPathField.getText(),
                maaPathField.getText(),
                tipperPathField.getText()
        );

        Stage stage = getStage();
        stage.setScene(new Scene(root));
        stage.show();
    }

    // --- Called when coming back from Confirmation via Edit button ---

    public void setInitialPaths(String video, String maa, String tipper) {
        if (video != null) {
            videoPathField.setText(video);
        }
        if (maa != null) {
            maaPathField.setText(maa);
        }
        if (tipper != null) {
            tipperPathField.setText(tipper);
        }
    }
}
