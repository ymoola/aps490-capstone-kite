package RenamingAppCode.controllers;

import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.stage.Stage;
import javafx.scene.Scene;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import java.io.IOException;

public class WelcomeController {

    // Link to the button in FXML (fx:id)
    @FXML
    private Button nextButton;

    // Method called automatically when the button is clicked (onAction in FXML)
    @FXML
    private void gotoSelectPage() {
        try {

            System.out.println("Button clicked!");  // Verify button is linked
            // Load the next FXML page (Select Folder page)
            FXMLLoader loader = new FXMLLoader(getClass().getResource("/fxml/FolderSelection.fxml"));
            Parent root = loader.load();

            // Get the current stage from the button
            Stage stage = (Stage) nextButton.getScene().getWindow();

            // Set the new scene
            Scene scene = new Scene(root);
            stage.setScene(scene);
            stage.show();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}


