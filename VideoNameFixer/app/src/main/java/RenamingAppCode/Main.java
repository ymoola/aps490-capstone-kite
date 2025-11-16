package RenamingAppCode;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;
import javafx.scene.Parent;

public class Main extends Application {

    @Override
    public void start (Stage primaryStage){
        try {
             // Load your first FXML file (the welcome page)
            FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource("/RenamingAppCode/src/com/renamingapp/fxml/welcome.fxml"));
            Parent root = (Parent)fxmlLoader.load();
            Scene scene = new Scene(root);

             // Set window title
            primaryStage.setTitle("File Renaming App");
            // Set and show the scene
            primaryStage.setScene(scene);
            primaryStage.show();
        } catch (Exception e) {
            e.printStackTrace();
        }
    } 


    public static void main(String[] args) {
        launch(args); // Launches JavaFX application
    }
    
}


