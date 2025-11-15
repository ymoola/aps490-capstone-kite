package videonamefixer;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import videonamefixer.video_processing.MovementVideoProcessor;
import videonamefixer.video_processing.VideoBatchRunner;
import videonamefixer.video_processing.VideoProcessor;

public class VideoDirectionBatch {

    public static void main(String[] args) throws IOException {
        Path videosDir = Paths.get(args.length > 0 ? args[0] : "videos");
        Path csvOut    = Paths.get(args.length > 1 ? args[1] : "results.csv");

        VideoProcessor processor = new MovementVideoProcessor(
                /* sampleStep */ 10,
                DirectionDetector.DEFAULT_NO_MOTION_THRESHOLD,
                DirectionDetector.DEFAULT_EARLY_SECONDS,
                DirectionDetector.DEFAULT_EARLY_WEIGHT
        );

        if (csvOut.getParent() != null) {
            Files.createDirectories(csvOut.getParent());
        }

        try (BufferedWriter writer = Files.newBufferedWriter(csvOut)) {
            writer.write("participant_id,file_name,direction");
            writer.newLine();

            VideoBatchRunner runner = new VideoBatchRunner(processor);
            runner.runBatch(videosDir, writer);
        }

        System.out.println("Done.");
    }
}

