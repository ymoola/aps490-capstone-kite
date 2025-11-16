package videonamefixer;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.*;
import java.util.Comparator;
import java.util.Optional;
import java.util.stream.Stream;

import videonamefixer.video_processing.*;

public class Main {

    public static void main(String[] args) throws IOException {

        Path videosDir;

        // 1. If user passed a folder → use it
        if (args.length > 0) {
            videosDir = Paths.get(args[0]);

        } else {
            // 2. Default to src/test/resources/videonamefixer/Videos (relative to the :app module)
            Path base = Paths.get("VideoNameFixer/app/src/test/resources/videonamefixer/Videos");

            System.out.println("Base Videos path: " + base.toAbsolutePath());

            if (!Files.isDirectory(base)) {
                throw new IOException("Expected folder not found: " + base.toAbsolutePath());
            }

            // 3. Find newest *first-level* subfolder inside Videos/
            Path level1 = newestSubdirectory(base)
                    .orElseThrow(() ->
                            new IOException("No first-level subfolders found inside: " + base.toAbsolutePath()));

            System.out.println("Selected level 1 folder: " + level1.toAbsolutePath());

            // 4. Find newest *second-level* subfolder inside that folder (if any)
            Path level2 = newestSubdirectory(level1).orElse(level1);

            System.out.println("Selected level 2 folder (final videos dir): " + level2.toAbsolutePath());

            videosDir = level2;
        }

        Path csvOut = (args.length > 1)
                ? Paths.get(args[1])
                : Paths.get("VideoNameFixer/app/src/test/resources/video_processing_results.csv");

        // Ensure output folder exists
        if (csvOut.getParent() != null) {
            Files.createDirectories(csvOut.getParent());
        }

        // Build the video processor
        VideoProcessor processor = new MovementVideoProcessor(
                10,
                DirectionDetector.DEFAULT_NO_MOTION_THRESHOLD,
                DirectionDetector.DEFAULT_EARLY_SECONDS,
                DirectionDetector.DEFAULT_EARLY_WEIGHT
        );

        try (BufferedWriter writer = Files.newBufferedWriter(csvOut)) {
            writer.write("participant_id,file_name,direction");
            writer.newLine();

            VideoBatchRunner runner = new VideoBatchRunner(processor);
            runner.runBatch(videosDir, writer);
        }

        System.out.println("Done.");
    }

    private static Optional<Path> newestSubdirectory(Path parent) throws IOException {
        try (Stream<Path> stream = Files.list(parent)) {
            return stream
                    .filter(Files::isDirectory)
                    .max(Comparator.comparing(p -> p.toFile().lastModified()));
        }
    }
}
