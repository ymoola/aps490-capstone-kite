package videonamefixer;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class VideoDirectionBatch {

    private static String csv(String s) {
        if (s == null) return "";
        boolean needsQuote = s.contains(",") || s.contains("\"") || s.contains("\n") || s.contains("\r");
        if (!needsQuote) return s;
        return '"' + s.replace("\"", "\"\"") + '"';
    }

    public static void main(String[] args) throws IOException {
        String videosDirArg = args.length >= 1 ? args[0] : "videos";
        String outputCsvArg = args.length >= 2 ? args[1] : "direction_results.csv";
        int sampleStep = 10; // increase default for faster runtime
        double noMotionThreshold = videonamefixer.DirectionDetector.DEFAULT_NO_MOTION_THRESHOLD;
        if (args.length >= 3) {
            try {
                sampleStep = Integer.parseInt(args[2]);
            } catch (NumberFormatException ignored) {}
        }
        if (args.length >= 4) {
            try {
                noMotionThreshold = Double.parseDouble(args[3]);
            } catch (NumberFormatException ignored) {}
        }

        Path videosDir = Paths.get(videosDirArg).toAbsolutePath().normalize();
        Path outputCsv = Paths.get(outputCsvArg).toAbsolutePath().normalize();

        if (!Files.isDirectory(videosDir)) {
            System.err.println("Videos directory not found: " + videosDir);
            System.exit(2);
        }

        System.out.println("Scanning participants in: " + videosDir);
        System.out.println("Writing results to: " + outputCsv);
        System.out.println("Using sampleStep: " + sampleStep);
        System.out.println("No-motion threshold: " + noMotionThreshold);

        // Ensure parent directories for output exist
        if (outputCsv.getParent() != null) {
            Files.createDirectories(outputCsv.getParent());
        }

        try (BufferedWriter writer = Files.newBufferedWriter(outputCsv)) {
            writer.write("participant_id,file_name,direction");
            writer.newLine();

            try (DirectoryStream<Path> participants = Files.newDirectoryStream(videosDir, Files::isDirectory)) {
                for (Path participantDir : participants) {
                    String participantId = participantDir.getFileName().toString();

                    try (DirectoryStream<Path> videos = Files.newDirectoryStream(participantDir, p ->
                            Files.isRegularFile(p) && p.getFileName().toString().toLowerCase().endsWith(".mp4"))) {
                        for (Path video : videos) {
                            String fileName = video.getFileName().toString();
                            String resultLabel;
                            try {
                                DirectionDetector.DetectionResult res = DirectionDetector.detectMovementResult(video.toString(), sampleStep, noMotionThreshold);
                                resultLabel = res.error ? "error processing" : res.direction.name();
                            } catch (Throwable t) {
                                System.err.println("Error processing " + video + ": " + t.getMessage());
                                resultLabel = "error processing";
                            }

                            writer.write(csv(participantId) + "," + csv(fileName) + "," + csv(resultLabel));
                            writer.newLine();
                            writer.flush();
                            System.out.println("Processed: " + participantId + "/" + fileName + " -> " + resultLabel);
                        }
                    }
                }
            }
        }

        System.out.println("Done.");
    }
}
