package videonamefixer.video_processing;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.*;
import java.time.Duration;
import java.time.Instant;
import java.util.*;

public class VideoBatchRunner {

    private final VideoProcessor processor;

    public VideoBatchRunner(VideoProcessor processor) {
        this.processor = processor;
    }

    public void runBatch(Path videosDir, BufferedWriter writer) throws IOException {

        System.out.println("\n=== VideoBatchRunner START ===");
        System.out.println("Scanning directory: " + videosDir.toAbsolutePath());
        System.out.println();

        boolean containsSubfolders = false;
        boolean containsVideosDirectly = false;

        try (DirectoryStream<Path> ds = Files.newDirectoryStream(videosDir)) {
            for (Path p : ds) {
                if (Files.isDirectory(p)) {
                    containsSubfolders = true;
                } else if (p.toString().toLowerCase().endsWith(".mp4")) {
                    containsVideosDirectly = true;
                }
            }
        }

        if (!containsSubfolders && !containsVideosDirectly) {
            System.out.println("No videos or subfolders found. Exiting.");
            return;
        }

        // LEVEL-2 — folder IS a participant folder
        if (containsVideosDirectly) {
            String participantId = videosDir.getFileName().toString();
            System.out.println("> Level-2 participant folder: " + participantId);

            processParticipantFolder(videosDir, participantId, writer);

            System.out.println("=== VideoBatchRunner DONE ===");
            return;
        }

        // LEVEL-1 — iterate participant subfolders
        System.out.println("> Level-1 folder contains participants.");

        try (DirectoryStream<Path> participants =
                     Files.newDirectoryStream(videosDir, Files::isDirectory)) {

            for (Path participantDir : participants) {
                String participantId = participantDir.getFileName().toString();
                processParticipantFolder(participantDir, participantId, writer);
            }
        }

        System.out.println("\n=== VideoBatchRunner COMPLETE ===\n");
    }


    // ---------------------------------------------------------
    //  PROCESS A PARTICIPANT FOLDER
    // ---------------------------------------------------------
    private void processParticipantFolder(Path participantDir,
                                          String participantId,
                                          BufferedWriter writer) throws IOException {

        System.out.println("\n--- Processing participant: " + participantId + " ---");
        System.out.println("Folder: " + participantDir.toAbsolutePath());

        List<VideoWithMetadata> list = new ArrayList<>();

        try (DirectoryStream<Path> videos = Files.newDirectoryStream(
                participantDir,
                p -> Files.isRegularFile(p) && p.toString().toLowerCase().endsWith(".mp4"))) {

            for (Path video : videos) {
                VideoMetadata metadata = VideoMetadataReader.readMetadata(video.toFile());
                list.add(new VideoWithMetadata(video, metadata));
            }
        }

        if (list.isEmpty()) {
            System.out.println("No mp4 files found for: " + participantId);
            return;
        }

        // Sort by creation time
        list.sort(Comparator.comparing(v -> v.metadata().getCreationTime()));

        Instant prevTime = null;

        for (VideoWithMetadata v : list) {
            Instant t = v.metadata().getCreationTime();
            long deltaSeconds = (prevTime == null)
                    ? 0
                    : Duration.between(prevTime, t).getSeconds();

            System.out.println("\n  >>> PROCESSING VIDEO <<<");
            System.out.println("      file         = " + v.path().getFileName());
            System.out.println("      timestamp    = " + t);

            if (prevTime == null) {
                System.out.println("      previous     = (none – first video)");
            } else {
                System.out.println("      previous     = " + prevTime);
            }

            System.out.println("      deltaSeconds = " + deltaSeconds);

            VideoProcessingResult result = processor.process(v.path());
            System.out.println("      direction    = " + result.getLabel());

            // Mark missing metadata
            if (t.equals(Instant.EPOCH)) {
                System.out.println("      WARNING: Timestamp is EPOCH (metadata missing or unreadable)");
            }

            prevTime = t;

            writer.write(String.join(",",
                    csv(participantId),
                    csv(v.path().getFileName().toString()),
                    csv(result.getLabel()),
                    String.valueOf(deltaSeconds)
            ));
            writer.newLine();
        }


            System.out.println("Completed: " + participantId);
        }


    private static String csv(String s) {
        if (s == null) return "";
        boolean needsQuote =
                s.contains(",") || s.contains("\"") || s.contains("\n") || s.contains("\r");
        if (!needsQuote) return s;
        return '"' + s.replace("\"", "\"\"") + '"';
    }

    private record VideoWithMetadata(Path path, VideoMetadata metadata) {}
}
