package videonamefixer.video_processing;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;

public class VideoBatchRunner {

    private final VideoProcessor processor;

    public VideoBatchRunner(VideoProcessor processor) {
        this.processor = processor;
    }

    public void runBatch(Path videosDir, BufferedWriter writer) throws IOException {
        try (DirectoryStream<Path> participants = Files.newDirectoryStream(videosDir, Files::isDirectory)) {
            for (Path participantDir : participants) {
                String participantId = participantDir.getFileName().toString();

                try (DirectoryStream<Path> videos = Files.newDirectoryStream(
                        participantDir,
                        p -> Files.isRegularFile(p) && p.toString().toLowerCase().endsWith(".mp4"))) {

                    for (Path video : videos) {
                        VideoProcessingResult result = processor.process(video);

                        writer.write(String.join(",",
                                csv(participantId),
                                csv(video.getFileName().toString()),
                                csv(result.getLabel())
                        ));
                        writer.newLine();
                    }
                }
            }
        }
    }

    private static String csv(String s) {
        if (s == null) return "";
        boolean needsQuote = s.contains(",") || s.contains("\"") || s.contains("\n") || s.contains("\r");
        if (!needsQuote) return s;
        return '"' + s.replace("\"", "\"\"") + '"';
    }
}
