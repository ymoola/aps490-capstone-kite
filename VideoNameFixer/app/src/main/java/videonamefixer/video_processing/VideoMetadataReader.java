package videonamefixer.video_processing;

import com.drew.imaging.ImageMetadataReader;
import com.drew.metadata.Directory;
import com.drew.metadata.Metadata;
import com.drew.metadata.mp4.Mp4Directory;

import java.io.File;
import java.time.Instant;
import java.util.Date;
import java.util.TimeZone;

public class VideoMetadataReader {

    public static VideoMetadata readMetadata(File file) {

        Instant creation = Instant.EPOCH;

        try {
            Metadata metadata = ImageMetadataReader.readMetadata(file);

            // --- 1. Prefer MP4-specific directory ---
            for (Directory dir : metadata.getDirectories()) {

                // MP4 container creation time (highly reliable for GoPro)
                if (dir instanceof Mp4Directory mp4) {
                    if (mp4.containsTag(Mp4Directory.TAG_CREATION_TIME)) {
                        Date date = mp4.getDate(
                                Mp4Directory.TAG_CREATION_TIME,
                                TimeZone.getTimeZone("UTC")
                        );
                        if (date != null) {
                            return new VideoMetadata(date.toInstant());
                        }
                    }
                }

                // Fall back to looking for tag 256 in MP4-ish directories
                String name = dir.getName().toLowerCase();

                if (name.contains("mp4")) {
                    if (dir.containsTag(256)) {
                        Date date = dir.getDate(256, TimeZone.getTimeZone("UTC"));
                        if (date != null) {
                            return new VideoMetadata(date.toInstant());
                        }
                    }
                }
            }

        } catch (Exception e) {
            System.err.println("Error reading metadata for " + file.getAbsolutePath() +
                    ": " + e.getMessage());
        }

        // Fallback if nothing found
        return new VideoMetadata(creation);
    }
}
