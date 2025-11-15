package videonamefixer.video_processing;

import java.nio.file.Path;

public interface VideoProcessor {
    VideoProcessingResult process(Path videoFile);
}
