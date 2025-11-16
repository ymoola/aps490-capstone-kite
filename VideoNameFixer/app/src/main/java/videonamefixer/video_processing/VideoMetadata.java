package videonamefixer.video_processing;

import java.time.Instant;

public class VideoMetadata {
    private final Instant creationTime;

    public VideoMetadata(Instant creationTime) {
        this.creationTime = creationTime;
    }

    public Instant getCreationTime() {
        return creationTime;
    }
}
