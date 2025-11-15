package videonamefixer.video_processing;

public class VideoProcessingResult {
    private final boolean success;
    private final String label;
    private final String message;

    public VideoProcessingResult(boolean success, String label, String message) {
        this.success = success;
        this.label = label;
        this.message = message;
    }

    public boolean isSuccess() { return success; }
    public String getLabel() { return label; }
    public String getMessage() { return message; }
}
