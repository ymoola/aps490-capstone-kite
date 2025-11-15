package videonamefixer.video_processing;

import java.nio.file.Path;

import videonamefixer.DirectionDetector;

public class MovementVideoProcessor implements VideoProcessor {

    private final int sampleStep;
    private final double noMotionThreshold;
    private final double earlySeconds;
    private final double earlyWeight;

    public MovementVideoProcessor(int sampleStep, double noMotionThreshold,
                                  double earlySeconds, double earlyWeight) {
        this.sampleStep = sampleStep;
        this.noMotionThreshold = noMotionThreshold;
        this.earlySeconds = earlySeconds;
        this.earlyWeight = earlyWeight;
    }

    @Override
    public VideoProcessingResult process(Path videoFile) {
        try {
            DirectionDetector.DetectionResult r =
                    DirectionDetector.detectMovementResult(
                            videoFile.toString(),
                            sampleStep,
                            noMotionThreshold,
                            earlySeconds,
                            earlyWeight
                    );

            return new VideoProcessingResult(
                    !r.error,
                    r.direction.name(),
                    r.message
            );

        } catch (Throwable t) {
            return new VideoProcessingResult(false, "error", t.getMessage());
        }
    }
}
