package videonamefixer;

import org.opencv.core.*;
import org.opencv.video.Video;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.List;

public class DirectionDetector {

    static { nu.pattern.OpenCV.loadLocally(); }

    public enum Direction { LEFT, RIGHT, UNDETERMINED }

    public static final double DEFAULT_NO_MOTION_THRESHOLD = 0.005;
    public static final double DEFAULT_EARLY_SECONDS = 3.0;
    public static final double DEFAULT_EARLY_WEIGHT = 3.0;

    public static class DetectionResult {
        public final Direction direction;
        public final boolean error;
        public final String message;

        public DetectionResult(Direction direction, boolean error, String message) {
            this.direction = direction;
            this.error = error;
            this.message = message;
        }
    }

    public static DetectionResult detectMovementResult(String videoPath, int sampleStep) {
        return detectMovementResult(videoPath, sampleStep, DEFAULT_NO_MOTION_THRESHOLD,
                DEFAULT_EARLY_SECONDS, DEFAULT_EARLY_WEIGHT);
    }

    public static DetectionResult detectMovementResult(String videoPath, int sampleStep, double noMotionThreshold) {
        return detectMovementResult(videoPath, sampleStep, noMotionThreshold,
                DEFAULT_EARLY_SECONDS, DEFAULT_EARLY_WEIGHT);
    }

    public static DetectionResult detectMovementResult(String videoPath, int sampleStep,
                                                       double noMotionThreshold,
                                                       double earlySeconds,
                                                       double earlyWeight) {
        VideoCapture cap = new VideoCapture(videoPath);
        if (!cap.isOpened()) {
            System.err.println("Cannot open: " + videoPath);
            return new DetectionResult(Direction.UNDETERMINED, true, "Cannot open");
        }

        Mat prevGray = new Mat();
        Mat gray = new Mat();
        List<Double> meanDx = new ArrayList<>();
        int frameCount = 0;
        double weightedDxSum = 0.0;
        double weightSum = 0.0;
        double fps = cap.get(Videoio.CAP_PROP_FPS);
        if (fps <= 0 || Double.isNaN(fps) || Double.isInfinite(fps)) {
            fps = 30.0; // fallback if FPS is unavailable
        }

        try {
            while (true) {
                Mat frame = new Mat();
                if (!cap.read(frame) || frame.empty()) { frame.release(); break; }
                frameCount++;
                if (frameCount % sampleStep != 0) { frame.release(); continue; }

                Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
                // Release the original BGR frame as soon as we have grayscale
                frame.release();
                if (!prevGray.empty()) {
                    Mat flow = new Mat();
                    Video.calcOpticalFlowFarneback(prevGray, gray, flow,
                            0.5, 3, 15, 3, 5, 1.2, 0);
                    List<Mat> flowParts = new ArrayList<>(2);
                    Core.split(flow, flowParts);
                    Mat flowX = flowParts.get(0);
                    Scalar meanX = Core.mean(flowX);
                    double dx = meanX.val[0];
                    meanDx.add(dx);
                    double timeSec = frameCount / fps;
                    double w = (earlySeconds > 0 && timeSec <= earlySeconds) ? Math.max(1.0, earlyWeight) : 1.0;
                    weightedDxSum += dx * w;
                    weightSum += w;
                    flow.release();
                    flowX.release();
                    flowParts.get(1).release();
                }
                gray.copyTo(prevGray);
            }
        } finally {
            cap.release(); prevGray.release(); gray.release();
        }

        double avgDxWeighted = (weightSum > 0.0) ? (weightedDxSum / weightSum) : 0.0;
        if (Math.abs(avgDxWeighted) < noMotionThreshold) return new DetectionResult(Direction.UNDETERMINED, false, "Low motion");
        return new DetectionResult(avgDxWeighted > 0 ? Direction.RIGHT : Direction.LEFT, false, "OK");
    }

    public static Direction detectMovement(String videoPath, int sampleStep) {
        DetectionResult r = detectMovementResult(videoPath, sampleStep);
        return r.direction;
    }
}
