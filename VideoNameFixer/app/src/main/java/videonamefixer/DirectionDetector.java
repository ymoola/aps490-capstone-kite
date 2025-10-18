package videonamefixer;

import org.opencv.core.*;
import org.opencv.video.Video;
import org.opencv.videoio.VideoCapture;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.List;

public class DirectionDetector {

    static { nu.pattern.OpenCV.loadLocally(); }

    public enum Direction { LEFT, RIGHT, UNDETERMINED }

    public static Direction detectMovement(String videoPath, int sampleStep) {
        VideoCapture cap = new VideoCapture(videoPath);
        if (!cap.isOpened()) {
            System.err.println("Cannot open: " + videoPath);
            return Direction.UNDETERMINED;
        }

        Mat prevGray = new Mat();
        Mat gray = new Mat();
        List<Double> meanDx = new ArrayList<>();
        int frameCount = 0;

        while (true) {
            Mat frame = new Mat();
            if (!cap.read(frame) || frame.empty()) break;
            frameCount++;
            if (frameCount % sampleStep != 0) continue;

            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
            if (!prevGray.empty()) {
                Mat flow = new Mat();
                Video.calcOpticalFlowFarneback(prevGray, gray, flow,
                        0.5, 3, 15, 3, 5, 1.2, 0);
                List<Mat> flowParts = new ArrayList<>(2);
                Core.split(flow, flowParts);
                Mat flowX = flowParts.get(0);
                Scalar meanX = Core.mean(flowX);
                meanDx.add(meanX.val[0]);
                flow.release(); flowX.release();
                flowParts.get(1).release();
            }
            gray.copyTo(prevGray);
        }

        cap.release(); prevGray.release(); gray.release();
        double avgDx = meanDx.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        if (Math.abs(avgDx) < 0.05) return Direction.UNDETERMINED;
        return avgDx > 0 ? Direction.RIGHT : Direction.LEFT;
    }
}
