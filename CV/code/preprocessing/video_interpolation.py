import os
import cv2
from ccvfi import AutoModel, ConfigType
from tqdm import tqdm
import torch



class RIFEVideoInterpolator:
    """
    Handles loading a pretrained RIFE model and interpolating
    full videos to a higher FPS using in-memory frame interpolation.
    """

    def __init__(self):
        """
        Loads the pretrained RIFE model from the cloud (via ccvfi).
        This happens once per process.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModel.from_pretrained(
            pretrained_model_name=ConfigType.RIFE_IFNet_v426_heavy,
            device=device
        )

    def interpolate_video_to_file(
        self,
        input_video: str,
        output_video: str,
        fps_multiplier: int = 2
    ):
        """
        Interpolates an entire video using RIFE and writes it to disk.

        Args:
            input_video: path to original video
            output_video: path to temporary interpolated video
            fps_multiplier: 2 = double FPS (default)
        """

        input_video = os.path.abspath(input_video)
        output_video = os.path.abspath(output_video)
        os.makedirs(os.path.dirname(output_video), exist_ok=True)

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_video}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

        out_fps = fps * fps_multiplier
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video, fourcc, out_fps, (width, height))

        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            out.release()
            raise RuntimeError("Failed to read first frame")

        # tqdm tracks frame-pairs (N-1 interpolation steps)
        total_steps = (frames_total - 1) if frames_total else None

        with tqdm(
            total=total_steps,
            desc="RIFE interpolation",
            unit="frame-pair",
            leave=False
        ) as pbar:

            while True:
                ret, curr_frame = cap.read()
                if not ret:
                    break

                # Write original frame
                out.write(prev_frame)

                # Generate interpolated frame(s)
                if fps_multiplier == 2:
                    mid = self.model.inference_image_list(
                        img_list=[prev_frame, curr_frame]
                    )[0]
                    out.write(mid)
                else:
                    frames = self.model.inference_image_list(
                        img_list=[prev_frame, curr_frame],
                        num=fps_multiplier - 1
                    )
                    for f in frames:
                        out.write(f)

                prev_frame = curr_frame
                pbar.update(1)

        # Write final frame
        out.write(prev_frame)

        cap.release()
        out.release()