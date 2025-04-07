"""Wrapper for recording videos."""
# taken from gym.wrappers.RecordVideo but with some modifications for wandb logging

import os
import wandb
import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


class VideoRecorder:

    def __init__(
            self,
            env,
            video_folder: str,
            video_interval: 100,
            video_length: int = 0,
            video_prefix: str = "video",
            video_fps: int = 30,
            wandb: bool = False,
            enabled: bool = False,
            save_video: bool = False,
    ):
        if env.render_mode in {None, "human", "ansi", "ansi_list"} and enabled:
            raise ValueError(
                f"Render mode is {env.render_mode}, which is incompatible with"
                f" RecordVideo. Initialize your environment with a render_mode"
                f" that returns an image, such as rgb_array."
            )
        self.env = env
        self.wandb = wandb

        self.video_folder = os.path.abspath(video_folder)
        os.makedirs(self.video_folder, exist_ok=True)

        self.video_prefix = video_prefix
        self.video_length = video_length
        self.video_fps = video_fps
        self.video_interval = video_interval
        self.enabled = enabled
        self.video_id = 0
        self.recording = False
        self.recorded_frames = []
        self.save_video = save_video

    def start_video_recorder(self, iter_id):
        """Starts video recorder using :class:`video_recorder.VideoRecorder`."""
        self.close_video_recorder()
        self.video_id = iter_id
        self.capture_frame()
        self.recording = True

    def capture_frame(self):
        frame = self.env.unwrapped.render()
        self.recorded_frames.append(frame)

    def step(self, iter_id):
        if not self.enabled:
            return
        if self.recording:
            self.capture_frame()
            if (iter_id - self.video_id) >= self.video_length:
                self.close_video_recorder(iter_id)

        else:
            self.start_video_recorder(iter_id)

    def close_video_recorder(self, iter_id=None):
        """Closes the video recorder if currently recording."""
        if self.recording and iter_id is not None:
            if self.save_video:
                video_name = f"{self.video_prefix}-step-{self.video_id}"

                base_path = os.path.join(self.video_folder, video_name)
                video_path = base_path + ".mp4"
                clip = ImageSequenceClip(self.recorded_frames, fps=self.video_fps)
                clip.write_videofile(video_path, logger=None)
            if self.wandb:
                # adding wandb step here will not work since the video is not logged immediately...
                wandb.log({f"{self.video_prefix}": wandb.Video(np.array(self.recorded_frames).transpose(0, 3, 1, 2),
                                                               fps=self.video_fps, format="mp4")}, step=self.video_id)
            self.video_id = None

        self.recording = False
        self.recorded_frames = []

    def close(self):
        self.close_video_recorder()
