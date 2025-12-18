import wandb
import gc
import os
import gymnasium as gym
from typing import Any, Callable, List, SupportsFloat
from gymnasium import error, logger
from gymnasium.core import ActType, ObsType, RenderFrame

class RecordVideo_TRAIN_GC(gym.wrappers.RecordVideo):

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        video_folder: str,
        episode_trigger: Callable[[int], bool] | None = None,
        step_trigger: Callable[[int], bool] | None = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        fps: int | None = None,
        disable_logger: bool = True,
        use_wandb: bool = False,
    ):
        super().__init__(
            env,
            video_folder,
            episode_trigger,
            step_trigger,
            video_length,
            name_prefix,
            fps,
            disable_logger,
        )

        self.use_wandb = use_wandb # 新增 wandb 选项


    # prevent out of memory
    # reference: https://github.com/isaac-sim/IsaacLab/issues/1996
    def stop_recording(self):

        ############################### Original Code ###############################
        assert self.recording, "stop_recording was called, but no recording was started"

        if len(self.recorded_frames) == 0:
            logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError as e:
                raise error.DependencyNotInstalled(
                    'MoviePy is not installed, run `pip install "gymnasium[other]"`'
                ) from e

            clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
            moviepy_logger = None if self.disable_logger else "bar"
            path = os.path.join(self.video_folder, f"{self._video_name}.mp4")
            clip.write_videofile(path, logger=moviepy_logger)

        self.recorded_frames = []
        self.recording = False
        ############################### Original Code ###############################
        if self.use_wandb:
            wandb.log({f"video/{self.name_prefix}": wandb.Video(path, format="mp4")})
        gc.collect()


class RecordVideo_EVAL_GC(gym.wrappers.RecordVideo):

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        video_folder: str,
        episode_trigger: Callable[[int], bool] | None = None,
        step_trigger: Callable[[int], bool] | None = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        fps: int | None = None,
        disable_logger: bool = True,
        use_wandb: bool = False,
    ):
        super().__init__(
            env,
            video_folder,
            episode_trigger,
            step_trigger,
            video_length,
            name_prefix,
            fps,
            disable_logger,
        )

        self.use_wandb = use_wandb # 新增 wandb 选项

    # prevent out of memory
    # reference: https://github.com/isaac-sim/IsaacLab/issues/1996
    def stop_recording(self, name_prefix: str, step: int):

        ############################### Original Code ###############################
        assert self.recording, "stop_recording was called, but no recording was started"

        if len(self.recorded_frames) == 0:
            logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError as e:
                raise error.DependencyNotInstalled(
                    'MoviePy is not installed, run `pip install "gymnasium[other]"`'
                ) from e

            clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
            moviepy_logger = None if self.disable_logger else "bar"
            path = os.path.join(self.video_folder, f"{name_prefix}_{step}.mp4")
            clip.write_videofile(path, logger=moviepy_logger)

        self.recorded_frames = []
        self.recording = False
        ############################### Original Code ###############################
        if self.use_wandb:
            wandb.log({f"video/{name_prefix}": wandb.Video(path, format="mp4")}, step=step)
        gc.collect()