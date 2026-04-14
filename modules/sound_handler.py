import numpy as np
import numpy.typing as npt
import sounddevice as sd
import soundfile as sf
from enum import Enum

class SoundHandler:
    class SoundMode(Enum):
        PLAYING = 1
        SILENCE = 2

    def __init__(self, sound_path: str):
        self.data, fs = sf.read(sound_path)
        self.cur_data = self.data * 0.7

        self.pos = 0
        self.mode = SoundHandler.SoundMode.SILENCE
        self.enabled = False

        self.cur_pan = 0.0
        self.prev_pan = 0.0

        delay_sec = 0.5
        self.silence_samples_total = int(delay_sec * fs)
        self.silence_remaining = 0

        self.sound_stream = sd.OutputStream(
            samplerate=fs,
            channels=2,
            blocksize=1024,
            callback=lambda outdata, frames, t, status: self.__callback(outdata, frames, t, status)
        )
        self.sound_stream.start()

    def __pan_sound(self):
        panned_data = self.data.copy()

        pan = np.clip(self.prev_pan, -1, 1)
        angle = (pan + 1) * (np.pi / 4)
        left_gain = np.cos(angle)
        right_gain = np.sin(angle)

        panned_data[:, 0] *= left_gain
        panned_data[:, 1] *= right_gain

        self.cur_data = panned_data
    
    def __callback(self, outdata: npt.NDArray, frames: int, time, status):
        outdata.fill(0)

        if not self.enabled and self.mode == SoundHandler.SoundMode.SILENCE:
            return
        
        if self.cur_pan != self.prev_pan and self.mode == SoundHandler.SoundMode.SILENCE:
            self.prev_pan = self.cur_pan
            self.__pan_sound()

        i = 0
        while i < frames:
            if self.mode == SoundHandler.SoundMode.PLAYING:
                remaining_audio = len(self.cur_data) - self.pos
                remaining_buffer = frames - i

                n = min(remaining_audio, remaining_buffer)

                outdata[i: i + n] = self.cur_data[self.pos: self.pos + n]

                self.pos += n
                i += n

                if self.pos >= len(self.cur_data):
                    self.pos = 0
                    self.mode = SoundHandler.SoundMode.SILENCE
                    self.silence_remaining = self.silence_samples_total

            elif self.mode == SoundHandler.SoundMode.SILENCE:
                remaining_buffer = frames - i
                n = min(self.silence_remaining, remaining_buffer)

                self.silence_remaining -= n
                i += n

                if self.silence_remaining <= 0:
                    self.mode = SoundHandler.SoundMode.PLAYING

    def play(self, pan: float = 0.0):
        if not self.enabled:
            self.pos = 0
            self.mode = SoundHandler.SoundMode.PLAYING if pan == self.prev_pan else SoundHandler.SoundMode.SILENCE
            self.silence_remaining = 0
        
        self.cur_pan = pan
        self.enabled = True
    
    def stop(self):
        self.enabled = False
