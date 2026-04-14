if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from modules.sound_handler import SoundHandler

def run_audio_handler():
    sound_path = "audio/beep.wav"
    sound_handler = SoundHandler(sound_path)

    retry_keys = { "r", "R" }

    while True:
        input("\nPress Enter to start: ")
        sound_handler.play()
        
        while True:
            stop_str = input("Enter nothing (to stop) or pan [-1, 1]: ")
            if stop_str == "":
                break
            try:
                pan = float(stop_str)
                sound_handler.play(pan)
            except:
                pass
        
        sound_handler.stop()

        c = input("Press R to retry: ")
        if c not in retry_keys:
            break

if __name__ == "__main__":
    run_audio_handler()
