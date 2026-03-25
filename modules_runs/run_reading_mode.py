import cv2
import line_profiler
import numpy as np
import numpy.typing as npt
import threading

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from modules.reading_mode import ReadingMode

OCR_RUNNING = False
ocr_image: npt.NDArray = None

def reading_mode_thread(
    reading_mode: ReadingMode, 
    frame: npt.NDArray, 
):
    global ocr_image, OCR_RUNNING
    
    texts, scores, boxes = reading_mode.get_text(frame)
    ocr_image = reading_mode.draw_boxes(frame)
    
    print("\n-----PRINTING DETECTED TEXT-----")
    for text, score in zip(texts, scores):
        print(f"Text ({score:0.2f}): \"{text}\"")
    print("-------PRINTING FINISHED-------\n")
    
    OCR_RUNNING = False

@line_profiler.profile
def run_reading_mode():
    global OCR_RUNNING

    reading_mode = ReadingMode()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to capture frame!")
            break

        # scene_binary = model.get_scene_type_binary(frame)
        # label = f"{scene_binary}"

        output_image = np.hstack((frame, frame)) if ocr_image is None else np.hstack((frame, ocr_image))

        label = "RUNNING OCR" if OCR_RUNNING else ""
        cv2.putText(output_image, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
        cv2.putText(output_image, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Reading Mode", output_image)

        key_pressed = cv2.waitKey(1)
        if key_pressed == ord("q") or key_pressed == ord("Q"):
            break
        if key_pressed == ord("r") or key_pressed == ord("R"):
            OCR_RUNNING = True
            threading.Thread(target=reading_mode_thread, args=(reading_mode, frame), daemon=True).start()


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_reading_mode()
