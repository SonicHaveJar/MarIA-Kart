import numpy as np
import cv2
from mss import mss
from PIL import ImageFont, ImageDraw, Image
from time import time

# screen_size = (1920, 1080)
# gta_window_size = (800, 600)
# capture_size = (306, 173)

# top = int((screen_size[1]/2)-(capture_size[1]/2))
# left = int((screen_size[0]/2)-(capture_size[0]/2))

def fps(last_time):
    time_elapsed =  time() - last_time

    fps = 1/time_elapsed

    return fps

mon = {'top': (559), 'left': (551), 'width': 306, 'height': 173} #abs position= x1:551, y1:559, x2:856, y2:731

sct = mss()

def capture(view=False, fpss=None):
    frame = sct.grab(mon)

    frame = np.array(Image.frombytes("RGB", frame.size, frame.bgra, "raw", "BGRX"))
    
    frame_rgb = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    fps_frame = frame_rgb

    if fpss:
        pil_im = Image.fromarray(fps_frame)

        draw = ImageDraw.Draw(pil_im)

        font = ImageFont.truetype("C:\Windows\Fonts\Arial.ttf", 25)

        draw.text((0, 0), f"FPS: {round(fps(fpss))}", font=font, fill=(0,255,255))
        
        fps_frame = np.array(pil_im)

    if view:
        cv2.imshow('Game view', fps_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    return frame_rgb
    