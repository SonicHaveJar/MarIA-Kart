from utils.controller import KeyboardInputs, Agent
from utils.screen import capture, fps
import os
from time import sleep, time
import numpy as np
from time import sleep

keyboard = KeyboardInputs()
luigi = Agent('./data/models', 'cnn')

print(f"Waiting 5 seconds...")
sleep(5)
print("Running...")

view_=True
fps_=True

while 1:
    pause, exit_ = keyboard.shortcuts()

    if not pause:
        print("Running...")

        if fps_:
            last = time()
                    
        frame = capture(view=view_, fpss=last if fps_ else None)

        luigi.drive(frame, pause)

        if fps_ and not view_:
            print(fps(last))

    else:
        print("Paused.")

    if exit_:
        print("Close.")
        exit()
        break