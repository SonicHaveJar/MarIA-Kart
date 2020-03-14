from utils.controller import KeyboardInputs
from utils.screen import capture, fps
import cv2
import csv
import os
from time import sleep, time
import numpy as np
import pandas as pd

import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Parameters.')
parser.add_argument("--view", type=str2bool, nargs='?', const=True, default=False, help='View captured data.', required=True)
parser.add_argument("--fps", type=str2bool, nargs='?', const=True, default=False, help='See FPS.', required=True)
args = parser.parse_args()

keyboard = KeyboardInputs()

path = './data/'
imgs_path = path + 'imgs/'

models_path = path + 'models/'

file = path + 'data.csv'

if not os.path.exists(path):
    os.makedirs(path)
    os.makedirs(imgs_path)
    os.makedirs(models_path)

elif not os.path.exists(imgs_path):
    os.makedirs(imgs_path)

elif not os.path.exists(models_path):
    os.makedirs(models_path)

if not os.path.exists(file):
    csv_file = open(file, 'a')
    writer = csv.writer(csv_file)
    writer.writerow(['img', 'x', 'left', 'right'])
    i = 0
else:
    csv_file = open(file, 'a')
    writer = csv.writer(csv_file)

    df = pd.read_csv(file, names=['img', 'x', 'left', 'right'])
    i = len(np.array(df['img']))-1

print(f"Waiting 5 seconds... (Start with image {i})")
sleep(5)
print("Running...")
print(args.view)
while 1:
    pause, exit_ = keyboard.shortcuts()

    if not pause:
        print("Running...")

        if args.fps:
            last = time()
            
        k_input = keyboard.read()
        
        frame = capture(view=args.view, fpss=last if args.fps else None)

        frame_path = f"{imgs_path}{i}.jpg"
        cv2.imwrite(frame_path, frame)

        data = [frame_path, *k_input]

        writer.writerow(data)

        if args.fps and not args.view:
            print(fps(last))

        i+=1

    else:
        print("Paused.")

    if exit_:
        print("Close.")
        csv_file.close()
        exit()
        break