import threading
import keyboard
from PIL import Image
import numpy as np
import torch
import os

class KeyboardInputs():
    def __init__(self):

        self.x = 0

        self.left = 0
        self.right = 0

        self.pause = False
        self.exit = False

        self.monitor_thread = threading.Thread(target=self.monitor, args=())
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def read(self):
        return [self.x, self.left, self.right]
    
    def shortcuts(self):
        return self.pause, self.exit

    def monitor(self):
        while True:
            
            if keyboard.is_pressed('x'):
                self.x = 1
            else:
                self.x = 0

            if keyboard.is_pressed('left'):
                self.left = 1
            else:
                self.left = 0
            if keyboard.is_pressed('right'):
                self.right = 1
            else:
                self.right = 0

            if keyboard.is_pressed('ctrl+p'):
                self.pause = not self.pause

            if keyboard.is_pressed('ctrl+e'):
                self.exit = True

class Agent():
    def __init__(self, path, name):
        from .models import CLSTM, resnet18
        
        from torchvision import transforms

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if name.lower() == 'clstm':
            self.model = CLSTM
            self.model.load_state_dict(torch.load(os.path.join(os.path.abspath(path), "clstm.pth")))
            self.model.to(self.device)
            self.model.eval()
        else:
            self.model = resnet18
            self.model.load_state_dict(torch.load(os.path.join(os.path.abspath(path), "cnn.pth")))
            self.model.to(self.device)
            self.model.eval()
        

        self.pause = False

        # self.update_thread = threading.Thread(target=self.update, args=())
        # self.update_thread.daemon = True
        # self.update_thread.start()


    def drive(self, frame, pause):
        self.pause = pause

        frame = self.transform(frame)

        frame_expanded = frame.unsqueeze_(0)

        output = self.model(frame_expanded.to(self.device))

        clipped_output = torch.clamp(output[0], min=0, max=1)

        keys = list(map(lambda x: 0 if x <= 0.5 else 1, clipped_output))

        print(keys)
    #     prediction = np.argmax(self.model.predict(np.array([frame]))[0])
        
    #     if not self.pause:
    #         PressKey(W)

    #         if prediction == 0:
    #             if random.randrange(0,3) == 1:
    #                 PressKey(W)
    #             else:
    #                 ReleaseKey(W)
    #             PressKey(A)
    #             ReleaseKey(D)

    #         if prediction == 1:
    #             PressKey(W)
    #             ReleaseKey(A)
    #             ReleaseKey(D)
    #             print('Nothing')

    #         if prediction == 2:
    #             if random.randrange(0,3) == 1:
    #                 PressKey(W)
    #             else:
    #                 ReleaseKey(W)
    #             PressKey(D)
    #             ReleaseKey(A)
    #             ReleaseKey(S)
    #             print('Right')
            
    #     else:
    #         ReleaseKey(W)
    #         ReleaseKey(D)
    #         ReleaseKey(A)
    
    # def update(self):
    #     while not self.pause:
    #         sleep(0.2)
    #         #ReleaseKey(W)
    #         ReleaseKey(D)
    #         ReleaseKey(A)
