from os import PRIO_PGRP
from tkinter import W
import cv2
import glob
import numpy as np

import cv2
import numpy as np
from torch.tensor import Tensor
from torch.utils.data.dataloader import Dataset
from torchvision import transforms
import glob

from torchvision.transforms.transforms import ToPILImage


class LMK3DDataSet(Dataset):
    def __init__(self,size) -> None:
        super().__init__()
        self.image_lst = glob.glob(
            '/home/ZHEQIUSHUI/Pictures/landmark_data_base/landmark_dataset/*/images/*')

        self.transform = transforms.Compose([
            ToPILImage(),
            transforms.Resize((size, size)),
            transforms.RandomGrayscale(0.5),
            transforms.RandomAdjustSharpness(3),
            transforms.RandomAutocontrast(),
            # transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        img_path = self.image_lst[index]
        label_path = img_path.replace(
            '/images/', '/labels/').replace('.jpg', '.npy')
        label = np.load(label_path, allow_pickle=True)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        lmk = label.item().get('lmk')
        angles = label.item().get('angle')

        lmk /= w
        lmk = (lmk-0.5)/0.5

        angles = angles / 180*3.14

        img = self.transform(img)

        lmk = Tensor(lmk)
        angles = Tensor(angles)
        return img, lmk, angles

    def __len__(self):
        return len(self.image_lst)


if __name__ == "__main__":
    dataset = LMK3DDataSet()

    print(dataset.__len__())
    print(dataset.__getitem__(64287))
