# %%
import os

from glob import glob
from PIL import Image
import numpy as np


import torch
from torch.utils.data import Dataset

# %%
class DatasetLoader(Dataset): #이미지 데이터 추출
    def __init__(self, input_dir, label_dir, image_size): #변수지정
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.image_size = image_size

        #파일리스트 가져오기
        self.input_images = sorted(os.listdir(input_dir))
        self.label_images = sorted(os.listdir(label_dir))

    @classmethod #데코레이터 상속 자식 클래스 가져옴
    def preprocess(cls, pil_img, image_size):
        w, h = pil_img.size
        if not image_size == w:
            pil_img = pil_img.resize((image_size, image_size))

        img_nd = np.array(pil_img)  #배열생성

        #z축 차원 생성
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = (img_trans / 127.5) - 1

        return img_trans

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, index):
        image_name = self.input_images[index] #input_images 가져옴
        label_name = self.label_images[index]

        input_file = glob(self.input_dir + image_name)
        label_file = glob(self.label_dir + label_name)

        input_image = Image.open(input_file[0]).convert('L')
        label_image = Image.open(label_file[0]).convert('L')

        input_image = self.preprocess(input_image, self.image_size)
        label_image = self.preprocess(label_image, self.image_size)
        # print('inputshape :', input_image.shape)
        # print('labelshape :', label_image.shape)

        #tensor 로 변경
        return {
            "input_image": torch.from_numpy(input_image).type(torch.FloatTensor),
            "label_image": torch.from_numpy(label_image).type(torch.FloatTensor),
        }



