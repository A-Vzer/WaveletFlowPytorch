from torch.utils.data import TensorDataset, DataLoader, Dataset
from os import listdir
from PIL import Image
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np


class ISIC(Dataset):
    def __init__(self, cf, benign=True, test=False, gray=False, standardize=True) -> None:
        super().__init__()
        self.cf = cf
        self.benign = benign
        self.gray = gray
        self.standardize = standardize
        if benign:
            dir = cf.path + "/benign/"
        else:
            dir = cf.path + "/malignant/"
        
        self.images = []

        for f in listdir(dir):
            temp1 = Image.open(dir + f)
            keep1 = temp1.copy()
            self.images.append(keep1)
            temp1.close()

        if benign and test:
            self.images = self.images[:300]
        elif benign and not test:
            self.images = self.images[300:]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):

        image = self.images[index]
        if not self.gray:
            image = np.array(image)
            if self.standardize:
                image = (image - np.array([204.59093244, 135.3067558, 139.64154845])) / np.array([33.20065213, 38.83154306, 42.72462019])
            image = np.uint8(image)
            image = transforms.functional.to_pil_image(image)
        image = transforms.functional.resize(image, tuple(self.cf.patch_size)) 

        if self.cf.augmentation:
            if self.cf.affineTransform:
                angle, translate, scale, shear = transforms.RandomAffine.get_params(degrees=self.cf.degrees,
                                    translate=self.cf.translate, scale_ranges=self.cf.scale_ranges, shears=self.cf.shears, img_size=[128, 128])
    
                image = TF.affine(image, angle, translate, scale, shear)
                
            
            if self.cf.imageAug:
                if not self.cf.brightness[0] == 1:
                    brightness = round(random.uniform(self.cf.brightness[0], self.cf.brightness[1]), 1)
                    image = transforms.functional.adjust_brightness(image, brightness)
                if not self.cf.gamma[0] == 1:
                    gamma = round(random.uniform(self.cf.gamma[0], self.cf.gamma[1]), 1)
                    image = transforms.functional.adjust_gamma(image, gamma)
        if self.gray:
            image = transforms.functional.to_grayscale(image)
        image = transforms.functional.to_tensor(image)
        return image
