import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
from PIL import Image





class ZooDatasets(Dataset):
    
    def __init__(self, mode='train', transform=None, data_path=None, batch_size=32):
        self.batch_size = batch_size
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.dataset_list = [
                'sldc_0_10',
                'corales_0_14',
                '6000-store-items-images-classified-by-color_imoore_0_12',
                'fcis-sc-deeplearning-competition_0_10',
                'lego-brick-images_joosthazelzet_0_16',
                'oregon-wildlife_virtualdvid_0_20',
                'perritos_0_10',
                'khu-deep-learning-competition_0_10',
                'plant-seedlings-classification_0_12',
                'skin-cancer9-classesisic_nodoubttome_0_9'
                
        ]

        '''self.dataset_list = [
                'corales_0_14',
                '6000-store-items-images-classified-by-color_imoore_0_12',
        ]'''
        
        self.curr_dataset = self.dataset_list[0]
        self.load_data()


              
        
    def get_loader(self, mode='train'):
        root_path = self.load_data()
        loader = ImageFolder(root=root_path, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]))

        loader = DataLoader(
            dataset=loader,
            batch_size=self.batch_size,
            shuffle=(mode == 'train'),
            num_workers=4
        )
        
        return loader
        
    def load_data(self):
        if self.mode == 'train':
            self.data_folder = os.path.join(self.data_path, f'{self.curr_dataset}/tr')
        elif self.mode == 'validation':
            self.data_folder = os.path.join(self.data_path, f'{self.curr_dataset}/va')
        elif self.mode == 'test':
            self.data_folder = os.path.join(self.data_path, f'{self.curr_dataset}/te')
            
            
        self.classes = sorted([d.name for d in os.scandir(self.data_folder) if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []
        
        for target_class in self.classes:
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(self.data_folder, target_class)
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    # Modify here to convert label to integer
                    label = self.class_to_idx[target_class]
                    self.samples.append((path, label))
        
        
        return self.data_folder
        
     
        
    def get_dataset_list(self):
        return self.dataset_list
    
    
    def curr(self):
        return self.curr_dataset
    
    def set_mode(self, mode):
        self.mode = mode
        
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
            

        return img, target
    
    
    def set_dataset(self, dataset):
        self.curr_dataset = dataset
        self.load_data()
            
    def get_nclss(self):
        return (len(self.classes))
    
    def get_clss(self):
        return (self.classes)
