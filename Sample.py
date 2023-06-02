import col_of_def.dataset as dataset
import torch.utils.data as data
import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import os.path as osp

def make_datapath_list_for_Kodak(rootpath):
    # Create template for paths to image and annotation files
    imgpath_template = osp.join(rootpath, 'PNGImages', '%s.png')
    annopath_template = osp.join(rootpath, 'MaskImages', '%s.png')

    # Get ID (file name) of each file, training and validation
    val_id_names = osp.join(rootpath + 'ImageSets/mask.txt')

    # Create a list of paths to image files and annotation files for validation data
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()  # Strip blank spaces and line breaks
        img_path = (imgpath_template % file_id)  # Path of the image
        anno_path = (annopath_template % file_id)  # Path of annotation
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return val_img_list, val_anno_list

def prepare_dataset_Kodak(batch_size=1,rootpath = "./Kodak"):
    val_img_list, val_anno_list = dataset.make_datapath_list_for_Kodak(rootpath=rootpath)
    val_dataset = KodakDataset(val_img_list, val_anno_list, phase="test")

    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1)

    return val_dataloader,val_img_list

class KodakDataset(data.Dataset):

    def __init__(self, img_list, anno_list, phase="test"):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img, anno_class_img = self.pull_item(index)
        masked_image = torch.where((anno_class_img > 0), img, anno_class_img)
        maskdata = anno_class_img[0:1,:,:]
       
        images_with_alpha = torch.cat([masked_image, maskdata], dim=0)

        return masked_image, maskdata, img, anno_class_img, images_with_alpha
       

    def pull_item(self, index):
        transform = transforms.Compose([
        ])
        # 1. Load Image
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)   # [Height][Width][channnel]
        img = torchvision.transforms.functional.to_tensor(img)
        # 2. Load annotation
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)  
        anno_class_img = anno_class_img.convert("L").convert("RGB")
        anno_class_img = torchvision.transforms.functional.to_tensor(anno_class_img)
        img = transform(img)
        anno_class_img = transform(anno_class_img)
        return img, anno_class_img
    
if __name__ == "__main__":
    imglist,anolist = make_datapath_list_for_Kodak("./Kodak/")
    dataset = KodakDataset(imglist,anolist)
    img,ano,_,_,_ = dataset[3]
    img = torch.cat([img,ano],dim=0)
    torchvision.utils.save_image(img, "img.png")
    print(img.shape)