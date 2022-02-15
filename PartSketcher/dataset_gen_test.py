import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import h5py
from utils.point_transforms import *

class PartNet(data.Dataset):
    def __init__(self,
                 data_root='/media/administrator/Data2/don/PartNet.v0/PartSketcher/dataset/data/',
                 cat='chair', mode='train', n_view=9, view_pick=False, n_pc_occ_subsample=2048):
        self.data_root = data_root
        self.cat = cat
        self.mode = mode
        self.n_view = n_view
        if mode == 'train':
            self.pc_occ_transform = SubsamplePoints(n_pc_occ_subsample)
        else:
            self.pc_occ_transform = None

        self.list_file = os.path.join(self.data_root, 'split_file', self.cat+'_part_'+mode+'.txt')
        # self.list_file = os.path.join(self.data_root, 'split_file', self.cat+'_'+mode+'.txt')
        fnames = open(self.list_file, 'r').readlines()
        fnames = [f.strip() for f in fnames]

        self.data_paths = []
        data_path = os.path.join(self.data_root, self.cat)
        for name in fnames:
            folder_name, part_name = name.split('/')
            pc_occ_path = os.path.join(data_path, name+'.h5')
            sket_view_path = os.path.join(data_path, folder_name, 'sketch_part_unit')
            if not view_pick:
                for i in range(0, self.n_view):
                    sket_path = os.path.join(sket_view_path, part_name+'_'+str(i)+'.png')
                    self.data_paths.append((sket_path, pc_occ_path, name, str(i)))
            else:
                sket_path = os.path.join(sket_view_path, part_name + '_5.png')
                self.data_paths.append((sket_path, pc_occ_path, name, '5'))

        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # # RandomResizedCrop or RandomCrop
        # self.dataAugmentation = transforms.Compose([
        #     transforms.RandomCrop(127),
        #     transforms.RandomHorizontalFlip(),
        # ])
        # self.validating = transforms.Compose([
        #     transforms.CenterCrop(127),
        # ])

    def __getitem__(self, index):
        # view image data
        sket_path, pc_occ_path, name, view_id = self.data_paths[index]
        sket = Image.open(sket_path).convert('RGB')
        sket_data = self.img_transform(sket)

        # point cloud with occupancy value data
        with h5py.File(pc_occ_path, 'r') as pc_occ_dict:
            points = pc_occ_dict['points'][:].astype(np.float32)
            values = pc_occ_dict['values'][:].astype(np.float32)

        occ_data = {
            None: points,
            'occ': values,
        }

        if self.pc_occ_transform is not None:
            occ_data = self.pc_occ_transform(occ_data)

        return sket_data, occ_data, name, view_id

    def __len__(self):
        return len(self.data_paths)

if __name__ == '__main__':
    # from torchvision.utils import save_image

    dataset = PartNet(mode='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    for i, data in enumerate(dataloader, 0):
        sket, occ, name, view_id = data
        print(sket.shape)
        print(occ[None])
        print(occ[None].shape)
        print(occ['occ'])
        print(occ['occ'].shape)
        print(name)
        print(view_id)

        # save_image(sket.squeeze(0).cpu(), name[0].split('/')[-1] + '_' + view_id[0] + '.png')

        break