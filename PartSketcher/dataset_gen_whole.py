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
                 cat='chair', mode='test', n_view=9, view_pick=True):
        self.data_root = data_root
        self.cat = cat
        self.mode = mode
        self.n_view = n_view
        self.view_pick = view_pick

        self.list_file = os.path.join(self.data_root, 'split_file', self.cat+'_'+mode+'.txt')
        fnames = open(self.list_file, 'r').readlines()
        self.data_paths = [f.strip() for f in fnames]

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
        folder_name = self.data_paths[index]
        data_path = os.path.join(self.data_root, self.cat, folder_name)
        # view_id = '3'
        view_id = '5'
        if not self.view_pick:
            view_id = str(np.random.randint(0, self.n_view))

        sket_data_batch = None
        part_name_list = []
        with open(os.path.join(data_path, 'part_list.txt'), 'r') as fin_part:
            for part_name in fin_part.readlines():
                part_name = part_name.strip()
                part_name_list.append(part_name)

                # view image data
                sket = Image.open(os.path.join(data_path, 'sketch_part_unit', part_name + '_' + view_id + '.png')).convert('RGB')
                sket_data = self.img_transform(sket).unsqueeze(0)
                if sket_data_batch is None:
                    sket_data_batch = sket_data
                else:
                    sket_data_batch = torch.cat((sket_data_batch, sket_data), 0)

                # # point cloud with occupancy value data
                # with h5py.File(os.path.join(data_path, part_name + '.h5'), 'r') as pc_occ_dict:
                #     points = pc_occ_dict['points'][:].astype(np.float32)
                #     values = pc_occ_dict['values'][:].astype(np.float32)
                #
                # occ_data = {
                #     None: points,
                #     'occ': values,
                # }

        # part_name_list = np.array(part_name_list)

        return sket_data_batch, folder_name, part_name_list, view_id

    def __len__(self):
        return len(self.data_paths)

if __name__ == '__main__':
    # from torchvision.utils import save_image

    dataset = PartNet(mode='test_example')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    for i, data in enumerate(dataloader, 0):
        sket, folder_name, part_name_list, view_id = data
        sket = sket.squeeze(0)
        print(sket.shape)
        folder_name = folder_name[0]
        view_id = view_id[0]
        print(folder_name)
        print(part_name_list)
        print(view_id)

        # save_image(sket.squeeze(0).cpu(), name[0].split('/')[-1] + '_' + view_id[0] + '.png')

        break