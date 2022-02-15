import os
import numpy as np
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from PIL import Image
from utils import binvox_rw

class PartNetAss(data.Dataset):
    def __init__(self,
                 data_root='/media/administrator/Data2/don/PartNet.v0/PartSketcher/dataset/data/',
                 cat='chair', mode='train', n_view=9, view_pick=False, data_gen_root=None):
        self.data_root = data_root
        self.cat = cat
        self.mode = mode
        self.n_view = n_view
        self.view_pick = view_pick
        self.data_gen_root = data_gen_root
        if self.data_gen_root:
            self.data_gen_root = os.path.join(self.data_gen_root, self.cat)

        self.list_file = os.path.join(self.data_root, 'split_file', self.cat + '_part_' + mode + '.txt')
        fnames = open(self.list_file, 'r').readlines()
        fnames = [f.strip() for f in fnames]

        self.data_paths = []
        data_path = os.path.join(self.data_root, self.cat)
        for name in fnames:
            folder_name, part_name = name.split('/')
            pos_infor_path = os.path.join(data_path, name + '_ori_infor.txt')
            if data_gen_root:
                vox_path = os.path.join(self.data_gen_root, name)
            else:
                vox_path = os.path.join(data_path, name + '.binvox')
            sket_view_path = os.path.join(data_path, folder_name, 'sketch')
            self.data_paths.append((sket_view_path, vox_path, pos_infor_path, folder_name, part_name))

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
        sket_view_path, vox_path, pos_infor_path, folder_name, part_name = self.data_paths[index]
        view_id = '5'
        if not self.view_pick:
            view_id = str(np.random.randint(0, self.n_view))
        ass_sket_path = os.path.join(sket_view_path, 'assembly_' + view_id + '.png')
        sket_path = os.path.join(sket_view_path, part_name + '_' + view_id + '.png')

        # load view image data
        ass_sket = Image.open(ass_sket_path).convert('RGB')
        ass_sket_data = self.img_transform(ass_sket)

        part_sket = Image.open(sket_path).convert('RGB')
        part_sket_data = self.img_transform(part_sket)

        sket_data = torch.cat((ass_sket_data, part_sket_data), 0)

        # load part voxel
        if not self.data_gen_root:
            fp = open(os.path.join(vox_path), 'rb')
        else:
            fp = open(os.path.join(vox_path + '_' + view_id + '.binvox'), 'rb')

        vox_data = binvox_rw.read_as_3d_array(fp, fix_coords=True).data
        vox_size = vox_data.shape[0]
        vox_data = np.array(vox_data).reshape((1, vox_size, vox_size, vox_size))
        vox_data = torch.from_numpy(vox_data).type(torch.FloatTensor)
        fp.close()

        # load part position information in 128^3 space
        pos_infor = []
        with open(pos_infor_path, 'r') as fin:
            lines = fin.readlines()
            # bbox center
            bbox_center = lines[0].strip().split(' ')
            pos_infor.append(np.float32(bbox_center[1]) / 128.)
            pos_infor.append(np.float32(bbox_center[2]) / 128.)
            pos_infor.append(np.float32(bbox_center[3]) / 128.)
            # bbox scale
            bbox_scale = lines[2].strip().split(' ')
            pos_infor.append(np.float32(bbox_scale[1]) / 128.)
            fin.close()
        pos_infor = torch.from_numpy(np.array(pos_infor, dtype=np.float32)).type(torch.FloatTensor)

        return sket_data, vox_data, pos_infor, folder_name, part_name, view_id

    def __len__(self):
        return len(self.data_paths)

if __name__ == '__main__':
    # from torchvision.utils import save_image

    dataset = PartNetAss(mode='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    for i, data in enumerate(dataloader, 0):
        sket, vox, pos, folder_name, part_name, view_id = data
        print(sket.shape)
        print(vox.shape)
        print(pos)
        print(pos.shape)
        print(folder_name)
        print(part_name)
        print(view_id)

        # save_image(sket.squeeze(0).cpu(), name[0].split('/')[-1] + '_' + view_id[0] + '.png')

        break
