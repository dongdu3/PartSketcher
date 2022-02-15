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
                 cat='chair', mode='train', n_view=9, view_pick=False, test_path=None):
        self.data_root = data_root
        self.cat = cat
        self.mode = mode
        self.n_view = n_view
        self.view_pick = view_pick
        self.test_path = test_path
        if self.test_path:
            self.test_path = os.path.join(self.test_path, self.cat)

        self.list_file = os.path.join(self.data_root, 'split_file', self.cat + '_' + mode + '.txt')
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
        vox_data_batch = []
        pos_infor_batch = []
        part_name_list = []
        with open(os.path.join(data_path, 'part_list.txt'), 'r') as fin_part:
            for part_name in fin_part.readlines():
                part_name = part_name.strip()
                part_name_list.append(part_name)

                ass_sket_path = os.path.join(data_path, 'sketch', 'assembly_' + view_id + '.png')
                sket_path = os.path.join(data_path, 'sketch', part_name + '_' + view_id + '.png')

                # load view image data
                ass_sket = Image.open(ass_sket_path).convert('RGB')
                ass_sket_data = self.img_transform(ass_sket)

                part_sket = Image.open(sket_path).convert('RGB')
                part_sket_data = self.img_transform(part_sket)

                sket_data = torch.cat((ass_sket_data, part_sket_data), 0)
                sket_data = sket_data.unsqueeze(0)
                if sket_data_batch is None:
                    sket_data_batch = sket_data
                else:
                    sket_data_batch = torch.cat((sket_data_batch, sket_data), 0)

                # load part voxel
                if self.test_path is None:
                    fp = open(os.path.join(data_path, part_name + '.binvox'), 'rb')
                else:
                    fp = open(os.path.join(self.test_path, folder_name, part_name + '_' + view_id + '.binvox'), 'rb')
                vox_data = binvox_rw.read_as_3d_array(fp, fix_coords=True).data
                vox_size = vox_data.shape[0]
                vox_data = np.array(vox_data).reshape((1, vox_size, vox_size, vox_size))
                fp.close()
                vox_data_batch.append(vox_data)

                # load part position information in 128^3 space
                pos_infor = []
                with open(os.path.join(data_path, part_name + '_ori_infor.txt'), 'r') as fin_pos:
                    lines = fin_pos.readlines()
                    # bbox center
                    bbox_center = lines[0].strip().split(' ')
                    pos_infor.append(np.float32(bbox_center[1]) / 128.)
                    pos_infor.append(np.float32(bbox_center[2]) / 128.)
                    pos_infor.append(np.float32(bbox_center[3]) / 128.)
                    # bbox scale
                    bbox_scale = lines[2].strip().split(' ')
                    pos_infor.append(np.float32(bbox_scale[1]) / 128.)
                    fin_pos.close()
                pos_infor_batch.append(np.array(pos_infor, dtype=np.float32))
            fin_part.close()

        vox_data_batch = torch.from_numpy(np.array(vox_data_batch, dtype=np.float32)).type(torch.FloatTensor)
        pos_infor_batch = torch.from_numpy(np.array(pos_infor_batch, dtype=np.float32)).type(torch.FloatTensor)

        return sket_data_batch, vox_data_batch, pos_infor_batch, folder_name, part_name_list,  view_id

    def __len__(self):
        return len(self.data_paths)

if __name__ == '__main__':
    # from torchvision.utils import save_image

    dataset = PartNetAss(mode='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    for i, data in enumerate(dataloader, 0):
        sket, vox, pos, folder_name, part_name_list, view_id = data
        sket = sket.squeeze(0)
        vox = vox.squeeze(0)
        pos = pos.squeeze(0)
        print(sket.shape)
        print(vox.shape)
        # print(pos)
        print(pos.shape)
        print(folder_name)
        print(part_name_list)
        print(view_id)

        # save_image(sket.squeeze(0).cpu(), name[0].split('/')[-1] + '_' + view_id[0] + '.png')

        break
