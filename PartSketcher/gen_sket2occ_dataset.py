import argparse
import torch.backends.cudnn as cudnn
import os
from PIL import Image
from model import *
from common import *
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--sketRoot', type=str, default='/data/dudong/PartSketcher/', help='sketch data root path')
parser.add_argument('--saveRoot', type=str, default='/data/dudong/PartSketcher/data_gen', help='binvox data root path to save')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--nView', type=int, default=9, help='view number of sketch')
parser.add_argument('--thres', type=float, default=0.2, help='threshold for occupancy estimation and mesh extraction')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--model', type=str, default='checkpoint', help='model path')
parser.add_argument('--modelName', type=str, default='generator.pt', help='model name')
parser.add_argument('--cat', type=str, default='chair')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--cuda', type=str, default='0')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda

cudnn.benchmark = True

img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

sket_root = os.path.join(opt.sketRoot, opt.cat)
model_path = os.path.join(opt.model, opt.cat)
save_root = os.path.join(opt.saveRoot, opt.cat)
if not os.path.exists(save_root):
    os.makedirs(save_root)

list_file = os.path.join(opt.sketRoot, 'split_file', opt.cat+'_' + opt.mode + '.txt')
folder_name_set = open(list_file, 'r').readlines()
folder_name_set = [f.strip() for f in folder_name_set]

network = PartGenerator()
network.cuda()
network.load_state_dict(torch.load(os.path.join(model_path, opt.modelName)))
network.eval()

vox_res = 64
pts = make_3d_grid((-0.5,)*3, (0.5,)*3, (vox_res,)*3).contiguous().view(1, -1, 3)
pts = pts.cuda()
# vox = np.zeros((vox_res, vox_res, vox_res), dtype=np.uint8)

nv = pts.shape[1]

test_num = 0
with torch.no_grad():
    for folder_name in folder_name_set:
        test_num += 1
        print(test_num, 'processing', folder_name)

        sket_path = os.path.join(sket_root, folder_name, 'sketch_part_unit')
        voxel_path = os.path.join(save_root, folder_name)
        if not os.path.exists(voxel_path):
            os.makedirs(voxel_path)

        with open(os.path.join(sket_root, folder_name, 'part_list.txt'), 'r') as fin_part:
            for part_name in fin_part.readlines():
                part_name = part_name.strip()
                for view_id in range(0, opt.nView):
                    sket = Image.open(os.path.join(sket_path, part_name + '_' + str(view_id) + '.png')).convert('RGB')
                    sket = img_transform(sket)
                    sket = sket.cuda()
                    sket = sket.contiguous().view(1, 3, 224, 224)

                    pts_occ_val = network.predict(sket, pts)
                    pts_occ_val = pts_occ_val.contiguous().view(vox_res, vox_res, vox_res).cpu().data.numpy()

                    # save the output as voxel
                    out_vox = np.array(pts_occ_val + (1. - opt.thres), dtype=np.uint8)
                    write_binvox_file(out_vox, os.path.join(voxel_path, part_name + '_' + str(view_id) + '.binvox'),
                                      voxel_size=vox_res)
        fin_part.close()

print('Done!')
