import time
import argparse
import torch.backends.cudnn as cudnn
from dataset_gen_test import *
from model import *
from common import *
from torchvision.utils import save_image
# import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type=str, default='/media/administrator/Data2/don/PartNet.v0/PartSketcher/dataset/data/', help='data root path')
parser.add_argument('--thres', type=float, default=0.2, help='threshold for occupancy estimation and mesh extraction')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--model', type=str, default='checkpoint', help='model path')
parser.add_argument('--modelName', type=str, default='generator.pt', help='model name')
parser.add_argument('--test', type=str, default='test', help='test results path')
parser.add_argument('--cat', type=str, default='Chair')
parser.add_argument('--cuda', type=str, default='0')
opt = parser.parse_args()

# cat_set: 'Chair', ...

os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda

cudnn.benchmark = True

# Creat testing dataloader
dataset_test = PartNet(data_root=opt.dataRoot, cat=opt.cat, mode='test', view_pick=True)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))
len_dataset = len(dataset_test)
print('test set num', len_dataset)

model_path = os.path.join(opt.model, opt.cat)
test_path = os.path.join(opt.test, opt.cat)
if not os.path.exists(test_path):
    os.makedirs(test_path)

network = PartGenerator()
network.cuda()
network.load_state_dict(torch.load(os.path.join(model_path, opt.modelName)))
network.eval()

vox_res = 64
# # without refining results
# generator = Generator3D(network, threshold=0.2, refinement_step=0, resolution0=vox_res, upsampling_steps=0)

# # with refining results
# generator = Generator3D(network, threshold=0.2, refinement_step=30, resolution0=32, upsampling_steps=2, simplify_nfaces=5000)

pts = make_3d_grid((-0.5,)*3, (0.5,)*3, (vox_res,)*3).contiguous().view(1, -1, 3)
pts = pts.cuda()
# vox = np.zeros((vox_res, vox_res, vox_res), dtype=np.uint8)

total_time = 0
test_num = 0
with torch.no_grad():
    for i, data in enumerate(dataloader_test, 0):
        sket, _, name, view_id = data
        sket = sket.cuda()

        name = name[0]
        view_id = view_id[0]
        folder_name, part_name = name.split('/')
        print(i, 'processing', name)

        start_time = time.time()

        pts_occ_val = network.predict(sket, pts)

        pts_occ_val = pts_occ_val.contiguous().view(vox_res, vox_res, vox_res).cpu().data.numpy()

        cost_time = time.time() - start_time
        print('time cost:', cost_time)
        if i > 0:
            total_time += cost_time
            test_num += 1

        # save the output as mesh
        mesh = extract_mesh(pts_occ_val, threshold=opt.thres, n_face_simp=10000)
        mesh.export(os.path.join(test_path, folder_name + '_' + part_name + '_' + view_id + '.ply'), 'ply')

        # save the output as voxel
        out_vox = np.array(pts_occ_val + (1. - opt.thres), dtype=np.uint8)
        write_binvox_file(out_vox, os.path.join(test_path, folder_name + '_' + part_name + '_' + view_id + '.binvox'),
                          voxel_size=vox_res)

        # save the sketch image
        save_image(sket.squeeze(0).cpu(),
                   os.path.join(test_path, folder_name + '_' + part_name + '_' + view_id + '.png'))

        if i > 99:
            break
print('average time cost:', total_time / test_num)
print('Done!')
