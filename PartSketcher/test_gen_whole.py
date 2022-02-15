import time
import argparse
import torch.backends.cudnn as cudnn
from dataset_gen_whole_test import *
from model import *
from common import *
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
# parser.add_argument('--dataRoot', type=str, default='/data/dudong/PartNet.v0/PartSketcher/data/', help='data root path')
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
dataset_test = PartNet(data_root=opt.dataRoot, cat=opt.cat, mode='test_example', view_pick=True)
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

pts_src = make_3d_grid((-0.5,)*3, (0.5,)*3, (vox_res,)*3).contiguous().view(1, -1, 3)
pts_src = pts_src.cuda()
# vox = np.zeros((vox_res, vox_res, vox_res), dtype=np.uint8)

part_num_per = 6
with torch.no_grad():
    for i, data in enumerate(dataloader_test, 0):
        sket, folder_name, part_name_list, view_id = data
        sket = sket.squeeze(0).cuda()
        folder_name = folder_name[0]
        view_id = view_id[0]
        print(i, 'processing', folder_name)

        out_path = os.path.join(test_path, folder_name)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        num_iter = int(np.ceil(len(part_name_list) / part_num_per))
        for k in range(0, num_iter):
            begin_id = k*part_num_per
            end_id = begin_id + part_num_per
            if end_id > len(part_name_list):
                end_id = len(part_name_list)
            id_len = end_id - begin_id

            pts = pts_src.repeat(id_len, 1, 1)
            pts_occ_val = network.predict(sket[begin_id: end_id], pts)
            pts_occ_val = pts_occ_val.contiguous().view(-1, vox_res, vox_res, vox_res).cpu().data.numpy()

            for it in range(0, id_len):
                part_name = part_name_list[it+begin_id][0]

                # save the output as mesh
                mesh = extract_mesh(pts_occ_val[it], threshold=opt.thres, n_face_simp=5000)
                mesh.export(os.path.join(out_path, part_name + '_' + view_id + '.ply'), 'ply')

                # save the output as voxel
                out_vox = np.array(pts_occ_val[it] + (1. - opt.thres), dtype=np.uint8)
                write_binvox_file(out_vox,
                                  os.path.join(out_path, part_name + '_' + view_id + '.binvox'),
                                  voxel_size=vox_res)

                # save the sketch image
                save_image(sket[it+begin_id].squeeze(0).cpu(),
                           os.path.join(out_path, part_name + '_' + view_id + '.png'))

print('Done!')
