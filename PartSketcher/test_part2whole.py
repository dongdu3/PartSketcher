import argparse
import os
import numpy as np
from utils import binvox_rw
from common import *

parser = argparse.ArgumentParser()
# parser.add_argument('--dataRoot', type=str, default='/media/administrator/Data2/don/PartNet.v0/PartSketcher/dataset/data/', help='data root path')
parser.add_argument('--dataRoot', type=str, default='/mnt/groupprofxghan/dudong/data/PartNet.v0/PartSketcher/data/', help='data root path')
parser.add_argument('--thres', type=float, default=0.2, help='threshold for occupancy estimation and mesh extraction')
parser.add_argument('--spaceSize', type=int, default=128, help='voxel space size for assembly')
parser.add_argument('--viewNum', type=int, default=9, help='sketch view number of per model')
parser.add_argument('--viewPick', type=bool, default=True, help='is pick view or not')
# parser.add_argument('--test', type=str, default='/media/administrator/Code/don/SketchModeling/PartSketcher_v1_results/test', help='test results path')
parser.add_argument('--test', type=str, default='test', help='test results path')
parser.add_argument('--postfix', type=str, default='pre', help='the postfix of position prediction file')
parser.add_argument('--mode', type=str, default='test_example', help='the dataset list type: train/val/test')
parser.add_argument('--cat', type=str, default='Chair')
opt = parser.parse_args()

test_path = os.path.join(opt.test, opt.cat)

list_file = os.path.join(opt.dataRoot, 'split_file', opt.cat+'_'+opt.mode+'.txt')
fnames = open(list_file, 'r').readlines()
fnames = [f.strip() for f in fnames]

view_id_set = []
if not opt.viewPick:
    for i in range(0, 9):
        view_id_set.append(str(i))
else:
    view_id_set.append('5')

def write_binvox_file(data, filename, voxel_size=64, axis_order='xyz'):     # xyz or xzy
    with open(filename, 'wb') as f:
        voxel = binvox_rw.Voxels(data, [voxel_size, voxel_size, voxel_size], [0, 0, 0], 1, axis_order)
        binvox_rw.write(voxel, f)
    f.close()

# def write_ply(fname, v):
#     with open(fname, 'w') as fout:
#         fout.write('ply\n')
#         fout.write('format ascii 1.0\n')
#         fout.write('element vertex '+str(v.shape[0])+'\n');
#         fout.write('property float x\n')
#         fout.write('property float y\n')
#         fout.write('property float z\n')
#         fout.write('end_header\n')
#
#         for i in range(v.shape[0]):
#             fout.write('%f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))
#
# def write_ply_with_color(fname, v, c):
#     with open(fname, 'w') as fout:
#         fout.write('ply\n')
#         fout.write('format ascii 1.0\n')
#         fout.write('element vertex '+str(v.shape[0])+'\n')
#         fout.write('property float x\n')
#         fout.write('property float y\n')
#         fout.write('property float z\n')
#         fout.write('property uchar red\n')
#         fout.write('property uchar green\n')
#         fout.write('property uchar blue\n')
#         fout.write('end_header\n')
#
#         for i in range(v.shape[0]):
#             fout.write('%f %f %f %d %d %d\n' % (v[i, 0], v[i, 1], v[i, 2], c[i][0], c[i][1], c[i][2]))

# def write_part_infor(fname, center, length, scale):
#     with open(fname, 'w') as fout:
#         fout.write('bbox_center ' + str(center[0]) + ' ' + str(center[1]) + ' '  + str(center[2]) + '\n')
#         fout.write('bbox_length ' + str(length[0]) + ' ' + str(length[1]) + ' '  + str(length[2]) + '\n')
#         fout.write('bbox_scale ' + str(scale))
#     fout.close()
#
# def trans_part_vox(part_vox, center, length, bbox_min, new_vox_size=64):
#     new_center = np.array((new_vox_size//2-1, new_vox_size//2-1, new_vox_size//2-1), dtype=np.int)
#     new_max_len = new_vox_size * 0.9
#     scale = new_max_len/np.max(length)
#     new_bbox_min = np.array(bbox_min, dtype=np.float)
#     new_bbox_min -= center
#     new_bbox_min *= scale
#     new_bbox_min += new_center
#
#     new_length = np.array(length*scale, dtype=int) + 4
#     new_bbox_min = np.array(new_bbox_min, dtype=int) - 2
#
#     # calculate the transformed occupancies (avoid for-loop)
#     tmp_vox = np.zeros((new_vox_size, new_vox_size, new_vox_size), dtype=np.uint8)
#     tmp_vox[new_bbox_min[0]: new_bbox_min[0] + new_length[0], new_bbox_min[1]: new_bbox_min[1] + new_length[1],
#         new_bbox_min[2]: new_bbox_min[2] + new_length[2]] = 1
#     tmp_pos = np.where(tmp_vox > 0.1)
#     tmp_pos = np.array(tmp_pos, dtype=np.float).transpose()
#     tmp_pos_int = np.array(tmp_pos, dtype=np.int)
#     tmp_pos -= new_center
#     tmp_pos /= scale
#     tmp_pos += center
#     tmp_pos_int_ori = np.array(tmp_pos, dtype=np.int)
#
#     part_vox_trans = np.zeros((new_vox_size, new_vox_size, new_vox_size), dtype=np.uint8)
#     part_vox_trans[tmp_pos_int[:, 0], tmp_pos_int[:, 1], tmp_pos_int[:, 2]] = part_vox[
#         tmp_pos_int_ori[:, 0], tmp_pos_int_ori[:, 1], tmp_pos_int_ori[:, 2]]
#
#     return part_vox_trans
#####################################################################################################################

# load part voxel and do assembly
n_process = 0
for i in range(0, len(fnames)):
    folder_name = fnames[i]
    out_path = os.path.join(test_path, folder_name)
    for view_id in view_id_set:
        print(i, 'processing', folder_name, view_id, '...')
        whole_vox = np.zeros((opt.spaceSize, opt.spaceSize, opt.spaceSize), dtype=np.uint8)
        with open(os.path.join(opt.dataRoot, opt.cat, folder_name, 'part_list.txt'), 'r') as fin:
            for part_name in fin.readlines():
                part_name = part_name.strip()
                part_vox_path = os.path.join(out_path, part_name + '_' + view_id + '.binvox')

                # load voxel data
                fp = open(part_vox_path, 'rb')
                part_vox = binvox_rw.read_as_3d_array(fp, fix_coords=True).data
                part_vox = np.array(part_vox, dtype='uint8')
                fp.close()

                # calculate part voxel information
                part_size = part_vox.shape[0]
                part_pos = np.where(part_vox > 0.1)
                part_pos = np.array(part_pos).transpose()
                part_bbox_min = np.min(part_pos, axis=0)
                part_bbox_max = np.max(part_pos, axis=0)
                part_center = (part_bbox_min + part_bbox_max) / 2.
                part_scale = np.linalg.norm(part_bbox_max - part_bbox_min) / 2.

                # load position information for the target space
                pos_pre = np.loadtxt(os.path.join(out_path, part_name + '_' + view_id + '_pos_' + opt.postfix + '.txt'),
                                     dtype=np.float)
                center = np.array((pos_pre[0], pos_pre[1], pos_pre[2]), dtype=np.float)
                scale = np.float(pos_pre[3])
                scale_ratio = scale/part_scale
                length = (part_bbox_max - part_bbox_min) * scale_ratio
                bbox_min = np.array(np.clip(center - length / 2., a_min=0, a_max=opt.spaceSize-1), dtype=np.int)
                length = np.ceil(length).astype(np.int)

                # calculate the transformed occupancies (avoid for-loop)
                tmp_vox = np.zeros((opt.spaceSize, opt.spaceSize, opt.spaceSize), dtype=np.uint8)
                tmp_vox[bbox_min[0]: bbox_min[0] + length[0], bbox_min[1]: bbox_min[1] + length[1],
                    bbox_min[2]: bbox_min[2] + length[2]] = 1
                tmp_pos = np.where(tmp_vox > 0.1)
                tmp_pos = np.array(tmp_pos, dtype=np.float).transpose()
                tmp_pos_int = np.array(tmp_pos, dtype=np.int)
                tmp_pos -= center
                tmp_pos = tmp_pos/scale_ratio
                tmp_pos += part_center
                tmp_pos_part_int = np.array(tmp_pos, dtype=np.int)
                tmp_pos_part_int = np.clip(tmp_pos_part_int, a_min=0, a_max=part_size-1)

                whole_vox[tmp_pos_int[:, 0], tmp_pos_int[:, 1], tmp_pos_int[:, 2]] += part_vox[
                    tmp_pos_part_int[:, 0], tmp_pos_part_int[:, 1], tmp_pos_part_int[:, 2]]

            fin.close()

        # write the part assembly results
        write_binvox_file(whole_vox, os.path.join(out_path, 'assembly_' + view_id + '.binvox'), voxel_size=128)

        mesh = extract_mesh(whole_vox.astype(np.float), threshold=opt.thres, n_face_simp=6000)
        mesh.export(os.path.join(out_path, 'assembly_' + view_id + '.ply'), 'ply')

print('Processed done!')
