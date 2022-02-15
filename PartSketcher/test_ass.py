import time
import argparse
import torch.backends.cudnn as cudnn
from dataset_ass import *
from model import *
from common import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type=str, default='/media/administrator/Data2/don/PartNet.v0/PartSketcher/dataset/data/', help='data root path')
parser.add_argument('--spaceSize', type=int, default=128, help='voxel space size for assembly')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--model', type=str, default='checkpoint', help='model path')
parser.add_argument('--test', type=str, default='test', help='test results path')
parser.add_argument('--cat', type=str, default='Chair')
parser.add_argument('--cuda', type=str, default='0')
opt = parser.parse_args()

# cat_set: 'Chair', ...

os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda

cudnn.benchmark = True

# Creat testing dataloader
dataset_test = PartNetAss(data_root=opt.dataRoot, cat=opt.cat, mode='test', view_pick=True)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))
len_dataset = len(dataset_test)
print('test set num', len_dataset)

model_path = os.path.join(opt.model, opt.cat)
test_path = os.path.join(opt.test, opt.cat)
if not os.path.exists(test_path):
    os.makedirs(test_path)


# Create network
network = PartAssembler()
network.cuda()
network.load_state_dict(torch.load(os.path.join(model_path, 'assembler.pt')))
network.eval()

# Create Loss Module
criterion_l1 = nn.L1Loss()

total_time = 0
test_num = 0
with torch.no_grad():
    for i, data in enumerate(dataloader_test, 0):
        imgs, vox, pos_gt, folder_name, part_name, view_id = data
        imgs = imgs.cuda()
        vox = vox.cuda()
        pos_gt = pos_gt.cuda()

        start_time = time.time()
        pos_pre = network(imgs, vox)
        cost_time = time.time() - start_time
        if i > 0:
            total_time += cost_time
            test_num += 1

        folder_name = folder_name[0]
        part_name = part_name[0]
        view_id = view_id[0]

        loss = criterion_l1(pos_pre, pos_gt)
        print('test', folder_name + '_' + part_name, 'loss:', loss.item(), 'cost time:', cost_time)

        pos_pre_np = pos_pre.contiguous().view(-1).cpu().data.numpy() * opt.spaceSize
        pos_gt_np = pos_gt.contiguous().view(-1).cpu().data.numpy() * opt.spaceSize
        np.savetxt(os.path.join(test_path, folder_name + '_' + part_name + '_' + view_id + '_pos_pre.txt'), pos_pre_np,
                   fmt='%1.3f')
        np.savetxt(os.path.join(test_path, folder_name + '_' + part_name + '_' + view_id + '_pos_gt.txt'), pos_gt_np,
                   fmt='%1.3f')

        if i > 19:
            break
print('average time cost:', total_time / test_num)
print('Done!')