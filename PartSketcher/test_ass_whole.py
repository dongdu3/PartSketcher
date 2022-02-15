import time
import argparse
import torch.backends.cudnn as cudnn
from dataset_ass_whole import *
from model import *
from common import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type=str, default='/data/dudong/PartNet.v0/PartSketcher/data/', help='data root path')
parser.add_argument('--spaceSize', type=int, default=128, help='voxel space size for assembly')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--model', type=str, default='checkpoint', help='model path')
parser.add_argument('--modelName', type=str, default='assembler.pt', help='model name')
parser.add_argument('--test', type=str, default='test_whole', help='test results path')
parser.add_argument('--cat', type=str, default='Chair')
parser.add_argument('--cuda', type=str, default='0')
opt = parser.parse_args()

# cat_set: 'Chair', ...

os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda

cudnn.benchmark = True

# Creat testing dataloader
dataset_test = PartNetAss(data_root=opt.dataRoot, cat=opt.cat, mode='test', view_pick=True, test_path=opt.test)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))
len_dataset = len(dataset_test)
print('test set num', len_dataset)

model_path = os.path.join(opt.model, opt.cat, opt.modelName)
test_path = os.path.join(opt.test, opt.cat)
if not os.path.exists(test_path):
    os.makedirs(test_path)

# Create network
network = PartAssembler()
network.cuda()
network.load_state_dict(torch.load(model_path))
network.eval()

# Create Loss Module
criterion_l1 = nn.L1Loss()

total_time = 0
test_num = 0
with torch.no_grad():
    for i, data in enumerate(dataloader_test, 0):
        imgs, vox, pos_gt, folder_name, part_name_list, view_id = data
        imgs = imgs.squeeze(0).cuda()
        vox = vox.squeeze(0).cuda()
        pos_gt = pos_gt.squeeze(0).cuda()
        folder_name = folder_name[0]
        view_id = view_id[0]

        out_path = os.path.join(test_path, folder_name)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        start_time = time.time()
        pos_pre = network(imgs, vox)
        cost_time = time.time() - start_time
        if i > 0:
            total_time += cost_time
            test_num += 1

        loss = criterion_l1(pos_pre, pos_gt)
        print('test', folder_name, 'loss:', loss.item(), 'cost time:', cost_time)

        pos_pre_np = pos_pre.contiguous().view(len(part_name_list), -1).cpu().data.numpy() * opt.spaceSize
        pos_gt_np = pos_gt.contiguous().view(len(part_name_list), -1).cpu().data.numpy() * opt.spaceSize
        for it in range(0, len(part_name_list)):
            np.savetxt(os.path.join(out_path, part_name_list[it][0] + '_' + view_id + '_pos_pre.txt'),
                       pos_pre_np[it],
                       fmt='%1.3f')
            np.savetxt(os.path.join(out_path, part_name_list[it][0] + '_' + view_id + '_pos_gt.txt'),
                       pos_gt_np[it],
                       fmt='%1.3f')

print('average time cost:', total_time / test_num)
print('Done!')
