from __future__ import print_function
import argparse
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
# import time,datetime
from tensorboardX import SummaryWriter
from dataset_ass import *
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type=str, default='/data/dudong/PartSketcher/', help='data root path')
parser.add_argument('--dataGenRoot', type=str, default='/data/dudong/PartSketcher/data_gen/', help='data generation root path')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--model', type=str, default='checkpoint', help='model path')
parser.add_argument('--modelName', type=str, default='assembler_best.pt', help='model name')
parser.add_argument('--log', type=str, default='log', help='log path')
parser.add_argument('--nepoch', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--cat', type=str, default='Chair')
parser.add_argument('--cuda', type=str, default='0')
opt = parser.parse_args()

# cat_set: 'Chair', ...

os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda

opt.manualSeed = random.randint(1, 10000)  # fix seed
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# Creat train dataloader
dataset = PartNetAss(data_root=opt.dataRoot, cat=opt.cat, mode='train', data_gen_root=opt.dataGenRoot)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
dataset_val = PartNetAss(data_root=opt.dataRoot, cat=opt.cat, mode='val')
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batchSize,
                                             shuffle=False, num_workers=int(opt.workers))

len_dataset = len(dataset)
len_dataset_val = len(dataset_val)
print('training set num', len_dataset)
print('validation set num', len_dataset_val)

cudnn.benchmark = True

# create path
model_path = os.path.join(opt.model, opt.cat)
model_name = opt.modelName
if not os.path.exists(os.path.join(model_path, model_name)):
    model_name = 'assembler.pt'
print('pre-trained model name:', model_name)

log_path = os.path.join(opt.log, opt.cat, 'ass_ft')
if not os.path.exists(log_path):
    os.makedirs(log_path)
logger = SummaryWriter(log_path)

# Create network
network = PartAssembler()
network.cuda()
network.load_state_dict(torch.load(os.path.join(model_path, model_name)))
print('loaded pre-trained weights.')

# Create Loss Module
criterion_l1 = nn.L1Loss()

# Create optimizer
optimizer = optim.Adam(network.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

it_step = 0
min_loss = 1e8
best_epoch = 0

print('Pre-testing the network.')
network.eval()
loss_val = 0
loss_n = 0
fw = open(os.path.join(log_path, 'val_log.txt'), 'w')
with torch.no_grad():
    for it, data in enumerate(dataloader_val, 0):
        imgs, vox, pos_gt, _, _, _ = data
        imgs = imgs.cuda()
        vox = vox.cuda()
        pos_gt = pos_gt.cuda()

        pos_pre = network(imgs, vox)
        loss = criterion_l1(pos_pre, pos_gt)

        loss_val += loss.item()
        loss_n += 1

        if it % 10 == 0:
            print('[%d: %d/%d] val loss: %f' % (0, it, len_dataset_val / opt.batchSize, loss.item()))

    loss_val /= loss_n
    logger.add_scalar('val/loss_ass', loss_val, it_step)

    if loss_val < min_loss:
        min_loss = loss_val
        best_epoch = 0

    fw.write('epoch 0: val_loss %s, best epoch %s. \n' % (str(loss_val), str(best_epoch)))

for epoch in range(1, opt.nepoch+1):
    # TRAIN MODE
    network.train()
    for it, data in enumerate(dataloader, 0):
        it_step += 1

        optimizer.zero_grad()
        imgs, vox, pos_gt, _, _, _ = data
        imgs = imgs.cuda()
        vox = vox.cuda()
        pos_gt = pos_gt.cuda()

        pos_pre = network(imgs, vox)

        loss = criterion_l1(pos_pre, pos_gt)

        loss.backward()
        optimizer.step()

        if it_step % 10 == 0:
            logger.add_scalar('train/loss_ass', loss, it_step)
            print('[%d: %d/%d] train loss: %f' % (epoch, it, len_dataset / opt.batchSize, loss.item()))

    # VALIDATION MODE
    network.eval()

    loss_val = 0
    loss_n = 0
    with torch.no_grad():
        for it, data in enumerate(dataloader_val, 0):
            imgs, vox, pos_gt, _, _, _ = data
            imgs = imgs.cuda()
            vox = vox.cuda()
            pos_gt = pos_gt.cuda()

            pos_pre = network(imgs, vox)
            loss = criterion_l1(pos_pre, pos_gt)

            loss_val += loss.item()
            loss_n += 1

            if it % 10 == 0:
                print('[%d: %d/%d] val loss: %f' % (epoch, it, len_dataset_val / opt.batchSize, loss.item()))

    loss_val /= loss_n
    logger.add_scalar('val/loss_ass', loss_val, it_step)

    if loss_val < min_loss:
        min_loss = loss_val
        best_epoch = epoch

        torch.save(network.state_dict(), os.path.join(model_path, 'assembler_ft.pt'))
        print('save model succeeded!')
        print('Best epoch is:', best_epoch)

    fw.write('epoch %s: val_loss %s, best epoch %s. \n' % (str(epoch), str(loss_val), str(best_epoch)))

fw.close()
print('Training done!')
print('Best epoch is:', best_epoch)
