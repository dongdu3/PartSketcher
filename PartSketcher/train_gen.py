from __future__ import print_function
import argparse
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
# import time,datetime
from tensorboardX import SummaryWriter
from dataset_gen import *
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type=str, default='/media/administrator/Data2/don/PartNet.v0/PartSketcher/dataset/data/', help='data root path')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--model', type=str, default='checkpoint', help='model path')
parser.add_argument('--log', type=str, default='log', help='log path')
parser.add_argument('--nepoch', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=3e-5, help='Initial learning rate.')
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
batch_size_val = 8
dataset = PartNet(data_root=opt.dataRoot, cat=opt.cat, mode='train')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
dataset_val = PartNet(data_root=opt.dataRoot, cat=opt.cat, mode='val')
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size_val,
                                             shuffle=False, num_workers=int(opt.workers))

len_dataset = len(dataset)
len_dataset_val = len(dataset_val)
print('training set num', len_dataset)
print('validation set num', len_dataset_val)

cudnn.benchmark = True

# create path
model_path = os.path.join(opt.model, opt.cat)
log_path = os.path.join(opt.log, opt.cat)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)
logger = SummaryWriter(log_path)

# Create network
network = PartGenerator()
network.cuda()

# Create Loss Module
criterion_bce = nn.BCEWithLogitsLoss()

# Create optimizer
optimizer = optim.Adam(network.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

it_step = 0
min_loss = 1e8
best_epoch = 0
for epoch in range(1, opt.nepoch+1):
    # TRAIN MODE
    network.train()
    for it, data in enumerate(dataloader, 0):
        it_step += 1

        optimizer.zero_grad()
        sket, occ_data, _, _, _ = data
        sket = sket.cuda()
        occ_p = occ_data[None].cuda()
        occ_val = occ_data['occ'].cuda()

        occ_pre, _ = network(sket, occ_p)
        loss = criterion_bce(occ_pre, occ_val)

        loss.backward()
        optimizer.step()

        if it_step % 10 == 0:
            logger.add_scalar('train/loss_gen', loss, it_step)
            print('[%d: %d/%d] train loss: %f' % (epoch, it, len_dataset / opt.batchSize, loss.item()))

    # # VALIDATION
    # loss_val = 0
    # loss_n = 0
    # network.eval()
    # with torch.no_grad():
    #     for it, data in enumerate(dataloader_val, 0):
    #         sket, occ_data, _, _ = data
    #         sket = sket.cuda()
    #         occ_p = occ_data[None].cuda()
    #         occ_val = occ_data['occ'].cuda()
    #
    #         occ_pre, _ = network(sket, occ_p)
    #         loss = criterion_bce(occ_pre, occ_val)
    #
    #         loss_val += loss.item()
    #         loss_n += 1
    #
    #         if it % 10 == 0:
    #             print('[%d: %d/%d] val loss: %f' % (epoch, it, len_dataset_val / batch_size_val, loss.item()))
    #
    # loss_val /= loss_n
    # logger.add_scalar('val/loss_gen', loss_val, it_step)
    #
    # if loss_val < min_loss:
    #     min_loss = loss_val
    #     best_epoch = epoch
    #     print('Best epoch is:', best_epoch)
    # elif loss_val-min_loss > 5e-3:
    #     break

    # SAVE MODEL
    torch.save(network.state_dict(), os.path.join(model_path, 'generator.pt'))
    print('save model succeeded!')

    # save the middle checkpoint every 100 epochs
    if epoch > 500 and epoch % 100 == 0:
        torch.save(network.state_dict(), os.path.join(model_path, 'generator_%s.pt' % (str(epoch))))
        print('save the %s epoch model succeeded!' % (str(epoch)))

print('Training done!')
print('Best epoch is', best_epoch)
