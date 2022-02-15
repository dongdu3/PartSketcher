import torch.nn as nn
from torch import distributions as dist
from layers import *
import resnet


class DecoderCBatchNorm(nn.Module):
    ''' Decoder with conditional batch normalization (CBN) class.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, leaky=False, legacy=False):
        super(DecoderCBatchNorm, self).__init__()

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)

        if not legacy:
            self.bn = CBatchNorm1d(c_dim, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, c, **kwargs):
        p = p.transpose(1, 2)
        # batch_size, D, T = p.size()
        net = self.fc_p(p)

        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        return out


class VoxelEncoder64(nn.Module):
    ''' Voxel64 Decoder with batch normalization (BN) class.
        Args:
            z_dim (int): input feature z dimension
            gf_dim (int): dimension of feature channel
    '''
    def __init__(self, f_dim=256, leak_val=0.2):
        super(VoxelEncoder64, self).__init__()

        self.f_dim = f_dim
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, self.f_dim // 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.f_dim // 16),
            nn.LeakyReLU(leak_val)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(self.f_dim // 16, self.f_dim // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.f_dim // 8),
            nn.LeakyReLU(leak_val)
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(self.f_dim // 8, self.f_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.f_dim // 4),
            nn.LeakyReLU(leak_val)
        )
        self.layer4 = nn.Sequential(
            nn.Conv3d(self.f_dim // 4, self.f_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.f_dim // 2),
            nn.LeakyReLU(leak_val)
        )
        self.layer5 = nn.Sequential(
            nn.Conv3d(self.f_dim // 2, self.f_dim, kernel_size=4, stride=2, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        # print(x.size()) # torch.Size([bs, 1, 64, 64, 64])
        out = self.layer1(x)
        # print(out.size())  # torch.Size([bs, 16, 32, 32, 32])
        out = self.layer2(out)
        # print(out.size())  # torch.Size([bs, 32, 16, 16, 16])
        out = self.layer3(out)
        # print(out.size())  # torch.Size([bs, 64, 8, 8, 8])
        out = self.layer4(out)
        # print(out.size())  # torch.Size([bs, 128, 4, 4, 4])
        out = self.layer5(out)
        # print(out.size())  # torch.Size([bs, 256, 1, 1, 1])
        out = out.view(out.size(0), -1)

        return out


class PositionParser(nn.Module):    # output 4dof information: center + length
    def __init__(self, bottleneck_size=512, pos_dim=4):
        super(PositionParser, self).__init__()

        self.bottleneck_size = bottleneck_size
        self.pos_dim = pos_dim
        self.layer1 = nn.Sequential(
            nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1),
            nn.BatchNorm1d(self.bottleneck_size//2),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1),
            nn.BatchNorm1d(self.bottleneck_size//4),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(self.bottleneck_size//4, self.pos_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.contiguous().view(-1, self.bottleneck_size, 1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.contiguous().view(-1, self.pos_dim)

        return out

"""
##########################################define network##########################################
"""

class PartGenerator(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
    '''

    def __init__(self, z_dim=256, img_dim=3):
        super(PartGenerator, self).__init__()
        self.encoder = resnet.Resnet18(z_dim)
        self.decoder = DecoderCBatchNorm(dim=img_dim, c_dim=256, hidden_size=256)

        self.sigmoid = nn.Sigmoid()

    def forward(self, img, p):
        ''' Performs a forward pass through the network.

        Args:
            img (tensor): input image
            p (tensor): sampled points
        '''
        img = img[:, :3, :, :].contiguous()

        c = self.encoder(img)

        logits = self.decoder(p, c)
        p_occ = dist.Bernoulli(logits=logits).logits
        p_occ_sigmoid = self.sigmoid(p_occ)

        return p_occ, p_occ_sigmoid

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def decode(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def predict(self, img, pts):
        c = self.encoder(img)

        # print('p_shape', p.size())
        pts_occ = self.decode(pts, c).logits
        pts_occ_sigmoid = self.sigmoid(pts_occ)

        return pts_occ_sigmoid


class PartAssembler(nn.Module):
    ''' Part Assembly Network class.

    Args:
        image encoder (nn.Module): encoder network for 2D sketch images
        voxel encoder (nn.Module): encoder network for 3D voxel
    '''

    def __init__(self, f_dim=256, img_dim=6, pos_dim=4, feature_transform=False):
        super(PartAssembler, self).__init__()
        self.encoder2d = resnet.ModifiedResnet18(img_dim, f_dim)
        self.encoder3d = VoxelEncoder64(f_dim=f_dim)
        self.decoder = PositionParser(bottleneck_size=f_dim*2, pos_dim=pos_dim)

    def forward(self, imgs, vox):
        feat_img = self.encoder2d(imgs)
        feat_vox = self.encoder3d(vox)
        feat = torch.cat((feat_img, feat_vox), 1)
        out = self.decoder(feat)

        return out