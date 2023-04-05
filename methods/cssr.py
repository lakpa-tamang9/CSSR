import methods.wideresnet as wideresnet
from methods.augtools import HighlyCustomizableAugment, RandAugmentMC
import methods.util as util
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import random
from methods.util import AverageMeter
import time
from torchvision.transforms import transforms
from methods.resnet import ResNet
from torchvision import models as torchvision_models
from methods.cac_utils import *
from methods.utils import *
from tensorboardX import SummaryWriter


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class GramRecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.gram_feats = []
        self.collecting = False

    def begin_collect(self,):
        self.gram_feats.clear()
        self.collecting = True
        # print("begin collect")

    def record(self, ft):
        if self.collecting:
            self.gram_feats.append(ft)
            # print("record")

    def obtain_gram_feats(self,):
        tmp = self.gram_feats
        self.collecting = False
        self.gram_feats = []
        # print("record")
        return tmp


class PretrainedResNet(nn.Module):

    def __init__(self, rawname, pretrain_path=None) -> None:
        super().__init__()
        if pretrain_path == 'default':
            self.model = torchvision_models.__dict__[rawname](pretrained=True)
            self.output_dim = self.model.fc.weight.shape[1]
            self.model.fc = nn.Identity()
        else:
            self.model = torchvision_models.__dict__[rawname]()
            self.output_dim = self.model.fc.weight.shape[1]
            self.model.fc = nn.Identity()
            if pretrain_path is not None:
                sd = torch.load(pretrain_path)
                self.model.load_state_dict(sd, strict=True)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x


class Backbone(nn.Module):

    def __init__(self, config, inchan):
        super().__init__()

        if config['backbone'] == 'wideresnet28-2':
            self.backbone = wideresnet.WideResNetBackbone(
                None, 28, 2, 0, config['category_model']['projection_dim'])
        elif config['backbone'] == 'wideresnet40-4':
            self.backbone = wideresnet.WideResNetBackbone(
                None, 40, 4, 0, config['category_model']['projection_dim'])
        elif config['backbone'] == 'wideresnet16-8':
            self.backbone = wideresnet.WideResNetBackbone(
                None, 16, 8, 0.4, config['category_model']['projection_dim'])
        elif config['backbone'] == 'wideresnet28-10':
            self.backbone = wideresnet.WideResNetBackbone(
                None, 28, 10, 0.3, config['category_model']['projection_dim'])
        elif config['backbone'] == 'resnet18':
            self.backbone = ResNet(
                output_dim=config['category_model']['projection_dim'], inchan=inchan)
        elif config['backbone'] == 'resnet18a':
            self.backbone = ResNet(
                output_dim=config['category_model']['projection_dim'], resfirststride=2, inchan=inchan)
        elif config['backbone'] == 'resnet18b':
            self.backbone = ResNet(
                output_dim=config['category_model']['projection_dim'], resfirststride=2, inchan=inchan)
        elif config['backbone'] == 'resnet34':
            self.backbone = ResNet(output_dim=config['category_model']['projection_dim'], num_block=[
                                   3, 4, 6, 3], inchan=inchan)
        elif config['backbone'] in ['prt_r18', 'prt_r34', 'prt_r50']:
            self.backbone = PretrainedResNet(
                {'prt_r18': 'resnet18', 'prt_r34': 'resnet34', 'prt_r50': 'resnet50'}[config['backbone']])
        elif config['backbone'] in ['prt_pytorchr18', 'prt_pytorchr34', 'prt_pytorchr50']:
            name, path = {
                'prt_pytorchr18': ('resnet18', 'default'),
                'prt_pytorchr34': ('resnet34', 'default'),
                'prt_pytorchr50': ('resnet50', 'default')
            }[config['backbone']]
            self.backbone = PretrainedResNet(name, path)
        elif config['backbone'] in ['prt_dinor18', 'prt_dinor34', 'prt_dinor50']:
            name, path = {
                'prt_dinor50': ('resnet50', './model_weights/dino_resnet50_pretrain.pth')
            }[config['backbone']]
            self.backbone = PretrainedResNet(name, path)
        else:
            bkb = config['backbone']
            raise Exception(f'Backbone \"{bkb}\" is not defined.')

        # types : ae_softmax_avg , ae_avg_softmax , avg_ae_softmax
        self.output_dim = self.backbone.output_dim
        # self.classifier = CRFClassifier(self.backbone.output_dim,numclss,config)

    def forward(self, x):
        x = self.backbone(x)
        # latent , global prob , logits
        return x


class LinearClassifier(nn.Module):

    def __init__(self, inchannels, num_class, config):
        super().__init__()
        self.gamma = config['gamma']
        self.cls = nn.Conv2d(inchannels, num_class, 1, padding=0, bias=False)

    def forward(self, x):
        x = self.cls(x)
        return x * self.gamma


def sim_conv_layer(input_channel, output_channel, kernel_size=1, padding=0, use_activation=True):
    if use_activation:
        res = nn.Sequential(
            nn.Conv2d(input_channel, output_channel,
                      kernel_size, padding=padding, bias=False),
            nn.Tanh())
    else:
        res = nn.Conv2d(input_channel, output_channel,
                        kernel_size, padding=padding, bias=False)
    return res


class AutoEncoder(nn.Module):

    def __init__(self, inchannel, hidden_layers, latent_chan):
        super().__init__()
        layer_block = sim_conv_layer
        self.latent_size = latent_chan
        if latent_chan > 0:
            self.encode_convs = []
            self.decode_convs = []
            for i in range(len(hidden_layers)):
                h = hidden_layers[i]
                ecv = layer_block(inchannel, h,)
                dcv = layer_block(h, inchannel, use_activation=i != 0)
                inchannel = h
                self.encode_convs.append(ecv)
                self.decode_convs.append(dcv)
            self.encode_convs = nn.ModuleList(self.encode_convs)
            self.decode_convs.reverse()
            self.decode_convs = nn.ModuleList(self.decode_convs)
            self.latent_conv = layer_block(inchannel, latent_chan)
            self.latent_deconv = layer_block(
                latent_chan, inchannel, use_activation=(len(hidden_layers) > 0))
        else:
            self.center = nn.Parameter(torch.rand([inchannel, 1, 1]), True)

    # TODO: Create anchors and use cac based method here
    def forward(self, x):
        if self.latent_size > 0:
            output = x
            for cv in self.encode_convs:
                output = cv(output)
            latent = self.latent_conv(output)
            output = self.latent_deconv(latent)
            for cv in self.decode_convs:
                output = cv(output)
            return output, latent
        else:
            return self.center, self.center


class CSSRClassifier(nn.Module):
    def __init__(self, inchannels, num_class, config):
        super().__init__()
        ae_hidden = config['ae_hidden']
        ae_latent = config['ae_latent']
        self.class_aes = []
        for i in range(num_class):
            ae = AutoEncoder(inchannels, ae_hidden, ae_latent)
            self.class_aes.append(ae)
        # nn.Modulelist holds submodules in  list
        self.class_aes = nn.ModuleList(self.class_aes)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.useL1 = config['error_measure'] == 'L1'

        self.reduction = -1 if config['model'] == 'pcssr' else 1
        self.reduction *= config['gamma']

    def ae_error(self, rc, x):
        """Calculating the autoencoder reconstruction error.

        Args:
            rc (reconstructed): _description_
            x (input): _description_

        Returns:
            _type_: _description_
        """
        # if self.useL1:
        #     # torch.norm calculates the L1 norm (also known as Manhattan distance or taxicab distance)
        #     # between two tensors rc and x along second dimension
        #     # return torch.sum(torch.abs(rc-x) * self.reduction,dim=1,keepdim=True)
        #     return torch.norm(rc - x, p=1, dim=1, keepdim=True) * self.reduction
        # # keepdim = True argument ensures that the result has the same number of dimensions as the input tensor
        # else:
        # Euclidean dist norm
        return torch.norm(rc - x, p=2, dim=1, keepdim=True) ** 2 * self.reduction

    clip_len = 100

    def forward(self, x):
        cls_ers = []
        for i in range(len(self.class_aes)):
            rc, lt = self.class_aes[i](x)  # Pass the input to autoencoder i.
            z_mean = torch.mean(lt, dim=0, keepdim=True)
            # splitting the tensor into chunks of batch_size shape
            lt_chunks = torch.chunk(lt, chunks=lt.shape[0], dim=0)
            errs = []
            for lt_chunk in lt_chunks:
                err = self.ae_error(z_mean, lt_chunk)
                errs.append(err)
            # Concatenate all chunks to create size of batch_size.
            cat_errs = torch.cat(errs, dim=0)

            if CSSRClassifier.clip_len > 0:
                cat_errs = torch.clamp(
                    cat_errs, -CSSRClassifier.clip_len, CSSRClassifier.clip_len)

            cls_ers.append(cat_errs)
        logits = torch.cat(cls_ers, dim=1)

        return logits


def G_p(ob, p):
    temp = ob.detach()
    temp = temp**p
    temp = temp.reshape(temp.shape[0], temp.shape[1], -1)
    temp = ((torch.matmul(temp, temp.transpose(dim0=2, dim1=1))))
    temp = temp.reshape([temp.shape[0], -1])  # .sum(dim=2)
    temp = (temp.sign()*torch.abs(temp)**(1/p)).reshape(temp.shape[0], -1)

    return temp


def G_p_pro(ob, p=8):
    temp = ob.detach()
    temp = temp**p
    temp = temp.reshape(temp.shape[0], temp.shape[1], -1)
    temp = ((torch.matmul(temp, temp.transpose(dim0=2, dim1=1))))
    # temp = temp.reshape([temp.shape[0],-1])#.sum(dim=2)
    # .reshape(temp.shape[0],ob.shape[1],ob.shape[1])
    temp = (temp.sign()*torch.abs(temp)**(1/p))

    return temp


def G_p_inf(ob, p=1):
    temp = ob.detach()
    temp = temp**p
    # print(temp.shape)
    temp = temp.reshape([temp.shape[0], temp.shape[1], -1]
                        ).transpose(dim0=2, dim1=1).reshape([-1, temp.shape[1], 1])
    # print(temp.shape)
    temp = ((torch.matmul(temp, temp.transpose(dim0=2, dim1=1))))
    temp = (temp.sign()*torch.abs(temp)**(1/p))
    # print(temp.shape)
    return temp.reshape(ob.shape[0], ob.shape[2], ob.shape[3], ob.shape[1], ob.shape[1])

# import methods.pooling.MPNConv as MPN


class BackboneAndClassifier(nn.Module):

    def __init__(self, num_classes, config):
        super().__init__()
        # clsblock = {'linear': LinearClassifier,
        #             'pcssr': CSSRClassifier, 'rcssr': CSSRClassifier}

        clsblock = {'pcssr': CSSRClassifier, 'rcssr': CSSRClassifier}
        self.backbone = Backbone(config, 3)
        cat_config = config['category_model']
        # self.cat_cls = clsblock[cat_config['model']](
        #     self.backbone.output_dim, num_classes, cat_config)

        self.cat_cls = clsblock['pcssr'](
            self.backbone.output_dim, num_classes, cat_config)

    def forward(self, x, feature_only=False):
        x = self.backbone(x)
        logits = self.cat_cls(x)
        if feature_only:
            return x
        return x, logits


class CSSRModel(nn.Module):

    def __init__(self, num_classes, config, crt):
        super().__init__()
        self.crt = crt

        # ------ New Arch
        self.backbone_cs = BackboneAndClassifier(num_classes, config)

        self.config = config
        self.mins = {i: [] for i in range(num_classes)}
        self.maxs = {i: [] for i in range(num_classes)}
        self.num_classes = num_classes

        self.avg_feature = [[0, 0] for i in range(num_classes)]
        self.avg_gram = [[[0, 0]
                          for i in range(num_classes)] for i in self.powers]
        self.enable_gram = config['enable_gram']

    def update_minmax(self, feat_list, power=[], ypred=None):
        # feat_list = self.gram_feature_list(batch)
        for pr in range(self.num_classes):
            cond = ypred == pr
            if not cond.any():
                continue
            for L, feat_L in enumerate(feat_list):
                if L == len(self.mins[pr]):
                    self.mins[pr].append([None]*len(power))
                    self.maxs[pr].append([None]*len(power))

                for p, P in enumerate(power):
                    g_p = G_p(feat_L[cond], P)

                    current_min = g_p.min(dim=0, keepdim=True)[0]
                    current_max = g_p.max(dim=0, keepdim=True)[0]

                    if self.mins[pr][L][p] is None:
                        self.mins[pr][L][p] = current_min
                        self.maxs[pr][L][p] = current_max
                    else:
                        self.mins[pr][L][p] = torch.min(
                            current_min, self.mins[pr][L][p])
                        self.maxs[pr][L][p] = torch.max(
                            current_max, self.maxs[pr][L][p])

    def get_deviations(self, feat_list, power, ypred):
        batch_deviations = None
        for pr in range(self.num_classes):
            mins, maxs = self.mins[pr], self.maxs[pr]
            cls_batch_deviations = []
            cond = ypred == pr
            if not cond.any():
                continue
            for L, feat_L in enumerate(feat_list):
                dev = 0
                for p, P in enumerate(power):
                    g_p = G_p(feat_L[cond], P)
                    # print(L,len(mins))
                    # print(p,len(mins[L]))
                    dev += (F.relu(mins[L][p]-g_p)/torch.abs(mins[L]
                            [p]+10**-6)).sum(dim=1, keepdim=True)
                    dev += (F.relu(g_p-maxs[L][p])/torch.abs(maxs[L]
                            [p]+10**-6)).sum(dim=1, keepdim=True)
                cls_batch_deviations.append(dev.cpu().detach().numpy())
            cls_batch_deviations = np.concatenate(cls_batch_deviations, axis=1)
            if batch_deviations is None:
                batch_deviations = np.zeros(
                    [ypred.shape[0], cls_batch_deviations.shape[1]])
            batch_deviations[cond] = cls_batch_deviations
        return batch_deviations

    powers = [8]

    def cal_feature_prototype(self, feat, ypred):
        feat = torch.abs(feat)
        for pr in range(self.num_classes):
            cond = ypred == pr
            if not cond.any():
                continue
            csfeat = feat[cond]
            cf = csfeat.mean(dim=[0, 2, 3])  # .cpu().numpy()
            # print(cf.shape)
            ct = cond.sum()
            ft = self.avg_feature[pr]
            self.avg_feature[pr] = [ft[0] + ct,
                                    (ft[1] * ft[0] + cf * ct)/(ft[0] + ct)]
            if self.enable_gram:
                for p in range(len(self.powers)):
                    gram = G_p_pro(csfeat, self.powers[p]).mean(dim=0)
                    gm = self.avg_gram[p][pr]
                    self.avg_gram[p][pr] = [gm[0] + ct,
                                            (gm[1] * gm[0] + gram * ct)/(gm[0] + ct)]

    def obtain_usable_feature_prototype(self):
        if isinstance(self.avg_feature, list):
            clsft_lost = []
            exm = None
            for x in self.avg_feature:
                if x[0] > 0:
                    clsft_lost.append(x[1])
                    exm = x[1]
                else:
                    clsft_lost.append(None)
            clsft = torch.stack(
                [torch.zeros_like(exm) if x is None else x for x in clsft_lost])
            # print(clsft.shape)
            clsft /= clsft.sum(dim=0)  # **2
            # clsft /= clsft.sum(dim = 1,keepdim = True)
            # print(clsft)
            self.avg_feature = clsft.reshape(
                [clsft.shape[0], 1, clsft.shape[1], 1, 1])
            if self.enable_gram:
                for i in range(len(self.powers)):
                    self.avg_gram[i] = torch.stack([x[1] if x[0] > 0 else torch.zeros(
                        [exm.shape[0], exm.shape[0]]).cuda() for x in self.avg_gram[i]])
            # self.avg_gram /= self.avg_gram.sum(dim = 0)
            # print(self.avg_gram.shape)
        return self.avg_feature, self.avg_gram

    def get_feature_prototype_deviation(self, feat, ypred):
        # feat = torch.abs(feat)
        avg_feature, _ = self.obtain_usable_feature_prototype()
        scores = np.zeros([feat.shape[0], feat.shape[2], feat.shape[3]])
        for pr in range(self.num_classes):
            cond = ypred == pr
            if not cond.any():
                continue
            scores[cond] = (avg_feature[pr] * feat[cond]
                            ).mean(axis=1).cpu().numpy()
        return scores

    def get_feature_gram_deviation(self, feat, ypred):
        _, avg_gram = self.obtain_usable_feature_prototype()
        scores = np.zeros([feat.shape[0], feat.shape[2], feat.shape[3]])
        for pr in range(self.num_classes):
            cond = ypred == pr
            if not cond.any():
                continue
            res = 0
            for i in range(len(self.powers)):
                gm = G_p_pro(feat[cond], p=self.powers[i])
                # scores[cond] = (gm / gm.mean(dim = [3,4],keepdim = True) * avg_gram[pr]).sum(dim = [3,4]).cpu().numpy()
                res += (gm * avg_gram[i][pr]).sum(dim=[1,
                                                       2], keepdim=True).cpu().numpy()
            scores[cond] = res
        return scores

    def pred_by_feature_gram(self, feat):
        _, avg_gram = self.obtain_usable_feature_prototype()
        scores = np.zeros([self.num_classes, feat.shape[0]])
        gm = G_p_pro(feat)
        for pr in range(self.num_classes):
            # scores[cond] = (gm / gm.mean(dim = [3,4],keepdim = True) * avg_gram[pr]).sum(dim = [3,4]).cpu().numpy()
            scores[pr] = (gm * avg_gram[pr]).sum(dim=[1, 2]).cpu().numpy()
        return scores.argmax(axis=0)

    def forward(self, x, ycls=None, reqpredauc=False, reqfeature=False):

        # ----- New Arch
        x, logits = self.backbone_cs(x, feature_only=reqfeature)
        # TODO: reshape the logits. current shape is (128, 6, 8, 8), reshape by changing the output dimension of network
        if reqfeature:
            return x
        x, xcls_raw = x, logits

        # xcls is the loss value obtained from the CSSRCriterion class
        xcls, targets, outlinear, outdistance, (anL, tupL) = self.crt(
            xcls_raw, ycls)

        return xcls, targets, outlinear, outdistance, (anL, tupL)


class CSSRCriterion(nn.Module):

    def get_onehot_label(self, y, clsnum):
        """create one hot encoded vector
        1. create the tensor of zeros of shape number of rows in y and clsnum.
        2. .scatter_(1,y,1) will fills the one hot encoding for each label in y
        For ex:
        (1,0,0)
        (0,1,0)
        (0,0,1)
        Args:
            y (_type_): _description_
            clsnum (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Reshape y to 1D and then convert to the tensor of integers
        y = torch.reshape(y, [-1, 1]).long()
        return torch.zeros(y.shape[0], clsnum).cuda().scatter_(1, y, 1)

    def __init__(self, avg_order, num_classes, mapping, enable_sigma=True):
        super().__init__()
        self.avg_order = {"avg_softmax": 1, "softmax_avg": 2}[avg_order]
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.enable_sigma = enable_sigma
        self.num_classes = num_classes
        self.anchors = nn.Parameter(torch.zeros(
            self.num_classes, self.num_classes).double(), requires_grad=False)

        with open("datasets/config.json") as config_file:
            self.cfg = json.load(config_file)["CIFAR10"]
        _, _, _, self.mapping = get_train_loaders(
            "CIFAR10", 0, self.cfg
        )

    def distance_classifier(self, x):
        ''' Calculates euclidean distance from x to each class anchor
                Returns n x m array of distance from input of batch_size n to anchors of size m
        '''

        n = x.size(0)
        m = self.num_classes
        d = self.num_classes

        x = x.unsqueeze(1).expand(n, m, d).double()
        anchors = self.anchors.unsqueeze(0).expand(n, m, d).cuda()
        dists = torch.norm(x-anchors, 2, 2)

        return dists

    def CACLoss(self, distances, gt):
        '''Returns CAC loss, as well as the Anchor and Tuplet loss components separately for visualisation.'''
        true = torch.gather(distances, 1, gt.view(-1, 1)).view(-1).cuda()
        non_gt = torch.Tensor([[i for i in range(self.num_classes) if gt[x] != i]
                              for x in range(len(distances))]).long().cuda()
        others = torch.gather(distances, 1, non_gt)

        anchor = torch.mean(true)

        tuplet = torch.exp(-others+true.unsqueeze(1))
        tuplet = torch.mean(torch.log(1+torch.sum(tuplet, dim=1)))

        total = 0.1*anchor + tuplet

        return total, anchor, tuplet

    def forward(self, x, y=None, prob=False, pred=False):

        outlinear = self.avg_pool(x).view(x.shape[0], -1)
        outdistance = self.distance_classifier(outlinear)

        if y is not None:
            targets = torch.Tensor([self.mapping[xi]
                                   for xi in y]).long().cuda()
            cacLoss, anchorL, tupletL = self.CACLoss(outdistance, targets)

        if prob:
            return outdistance
        if pred:
            return torch.argmax(outdistance, dim=1)

        return cacLoss, targets, outlinear, outdistance, (anchorL, tupletL)


def manual_contrast(x):
    s = random.uniform(0.1, 2)
    return x * s

    def __init__(self, labeled_ds, config, inchan_num=3) -> None:
        super().__init__()
        self.labeled_ds = labeled_ds

        __mean = [0.5, 0.5, 0.5][:inchan_num]
        __std = [0.25, 0.25, 0.25][:inchan_num]

        trans = [transforms.RandomHorizontalFlip()]
        if config['cust_aug_crop_withresize']:
            trans.append(transforms.RandomResizedCrop(
                size=util.img_size, scale=(0.25, 1)))
        elif util.img_size > 200:
            trans += [transforms.Resize(256),
                      transforms.RandomResizedCrop(util.img_size)]
        else:
            trans.append(transforms.RandomCrop(size=util.img_size,
                                               padding=int(
                                                   util.img_size*0.125),
                                               padding_mode='reflect'))
        if config['strong_option'] == 'RA':
            trans.append(RandAugmentMC(n=2, m=10))
        elif config['strong_option'] == 'CUST':
            trans.append(HighlyCustomizableAugment(
                2, 10, -1, labeled_ds, config))
        elif config['strong_option'] == 'NONE':
            pass
        else:
            raise NotImplementedError()
        trans += [transforms.ToTensor(),
                  transforms.Normalize(mean=__mean, std=__std)]

        if config['manual_contrast']:
            trans.append(manual_contrast)
        strong = transforms.Compose(trans)

        if util.img_size > 200:
            self.simple = [transforms.RandomResizedCrop(util.img_size)]
        else:
            self.simple = [transforms.RandomCrop(size=util.img_size,
                                                 padding=int(
                                                     util.img_size*0.125),
                                                 padding_mode='reflect')]
        self.simple = transforms.Compose(([transforms.RandomHorizontalFlip()]) + self.simple + [
            transforms.ToTensor(),
            transforms.Normalize(mean=__mean, std=__std)] + ([manual_contrast] if config['manual_contrast'] else []))

        self.test_normalize = transforms.Compose([
            transforms.CenterCrop(util.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=__mean, std=__std)])

        td = {'strong': strong, 'simple': self.simple}
        self.aug = td[config['cat_augmentation']]
        self.test_mode = False

    def __len__(self) -> int:
        return len(self.labeled_ds)

    def __getitem__(self, index: int):
        img, lb, _ = self.labeled_ds[index]
        if self.test_mode:
            img = self.test_normalize(img)
        else:
            img = self.aug(img)
        return img, lb, index


best_acc = 0
best_cac = 10000
best_anchor = 10000
start_epoch = 0


@util.regmethod('cssr')
class CSSRMethod:

    def get_cfg(self, key, default):
        return self.config[key] if key in self.config else default

    def __init__(self, config, clssnum, trainloader, valloader, mapping) -> None:
        self.config = config
        self.epoch = 0
        self.lr = config['learn_rate']
        self.batch_size = config['batch_size']

        self.clsnum = clssnum
        self.crt = CSSRCriterion(config['arch_type'], clssnum, mapping, False)
        # with open("datasets/config.json") as config_file:
        #     cfg = json.load(config_file)["CIFAR10"]

        self.trainloader, self.valloader = trainloader, valloader
        # Here building the whole network. The CSSRModel class contains the build of alll
        # model with backbone and classifier and loss function
        self.model = CSSRModel(self.clsnum, config, self.crt).cuda()
        self.modelopt = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, weight_decay=5e-4)

        # self.wrap_ds = WrapDataset(train_set, self.config, inchan_num=3,)
        # self.wrap_loader = data.DataLoader(self.wrap_ds,
        #                                    batch_size=self.config['batch_size'], shuffle=True, pin_memory=True, num_workers=0)
        self.lr_schedule = util.get_scheduler(self.config, self.trainloader)

        self.prepared = -999

    def train_epoch(self, epoch):
        print("\nEpoch: %d" % epoch)
        self.model.train()
        train_loss = 0
        correctDist = 0
        total = 0

        for batch_idx, (inputs, tgts) in enumerate(self.trainloader):

            self.lr = self.lr_schedule.get_lr(self.epoch, batch_idx, self.lr)
            util.set_lr([self.modelopt], self.lr)
            sx, lb = inputs.cuda(), tgts.cuda()
            loss, targets, _, outdistance, _ = self.model(
                sx, lb, reqpredauc=True)

            self.modelopt.zero_grad()
            # This back propagation of the loss should be the loss from CAC.
            loss.backward()
            self.modelopt.step()

            # outputs = net(inputs)
            # cacLoss, anchorLoss, tupletLoss = CACLoss(outputs[1], targets)

            train_loss += loss.item()

            _, predicted = outdistance.min(1)

            total += targets.size(0)
            correctDist += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(self.trainloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    train_loss / (batch_idx + 1),
                    100.0 * correctDist / total,
                    correctDist,
                    total,
                ),
            )

    def val(self, epoch, dataset_name, trial):
        global best_acc
        global best_anchor
        global best_cac
        self.model.eval()
        anchor_loss = 0
        cac_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, tgts) in enumerate(self.valloader):
                inputs = inputs.cuda()
                tgts = tgts.cuda()
                loss, targets, _, outdistance, (anchorLoss, tupletLoss) = self.model(
                    inputs, tgts, reqpredauc=True)

                # outputs = net(inputs)
                # cacLoss, anchorLoss, tupletLoss = CACLoss(outputs[1], targets)
                anchor_loss += anchorLoss
                cac_loss += loss

                _, predicted = outdistance.min(1)

                total += targets.size(0)

                correct += predicted.eq(targets).sum().item()

                progress_bar(
                    batch_idx,
                    len(self.valloader),
                    "Acc: %.3f%% (%d/%d)" % (100.0 * correct /
                                             total, correct, total),
                )

        anchor_loss /= len(self.valloader)
        cac_loss /= len(self.valloader)
        acc = 100.0 * correct / total

        # Save checkpoint.
        state = {
            "net": self.model.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir("networks/weights/{}".format(dataset_name)):
            os.makedirs("networks/weights/{}".format(dataset_name))
        if dataset_name == "CIFAR+10":
            if not os.path.isdir("networks/weights/CIFAR+50"):
                os.mkdir("networks/weights/CIFAR+50")
        save_name = "{}_{}CACclassifier".format(
            dataset_name, trial)
        if anchor_loss <= best_anchor:
            print("Saving..")
            torch.save(
                state,
                "networks/weights/{}/".format(dataset_name) +
                save_name + "AnchorLoss.pth",
            )
            best_anchor = anchor_loss

            if dataset_name == "CIFAR+10":
                save_name = save_name.replace("+10", "+50")
                torch.save(
                    state, "networks/weights/CIFAR+50/" + save_name + "AnchorLoss.pth"
                )

        if cac_loss <= best_cac:
            print("Saving..")
            torch.save(
                state,
                "networks/weights/{}/".format(dataset_name) +
                save_name + "CACLoss.pth",
            )
            best_cac = cac_loss
            if dataset_name == "CIFAR+10":
                save_name = save_name.replace("+10", "+50")
                torch.save(state, "networks/weights/CIFAR+50/" +
                           save_name + "CACLoss.pth")

        if acc >= best_acc:
            print("Saving..")
            torch.save(
                state,
                "networks/weights/{}/".format(dataset_name) +
                save_name + "Accuracy.pth",
            )
            best_acc = acc

            if dataset_name == "CIFAR+10":
                save_name = save_name.replace("+10", "+50")
                torch.save(state, "networks/weights/CIFAR+50/" +
                           save_name + "Accuracy.pth")

    def save_model(self, path):
        save_dict = {
            'model': self.model.state_dict(),
            'config': self.config,
            'optimzer': self.modelopt.state_dict(),
            'epoch': self.epoch,
        }
        torch.save(save_dict, path)

    def load_model(self, path):
        save_dict = torch.load(path)
        self.model.load_state_dict(save_dict['model'])
        if 'optimzer' in save_dict and self.modelopt is not None:
            self.modelopt.load_state_dict(save_dict['optimzer'])
        self.epoch = save_dict['epoch']
