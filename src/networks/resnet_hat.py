import torch
import torch.nn as nn
import torch.nn.functional as F

class BN_Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False, activation=True, mask_ids=[]):
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias),
                  nn.BatchNorm2d(out_channels)]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
        self.mask_ids = mask_ids

    def forward(self, x):
        return self.layers(x)

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, strides, num_tasks, masks_split):
        super(BasicBlock, self).__init__()
        self.conv1 = BN_Conv2d(in_channels, out_channels, 3, stride=strides, padding=1, bias=False, mask_ids=[masks_split[0] - 1, masks_split[0]])
        self.conv2 = BN_Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, activation=False, mask_ids=[masks_split[0], masks_split[0] + 1])
        self.short_cut = BN_Conv2d(in_channels, out_channels, 1, stride=strides, padding=0, bias=False, activation=False, mask_ids=[masks_split[0] - 1, masks_split[0] + 1]) if strides != 1 else nn.Sequential()

        self.gate = torch.sigmoid
        self.ec1 = torch.nn.Embedding(num_tasks, out_channels)
        self.ec2 = torch.nn.Embedding(num_tasks, out_channels)
        self.masks_split = masks_split

    def forward(self, x, masks):
        gcs = masks[self.masks_split[0]:self.masks_split[1]]
        out = self.conv1(x)
        out = self.mask_out(out, gcs[0])
        out = self.conv2(out)
        out = F.relu(out + self.short_cut(x))
        out = self.mask_out(out, gcs[1])
        return out

    def mask(self, t, s=1):
        gc1 = self.gate(s * self.ec1(t))
        gc2 = self.gate(s * self.ec2(t))
        return [gc1, gc2]

    def mask_out(self, out, mask):
        mask = mask.view(1, -1, 1, 1)
        out = out * mask.expand_as(out)
        return out


class BottleNeck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, strides, num_tasks, masks_split):
        super(BottleNeck, self).__init__()
        self.conv1 = BN_Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False, mask_ids=[masks_split[0] - 1, masks_split[0]])  # same padding
        self.conv2 = BN_Conv2d(out_channels, out_channels, 3, stride=strides, padding=1, bias=False, mask_ids=[masks_split[0], masks_split[0] + 1])  # same padding
        self.conv3 = BN_Conv2d(out_channels, out_channels * 4, 1, stride=1, padding=0, bias=False, activation=False, mask_ids=[masks_split[0] + 1, masks_split[0] + 2])  # same padding
        self.shortcut = BN_Conv2d(in_channels, out_channels * 4, 1, stride=strides, padding=0, bias=False, activation=False, mask_ids=[masks_split[0] - 1, masks_split[0] + 2]) if strides != 1 or in_channels != out_channels * 4 else nn.Sequential()

        self.gate = torch.sigmoid
        self.ec1 = torch.nn.Embedding(num_tasks, out_channels)
        self.ec2 = torch.nn.Embedding(num_tasks, out_channels)
        self.ec3 = torch.nn.Embedding(num_tasks, out_channels * 4)

        self.masks_split = masks_split

    def forward(self, x, masks):
        gcs = masks[self.masks_split[0]:self.masks_split[1]]
        out = self.conv1(x)
        out = self.mask_out(out, gcs[0])
        out = self.conv2(out)
        out = self.mask_out(out, gcs[1])
        out = self.conv3(out) 
        out = F.relu(out + self.shortcut(x))
        out = self.mask_out(out, gcs[2])
        return out

    def mask(self, t, s=1):
        gc1 = self.gate(s * self.ec1(t))
        gc2 = self.gate(s * self.ec2(t))
        gc3 = self.gate(s * self.ec3(t))
        return [gc1, gc2, gc3]

    def mask_out(self, out, mask):
        mask = mask.view(1, -1, 1, 1)
        out = out * mask.expand_as(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, groups, taskcla):
        super(ResNet, self).__init__()
        self.channels = 64
        self.block = block
        self.taskcla = taskcla
        self.masks_split = {}
        self.masks_id_count = 1

        self.conv1 = nn.Conv2d(3, self.channels, 7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(self.channels)
        self.pool1 = nn.MaxPool2d(3, 2, 1)

        self.gate = torch.sigmoid
        self.ec = torch.nn.Embedding(len(taskcla), self.channels)

        self.layer1 = self._make_layers(channels=64, blocks=groups[0], strides=1, index=1)
        self.layer2 = self._make_layers(channels=128, blocks=groups[1], strides=2, index=2)
        self.layer3 = self._make_layers(channels=256, blocks=groups[2], strides=2, index=3)
        self.layer4 = self._make_layers(channels=512, blocks=groups[3], strides=2, index=4)
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, channels, blocks, strides, index):
        self.masks_split[str('layer%d' % index)] = {}
        list_strides = [strides] + [1] * (blocks - 1)
        layers = nn.ModuleList()
        for i in range(len(list_strides)):
            layer_name = str("block%d" % (i))
            masks_num = 2 if self.block == BasicBlock else 3
            self.masks_split[str('layer%d' % index)][layer_name] = [self.masks_id_count, self.masks_id_count + masks_num]
            layers.add_module(layer_name, self.block(self.channels, channels, list_strides[i], len(self.taskcla), [self.masks_id_count, self.masks_id_count + masks_num]))
            self.masks_id_count += masks_num
            self.channels = channels * self.block.expansion
        return layers

    def masks(self, t, s=1):
        masks = []
        masks.append(self.gate(s * self.ec(t)))
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                block_masks = block.mask(t, s)
                masks.extend(block_masks)
        return masks
    
    def mask_out(self, out, mask):
        mask = mask.view(1, -1, 1, 1)
        out = out * mask.expand_as(out)
        return out

    def forward(self, x, t, s=1):
        masks = self.masks(t, s)
        out = self.conv1(x)
        out = F.relu(self.bn(out))
        out = self.pool1(out)
        out = self.mask_out(out, masks[0])
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                out = block(out, masks)
        out = self.pool2(out)
        out = torch.flatten(out, 1)
        return out, masks

def ResNet_18(taskcla):
    return ResNet(block=BasicBlock, groups=[2, 2, 2, 2], taskcla=taskcla)

def ResNet_34(taskcla):
    return ResNet(block=BasicBlock, groups=[3, 4, 6, 3], taskcla=taskcla)

def ResNet_50(taskcla):
    return ResNet(block=BottleNeck, groups=[3, 4, 6, 3], taskcla=taskcla)

def ResNet_101(taskcla):
    return ResNet(block=BottleNeck, groups=[3, 4, 23, 3], taskcla=taskcla)

def ResNet_152(taskcla):
    return ResNet(block=BottleNeck, groups=[3, 8, 36, 3], taskcla=taskcla)

class Net(torch.nn.Module):

    def __init__(self, inputsize, taskcla):
        super(Net,self).__init__()
        ncha, size, _ = inputsize
        self.taskcla = taskcla

        self.backbone = ResNet_18(taskcla)
        self.classifiers = torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.classifiers.append(torch.nn.Linear(512 * self.backbone.block.expansion, n))

    def forward(self, t, x, s=1):
        features, masks = self.backbone(x, t, s)
        out = []
        for i,_ in self.taskcla:
            out.append(self.classifiers[i](features))
        return out, masks

    def mask(self, t, s=1):
        return self.backbone.masks(t, s)

    def get_view_for(self, name, masks):
        name = name.replace('backbone.', '')
        if name == 'conv1.weight':
            return masks[0].data.view(-1,1,1,1).expand_as(self.backbone.conv1.weight)
        sub_names = name.split('.')
        if len(sub_names) < 5 or sub_names[4] == '1':
            return None
        if not sub_names[2].startswith('ec'):
            target_weight = get_sub_attr(self.backbone, sub_names[:4])[0].weight
            post = masks[get_sub_attr(self.backbone, sub_names[:3]).mask_ids[1]].data.view(-1, 1, 1, 1).expand_as(target_weight)
            pre = masks[get_sub_attr(self.backbone, sub_names[:3]).mask_ids[0]].data.view(1, -1, 1, 1).expand_as(target_weight)
            return torch.min(post, pre)
        return None

def get_sub_attr(obj, sub_names):
    for sub_name in sub_names:
        obj = getattr(obj, sub_name)
    return obj







