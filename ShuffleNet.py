import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math

class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3, GConv=True, type_selection='add'):
        
        super(ShuffleNetUnit, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.GConv = GConv
        self.type_selection = type_selection
        
        self.bottleneck_channels = self.out_channels // 4


        # Add or Concat?
        # ShuffleNetUnits Figure 2b
        if self.type_selection == 'add':
            self.DW_stride = 1
            self.selected_function = self._add
        # ShuffleNetUnits Figure 2c   
        elif self.type_selection == 'concat':
            self.DW_stride = 2
            self.selected_function = self._concat
            
            # 保证输出channels正确
            self.out_channels -= self.in_channels
        else:
            raise ValueError("Wrong type!")

        

        # 第一层可能用可能不用gconv
        # Note that for Stage 2, we do not apply group convolution on the first pointwise layer because the number of input channels is relatively small.

        # layer 1: the first pointwise layer
        self.first_1x1_GConv = self._1x1_GConv(
            self.in_channels,
            self.bottleneck_channels,
            self.groups if GConv else 1,
            ReLU=True
            )

        # layer 2: 3 × 3 depthwise convolution padding=1保证输出输入维度相同
        self._3x3_DWConv = nn.Conv2d(self.bottleneck_channels, self.bottleneck_channels, kernel_size=3, padding=1, stride=self.DW_stride, groups=self.bottleneck_channels, bias = False)
        self.DW_BN = nn.BatchNorm2d(self.bottleneck_channels)

        # layer 3: the second pointwise layer
        self.second_1x1_GConv = self._1x1_GConv(
            self.bottleneck_channels,
            self.out_channels,
            self.groups,
            ReLU=False
            )


    @staticmethod
    # element-wise addition
    def _add(x, out):
        return x + out


    @staticmethod
    # channel concatenation
    def _concat(x, out):
        return torch.cat((x, out), 1)

    @staticmethod
    def channel_shuffle(x, groups):
        batchsize, num_channels, height, width = x.data.size()

        channels_per_group = num_channels // groups
        
        # reshape g*n n=channels_per_group
        x = x.view(batchsize, groups, channels_per_group, height, width)

        # transpose n*g
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x
    


    def _1x1_GConv(self, in_channels, out_channels, groups, ReLU=False):

        modules = OrderedDict()

        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias = False)
        modules['1x1_Conv'] = conv

        
        modules['BN'] = nn.BatchNorm2d(out_channels)
        
        if ReLU:
            modules['ReLU'] = nn.ReLU(inplace=True)
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv


    def forward(self, x):
        residual = x

        if self.type_selection == 'concat':
            residual = F.avg_pool2d(residual, kernel_size=3, stride=2, padding=1)

        out = self.first_1x1_GConv(x)
        out = self.channel_shuffle(out, self.groups)
        out = self._3x3_DWConv(out)
        out = self.DW_BN(out)
        out = self.second_1x1_GConv(out)
        
        out = self.selected_function(residual, out)
        return F.relu(out)



class ShuffleNet(nn.Module):
    def __init__(self, groups=3, in_channels=3, num_classes=1000, scale_factor = 1):
        super(ShuffleNet, self).__init__()

        self.groups = groups
        self.stage_repeat = [3, 7, 3]
        self.in_channels =  in_channels
        self.num_classes = num_classes

        # Output channels for differnet group number
        if groups == 1:
            self.stage_out_channels = [-1, 24, int(scale_factor * 144), int(scale_factor * 288), int(scale_factor * 576)]
        elif groups == 2:
            self.stage_out_channels = [-1, 24, int(scale_factor * 200), int(scale_factor * 400), int(scale_factor * 800)]
        elif groups == 3:
            self.stage_out_channels = [-1, 24, int(scale_factor * 240), int(scale_factor * 480), int(scale_factor * 960)]
        elif groups == 4:
            self.stage_out_channels = [-1, 24, int(scale_factor * 272), int(scale_factor * 544), int(scale_factor * 1088)]
        elif groups == 8:
            self.stage_out_channels = [-1, 24, int(scale_factor * 384), int(scale_factor * 768), int(scale_factor * 1536)]
        else:
            raise ValueError("Wrong group number!")
        
        # Conv1, MaxPool 调整
        self.Conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stage_out_channels[1], kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(self.stage_out_channels[1]),
            nn.ReLU(inplace=True),
        )
        # 为c10调整的
        #self.MaxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 2
        self.stage2 = self.three_stages(2)
        # Stage 3
        self.stage3 = self.three_stages(3)
        # Stage 4
        self.stage4 = self.three_stages(4)


        # FC
        self.fc = nn.Linear(self.stage_out_channels[-1], self.num_classes)
        self.init_params()


    def init_params(self):
         for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def three_stages(self, stage):
        modules = OrderedDict()

        # Stage_n是三个Stage2，3，4每个内部的stage,repeat部分
        stage_name = "Stage_n{}".format(stage)
        
        # 第一层可能用可能不用gconv
        # Note that for Stage 2, we do not apply group convolution on the first pointwise layer because the number of input channels is relatively small.
        GConv = stage > 2
        
        # Concat
        unit1 = ShuffleNetUnit(
            self.stage_out_channels[stage-1],
            self.stage_out_channels[stage],
            groups = self.groups,
            GConv = GConv,
            type_selection = 'concat'
            )
        modules[stage_name+"_0"] = unit1

        # add, repeat
        for i in range(self.stage_repeat[stage-2]):
            name = stage_name + "_{}".format(i+1)
            unit2 = ShuffleNetUnit(
                self.stage_out_channels[stage],
                self.stage_out_channels[stage],
                groups = self.groups,
                GConv = True,
                type_selection = 'add'
                )
            modules[name] = unit2

        return nn.Sequential(modules)


    def forward(self, x):
        out = self.Conv1(x)
        #out = self.MaxPool(out)

        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        # global average pooling layer 调整
        out = F.adaptive_avg_pool2d(out, 1)
        
        # flatten for FC
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return F.log_softmax(out, dim=1)


if __name__ == "__main__":
    model = ShuffleNet()
