from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def parse_cfg(cfg_file_path):
    file = open(cfg_file_path, 'r')
    lines = file.read().split('\n')                         # cfg 파일의 line들을 한줄씩 읽어와서 list에 저장
    lines = [line for line in lines if len(line) > 0]       # 비어있는 line은 제거하고
    lines = [line for line in lines if line[0] != '#']      # comments도 무시하고
    lines = [line.rstrip().lstrip() for line in lines]      # 텍스트 양옆의 빈칸들 삭제

    model_block = {}
    network_structure = []

    for line in lines:
        if line[0] == "[":
            if len(model_block) != 0:
                network_structure.append(model_block)
                model_block = {}
            model_block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            model_block[key.rstrip()] = value.lstrip()
    network_structure.append(model_block)

    return network_structure

def create_network_modules(network_structure):
    network_info = network_structure[0]                     #[net] block에 쓰여있는 정보들은 신경망 입력과 training parameters에 대한 설명으로, Layer 정보가 아님
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for idx, mblock in enumerate(network_structure[1:]):
        module = nn.Sequential()

        if (mblock["type"] == "convolutional"):
            activation = mblock["activation"]
            try:
                batch_normalize = int(mblock["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(mblock["filters"])
            padding = int(mblock["pad"])
            kernel_size = int(mblock["size"])
            stride = int(mblock["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            convolutional_layer = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(idx), convolutional_layer)

            if batch_normalize:
                batch_norm_layer = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(idx), batch_norm_layer)

            if activation == "leaky":
                activation_layer = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(idx), activation_layer)
        
        elif (mblock["type"] == "upsample"):
            stride = int(mblock["stride"])
            upsample_layer = nn.Upsample(scale_factor=2, mode="nearest")
            module.add_module("upsample_{}".format(idx), upsample_layer)

        elif (mblock["type"] == "route"):
            mblock["layers"] = mblock["layers"].split(',')
            # Route 시작점
            start = int(mblock["layers"][0])
            # end가 있다면 거기가 end
            try:
                end = int(mblock["layers"][1])
            except:
                end = 0
            if start > 0:
                start = start - idx
            if end > 0:
                end = end - idx
            route_layer = EmptyLayer()
            module.add_module("route_{0}".format(idx), route_layer)
            if end < 0:
                filters = output_filters[idx + start] + output_filters[idx + end]
            else:
                filters = output_filters[idx + start]

        elif (mblock["type"] == "shortcut"):
            shortcut_layer = EmptyLayer()
            module.add_module("shortcut_{}".format(idx), shortcut_layer)
        
        elif (mblock["type"] == "yolo"):
            mask = mblock["mask"].split(",")
            mask = [int(m) for m in mask]

            anchors = mblock["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection_layer = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(idx), detection_layer)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (network_info, module_list)

# 신경망 정의하기
class DarkNet(nn.Module):
    def __init__(self, cfg_file_path):
        super(DarkNet, self).__init__()
        self.network_structure = parse_cfg(cfg_file_path)
        self.network_info, self.module_list = create_network_modules(self.network_structure)

    # 신경망의 순전파 구현하기
    # nn.Module 클래스의 forward method를 overriding하여 구현
    def forward(self, x, CUDA):
        modules = self.network_structure[1:]
        # route layer와 shortcut layer는 이전 layer들로부터의 output map이 필요하기 때문에,
        # 모든 layer의 output feature maps를 dict() 형태의 outputs에 저장해야 한다.
        # outputs에서 key는 layer index, value는 feature map이다.
        outputs = {}

        # module_list를 순회하며 신경망을 구성하는 module들을 순서대로 거치며 input를 순전파 시킨다.
        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])
            
            # convolution type과 upsample 타입의 경우 완전한 순전파
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(layer) for layer in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    # route는 경우에 따라 두 feature map을 연결해야 함
                    feature_map_1 = outputs[i + layers[0]]
                    feature_map_2 = outputs[i + layers[1]]

                    # 두 feature map을 연결하기 위해 torch.cat 함수를 두번째 인자를 1로 하여 사용
                    x = torch.cat((feature_map_1, feature_map_2), 1)

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]

            # yolo의 출력값은 feature map의 깊이에 따른 바운딩 박스 속성을 포함하고 있는
            # convolutional feature map이다.
            elif module_type == "yolo":
                anchors = self.module_list[i][0].anchors






