from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tools import *

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
    # filter(=kernel)의 차원에서 높이와 너비는 cfg 파일에서 제공됨
    # 그러나 깊이는 이전 layer의 filter의 수와 일치해야함
    # 따라서 prev_filters 변수로 이전 layer의 filter 수를 추적해야하고, 이전의 다른 layer들의 filter 수도 추적해야하므로
    # 각 block의 출력 filter의 수를 output_filters 리스트에 append하며 추적해야 함
    # 첫 input image의 depth는 3이고(R,G,B) 이를 위한 filter의 개수도 3개이므로 3으로 초기화
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
        
        # Prediction Feature Maps at different Scales를 위해 upsampling할 때 사용
        elif (mblock["type"] == "upsample"):
            stride = int(mblock["stride"])
            upsample_layer = nn.Upsample(scale_factor=2, mode="nearest")
            module.add_module("upsample_{}".format(idx), upsample_layer)

        # layers 속성이 하나의 값을 가질 때
        # 예) -4이면 route layer로부터 4번째 뒤에 있는 layer의 feature map을 출력으로 내보낸다.
        # layers 속성이 두개의 값을 가질때
        # 예) -1, 61이면 route layer의 이전 layer와 61번째 layer에서 깊이 차원에 따라 연결된 feature map을 출력으로 내보낸다.
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
            # 다른 Layer들처럼 nn.Module 객체를 생성하고 forward() 메서드에 지정된 feature map들을 연결/도입하는 연산을 작성할 수도 있지만,
            # 불필요한 코드 작성을 줄이고 간단하게 구현하기 위해
            # route layer는 EmptyLayer()를 통해 dummy layer로 두고
            # nn.Module을 상속받은 Darknet Class의 forward() 메서드에서 torch.cat()을 통해 직접 연결을 수행한다.
            route_layer = EmptyLayer()
            module.add_module("route_{0}".format(idx), route_layer)
            if end < 0:
                filters = output_filters[idx + start] + output_filters[idx + end]
            else:
                filters = output_filters[idx + start]

        # shortcut layer 역시 매우 간단한 연산만 수행하기 때문에 EmptyLayer()를 통해 dummy layer로 두고
        # Darknet Class의 forward() 메서드에서 바로 다음 layer의 feature map에 바로 이전 layer의
        # feature map을 추가해주는 연산을 정의해준다.
        elif (mblock["type"] == "shortcut"):
            shortcut_layer = EmptyLayer()
            module.add_module("shortcut_{}".format(idx), shortcut_layer)
        
        # yolo layer가 detection layer
        # yolov3가 feature map 예측을 수행하는 3개의 scale에서 각각 detection layer를 가지고 있다.
        elif (mblock["type"] == "yolo"):
            mask = mblock["mask"].split(",")
            mask = [int(m) for m in mask]

            # yolov3는 3가지 scale에서 feature map 예측을 수행하고,
            # 각 scale에서의 detection layer마다 하나의 cell은 3개의 box를 예측하기 위해
            # anchors를 사용하므로 anchors는 총 9개이다.
            # 하지만 9개가 한꺼번에 사용되는 것이 아니고 scale당 3개씩 사용되기 때문에
            # mask tag의 속성으로 index된 anchors 3개씩만 사용된다. 
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
    # forward() 메서드의 목적
    # 1. 출력을 계산하는 것
    # 2. 출력 detection feature map을 처리되기 쉬운 방법으로 변환하는 것
    #    yolov3는 3개의 scale에 걸쳐 detection을 위한 prediction map을 계산하기 때문에
    #    서로 다른 차원의 Tensor를 한번에 연산할 수 있게 단일 Tensor로 변환해주는 과정이 필요함
    def forward(self, x, CUDA):
        modules = self.network_structure[1:]
        # route layer와 shortcut layer는 이전 layer들로부터 출력된 feature map이 필요하기 때문에,
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
                    # 이것은 두 feature map을 깊이(channel dimension)에 따라 연결하기 때문
                    x = torch.cat((feature_map_1, feature_map_2), 1)

            # shortcut layer는 ResNet에서 사용하는 것과 유사한 skip connection이다.
            # from 값이 -3이면 shortcut layer의 출력 feature map은
            # 바로 이전 layer의 feature map들과 shortcut layer의 3번째 뒤에 있는 layer의
            # feature map들을 더하여 얻어진다.
            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]

            # yolo의 출력값은 feature map의 깊이에 각 cell이 예측한 3개의 bounding box 속성을 포함하고 있는
            # convolutional feature map이다.
            elif module_type == "yolo":
                anchors = self.module_list[i][0].anchors
                # input 차원 받아오기
                input_dim = int(self.network_info["height"])

                # 분류할 class 개수 받아오기
                num_classes = int(module["classes"])

                # yolov3는 detection이 3개의 서로 다른 scale에서 발생하기 때문에 prediction map의 차원이 서로 다르다.
                # 이 서로 다른 prediction map을 하나의 단일 Tensor로 만들어 쉽게 연산하기 위해 Transform을 시켜준다.
                # tools.py에 정의된 predict_transform() 함수 사용
                x = x.data
                x = predict_transform(x, input_dim, anchors, num_classes, CUDA)
                
                # 위의 Transform을 수행할 때, empty tensor를 초기화할 수 없으므로]
                # 첫번째 detection map을 얻을 때까지 collector(detection map을 지닌 tensor)의 초기화를 지연시킨다.
                # 첫번째 detection map을 얻고 난 이후 연속적으로 detection map을 얻을 때 이것을 연결시킨다.
                if not write:           # write flag는 첫번째 detection map을 얻었는지 아닌지를 나타내는데 사용
                    detections = x      # forward() 메서드에서 루프 바로 전에 write = 0으로 초기화했었음
                    write = 1           # write=0이면 collector가 초기화되지 않은 것, 1이면 초기화가 되었으므로 다른 detection map을 이곳으로 연결시킬 수 있음
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return detections
    
def get_test_input():
    image = cv2.imread("dog-cycle-car.png")
    image = cv2.resize(image, (416,416))
    image_ = image[:,:,::-1].transpose((2,0,1))
    image_ = image_[np.newaxis,:,:,:]/255.0
    image_ = torch.from_numpy(image_).float()
    image_ = Variable(image_)
    return image_

model = DarkNet("cfg\\model-yolov3.cfg")
input_image = get_test_input()
pred = model(input_image, False)
print(pred)