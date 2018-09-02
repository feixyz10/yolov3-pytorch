from models import Yolo3Tiny, Yolo3
import numpy as np
import torch

def load_weights(model, weightfile):
    fopen = open(weightfile, 'rb')
    _ = np.fromfile(fopen, dtype=np.int32, count=5)
    weights = np.fromfile(fopen, dtype=np.float32)

    pos = 0
    for name, child in model.named_children():
        print(name)
        if 'resblock' in name:
            child = list(child.module_list)
        else:
            child = [child]

        for layer in child:
            names = [x[0] for x in layer.named_children()]
            conv = layer.conv
            if 'batchnorm' in names:
                bn = layer.batchnorm
                num_bias = bn.bias.numel()
                bn.bias.data.copy_(torch.from_numpy(weights[pos: pos + num_bias]).view_as(bn.bias.data))
                pos += num_bias
                bn.weight.data.copy_(torch.from_numpy(weights[pos: pos + num_bias]).view_as(bn.bias.data))
                pos += num_bias
                bn.running_mean.data.copy_(torch.from_numpy(weights[pos: pos + num_bias]).view_as(bn.bias.data))
                pos += num_bias
                bn.running_var.data.copy_(torch.from_numpy(weights[pos: pos + num_bias]).view_as(bn.bias.data))
                pos += num_bias
            else:
                num_bias = conv.bias.numel()
                conv.bias.data.copy_(torch.from_numpy(weights[pos: pos + num_bias]).view_as(conv.bias.data))
                pos += num_bias

            num_weight = conv.weight.numel()
            conv.weight.data.copy_(torch.from_numpy(weights[pos: pos + num_weight]).view_as(conv.weight.data))
            pos += num_weight

    fopen.close()

def save_weight_to_pytorch_statedict(model, weightfile, savefile):
    load_weights(model, weightfile)
    torch.save(model.state_dict(), savefile)


# if __name__ == '__main__':
#     yolo3 = Yolo3Tiny()
#     weightfile = 'weights/yolov3-tiny.weights'
#     savefile = 'weights/yolov3_tiny.pth'
#     save_weight_to_pytorch_statedict(yolo3, weightfile, savefile)

if __name__ == '__main__':
    yolo3 = Yolo3()
    weightfile = 'weights/yolov3.weights'
    savefile = 'weights/yolov3.pth'
    save_weight_to_pytorch_statedict(yolo3, weightfile, savefile)