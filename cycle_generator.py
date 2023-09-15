import networks
import torch

def get_F_G(input_channel = 3):
    F_G = networks.define_G(input_channel, 3, 64, 'resnet_6blocks', 'instance', False, 'normal', 0.02)
    return F_G

def get_F_G_only():
    F_G = networks.define_G(18, 3, 64, 'resnet_6blocks', 'instance', False, 'normal', 0.02)
    return F_G

if __name__ == '__main__':
    net = get_F_G_only()
    img = torch.randn(4,18,224,224)
    output = net(img)
    torch.save(net,'/remote-home/share/42/cyc19307140030/yolov5/feature_distill/F_G_r6.pth')