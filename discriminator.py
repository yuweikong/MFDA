import torch
import torch.nn as nn

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                # m.bias.data.fill_(1e-4)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()
            elif isinstance(m, nn.ModuleList) or isinstance(m, nn.Sequential):
                for mm in m:
                    if isinstance(mm, nn.Conv2d):
                        nn.init.kaiming_normal_(mm.weight.data, nonlinearity='relu')
                    elif isinstance(mm, nn.BatchNorm2d):
                        mm.weight.data.fill_(1.)
                        # mm.bias.data.fill_(1e-4)
                        mm.bias.data.zero_()
                    elif isinstance(mm, nn.Linear):
                        mm.weight.data.normal_(0.0, 0.0001)
                        mm.bias.data.zero_()

class FCDiscriminator(nn.Module):
    def __init__(self, num_classes):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, num_classes//2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_classes//2, num_classes//4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_classes//4, num_classes//8, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(num_classes//8, 1, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        initialize_weights(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x


class OutspaceDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf = 32):
        super(OutspaceDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        initialize_weights(self)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x
    


if __name__ == '__main__':
    x = torch.randn(2, 1440, 64, 64)

    model_fc = FCDiscriminator(1440)
    model_out = OutspaceDiscriminator(1440,32)

    fc = model_fc(x)
    out = model_out(x)
    print(fc.shape)
    print(out.shape)
