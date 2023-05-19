'https://arxiv.org/pdf/1311.2524.pdf'
'https://github.com/object-detection-algorithm/R-CNN'
'https://github.com/Hulkido/RCNN/blob/master/RCNN.ipynb'
'https://medium.com/codex/implementing-r-cnn-object-detection-on-voc2012-with-pytorch-b05d3c623afe'
'https://colab.research.google.com/drive/1nCj54XryHcoMARS4cSxivn3Ci1I6OtvO?usp=sharing#scrollTo=B6LEi2IWWgLu'

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm=True, **kwargs) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.act = nn.ReLU(inplace=True)
        self.norm = nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75) if norm else nn.Identity()

    def forward(self, x):
        return self.norm(self.act(self.conv(x)))

class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out=0.5) -> None:
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop_out = nn.Dropout(p=drop_out)

    def forward(self, x):
        return self.drop_out(self.act(self.linear(x)))

class RCNN(nn.Module):
    '''
    feature layers are based on AlexNet
    i change kernel size (11, 11) -> (7, 7) to adjust input image size 227 -> 224
    '''

    def __init__(self, in_channels=3, num_classes=21, drop_out=0.5) -> None:
        super().__init__()

        self.feature_layers = nn.Sequential(
            ConvBlock(in_channels, out_channels=96, kernel_size=(7, 7), stride=4, padding=0, norm=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            ConvBlock(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=1, padding=2, norm=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            ConvBlock(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=1, padding=1, norm=False),
            ConvBlock(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=1, padding=1, norm=False),
            ConvBlock(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=1, padding=1, norm=False),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            LinearBlock(256 * 6 * 6, 4096, drop_out=drop_out),
            LinearBlock(4096, 4096, drop_out=drop_out),
            nn.Linear(4096, num_classes)
        )

        self.bbox_predictor = nn.Sequential(
            nn.Flatten(start_dim=1),
            LinearBlock(256 * 6 * 6, 4096, drop_out=drop_out),
            LinearBlock(4096, 4096, drop_out=drop_out),
            nn.Linear(4096, 4)
        )

    def forward(self, x, bbox=True):
        feature_map = self.feature_layers(x)
        class_pred = self.classifier(feature_map)
        bbox_pred = self.bbox_predictor(feature_map) if bbox else None
        return class_pred, bbox_pred


from torchvision.models import efficientnet_b0
class RCNN(nn.Module):
    ''' no improvement with alexnet backbone... why? '''

    def __init__(self, in_channels=3, num_classes=21, drop_out=0.5):
        super().__init__()

        self.feature_layers = efficientnet_b0(in_channels=in_channels, pre_trained=True)
        self.feature_layers._fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            LinearBlock(1000, 4096, drop_out=drop_out),
            LinearBlock(4096, 4096, drop_out=drop_out),
            nn.Linear(4096, num_classes)
        )

        self.bbox_predictor = nn.Sequential(
            nn.Flatten(start_dim=1),
            LinearBlock(1000, 4096, drop_out=drop_out),
            LinearBlock(4096, 4096, drop_out=drop_out),
            nn.Linear(4096, 4)
        )

    def forward(self, x, bbox=True):
        feature_map = self.feature_layers(x)
        class_pred = self.classifier(feature_map)
        bbox_pred = self.bbox_predictor(feature_map) if bbox else None
        return class_pred, bbox_pred

if __name__ == '__main__':
    model = RCNN(in_channels=3, num_classes=21, drop_out=0.2)
    x = torch.randn(1, 3, 224, 224)
    class_pred, bbox_pred = model(x, bbox=True)
    print(class_pred.shape, bbox_pred.shape)