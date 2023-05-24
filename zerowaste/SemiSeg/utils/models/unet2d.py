import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, conv_in_channels, conv_out_channels, up_in_channels=None, up_out_channels=None):
        super().__init__()
        if up_in_channels==None:
            up_in_channels=conv_in_channels
        if up_out_channels==None:
            up_out_channels=conv_out_channels

        self.up = nn.ConvTranspose2d(up_in_channels, up_out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(conv_in_channels, conv_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_out_channels, conv_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(inplace=True)
        )

    # x1-upconv , x2-downconv
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class UNetResNet(nn.Module):
    def __init__(self, 
                 n_classes=1,
                 using_resnet50=False,
                 using_resnet34=False,
                 using_resnet18=False,
                 using_pretrained_model=True,
                 ):
        super().__init__()
        self.n_classes = n_classes
        if using_resnet50:
            backbone = torchvision.models.resnet50(
                pretrained=using_pretrained_model)        
            filters = [256, 512, 1024, 2048]
        elif using_resnet34:
            backbone = torchvision.models.resnet34(
                pretrained=using_pretrained_model)
            filters = [64, 128, 256, 512]
        elif using_resnet18: 
            backbone = torchvision.models.resnet18(
                pretrained=using_pretrained_model)
            filters = [64, 128, 256, 512]
        else:
            raise TypeError("unsupport resnet type.")
        # filters = [64, 128, 256, 512]

        self.firstlayer = nn.Sequential(*list(backbone.children())[:3])
        self.maxpool = list(backbone.children())[3]
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        self.decoder1 = DecoderBlock(conv_in_channels=filters[3], 
                                     conv_out_channels=filters[2])
        self.decoder2 = DecoderBlock(conv_in_channels=filters[2], 
                                     conv_out_channels=filters[1])
        self.decoder3 = DecoderBlock(conv_in_channels=filters[1], 
                                     conv_out_channels=filters[0])
        self.decoder4 = DecoderBlock(
            conv_in_channels=320 if using_resnet50 else filters[1], conv_out_channels=filters[0], 
            up_in_channels=filters[0],
            up_out_channels=filters[0]
        )

        self.last_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=filters[0], 
                               out_channels=filters[0], kernel_size=2, stride=2),
        )
        
        self.semi_last_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=filters[0], 
                               out_channels=filters[0], kernel_size=2, stride=2),
        )        
        self.seg_logits = nn.Sequential(
            nn.Conv2d(filters[0], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, n_classes + 1, kernel_size=1),
        )
        self.feat_aug_seg_logits = nn.Sequential(
            nn.Conv2d(filters[0], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, n_classes + 1, kernel_size=1),
        )        
        self.drop = nn.Dropout2d(0.5)
        
        
    def forward(self, x, add_feats_noise=False, add_feats_drop=False):
        ouputs = {}
        e1 = self.firstlayer(x)
        maxe1 = self.maxpool(e1)
        e2 = self.encoder1(maxe1)
        e3 = self.encoder2(e2)
        e4 = self.encoder3(e3)
        e5 = self.encoder4(e4)
        
        d1 = self.decoder1(e5, e4)
        d2 = self.decoder2(d1, e3)
        d3 = self.decoder3(d2, e2)
        d4 = self.decoder4(d3, e1)
        last_layer = self.last_layer(d4)
        if add_feats_noise:
            batch_size = d4.size(0)
            if add_feats_drop:
                semi_last_layer = self.semi_last_layer(torch.cat((
                        d4[0 : batch_size // 2],
                        self.drop(d4[batch_size // 2 : ]))))
            else:
                semi_last_layer = self.semi_last_layer(d4)
            feat_aug_seg_logits = self.feat_aug_seg_logits(semi_last_layer)
            seg_logits, aug_seg_logits = feat_aug_seg_logits.chunk(2) 
            ouputs.update({"seg_logits": seg_logits})
            ouputs.update({"aug_seg_logits": aug_seg_logits})
        else:
            seg_logits = self.seg_logits(last_layer)
            ouputs.update({"seg_logits": seg_logits})
        return ouputs

def unet_2D(method, cfg):
    modal_dict = {"Base": UNetResNet}
    return modal_dict[method](n_classes=cfg.DATA.SEG_CLASSES, 
                              using_resnet50=cfg.MODEL.USING_RESNET50,
                              using_resnet34=cfg.MODEL.USING_RESNET34,
                              using_resnet18=cfg.MODEL.USING_RESNET18,
                              using_pretrained_model=cfg.MODEL.USING_PRETRAINED_MODEL,
                              )

if __name__ == "__main__":
    model = UNetResNet(using_resnet50=False, 
                       using_resnet34=False,
                       using_resnet18=True,
                       using_pretrained_model=False, 
                       )
    inputs = torch.ones((2, 3, 512, 512), dtype=torch.float32)
    preds = model(inputs)
    print(1)
