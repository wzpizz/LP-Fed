import torch
import torch.nn as nn
import torch.nn.functional as F


class UpLayer(nn.Module):
    def __init__(self, in_channels=1, mid_channels=2048, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        super(UpLayer, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels,
                          kernel_size=1, bias=False),
                BatchNorm(in_channels)
            )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)
        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))

        return sim_map


class Sim_map:
    def __init__(self, train_loader, device):
        self.device = device
        self.train_loader = train_loader

    def compute_similarity(self, old_model, old_classifier, new_model):
        old_model = old_model.to(self.device)
        old_classifier = old_classifier.to(self.device)
        new_model = new_model.to(self.device)

        similarity_list = []

        for data in self.train_loader:
            inputs, _ = data
            inputs = inputs.to(self.device)

            with torch.no_grad():
                old_out = old_classifier(old_model(inputs))
                new_out = new_model(inputs)
            uplayer = UpLayer(in_channels=1)  # 创建 UpLayer 实例
            uplayer = uplayer.to(self.device)
            old_out = old_out.unsqueeze(0)
            old_out = old_out.unsqueeze(0)
            new_out = new_out.unsqueeze(0)
            new_out = new_out.unsqueeze(0)
            sim_map = uplayer(old_out, new_out)
            average_sim_map=torch.sum(sim_map)/torch.numel(sim_map)





        return average_sim_map
