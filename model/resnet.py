import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    # residual function
    def __init__(self, in_c, out_c, kernel_size):  # stride is fixed to 1
        super(BasicBlock, self).__init__()
        assert kernel_size % 2 == 1, "kernel_size is not single"
        # set padding to adjust the output size to match the dimension
        self.n_pad = kernel_size // 2  # n_pad = ((stride-1)*input_size + kernel_size - stride) // 2
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=1, padding=self.n_pad, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=1, padding=self.n_pad, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=1, padding=self.n_pad, bias=False),
                nn.BatchNorm2d(out_c)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual_function(x)
        shortcut = self.shortcut(x)
        return self.relu(res + shortcut)


class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        input_channels = args.resnet_input_channels or [1, 4, 16, 32, 64, 128, 128]
        output_sizes = args.resnet_output_sizes or [128, 64, 8]
        kernel_sizes = args.resnet_kernel_sizes or [7, 5, 3]

        assert len(output_sizes) == len(kernel_sizes) and len(input_channels) == 2 * len(kernel_sizes) + 1

        self.res_cov_function = nn.Sequential()

        for i, (o, k) in enumerate(zip(output_sizes, kernel_sizes)):
            self.res_cov_function.append(
                BasicBlock(in_c=input_channels[2 * i], out_c=input_channels[2 * i + 1], kernel_size=k))
            self.res_cov_function.append(
                nn.Conv2d(input_channels[2 * i + 1], input_channels[2 * i + 2], kernel_size=k, padding=1))
            self.res_cov_function.append(nn.ReLU())
            self.res_cov_function.append(nn.AdaptiveMaxPool2d((o, o)))

        # Fully connected layer
        self.fc = nn.Linear(input_channels[-1] * output_sizes[-1] ** 2, args.class_num)

    def forward(self, x):
        x = self.res_cov_function(x)
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x)  # [batch_size, num_class=15]

# if __name__ == '__main__':
#     from torchviz import make_dot
#     x = torch.randn(1, 1, 224, 224).requires_grad_(True)  # 定义一个网络的输入值
#     model = ResNet()  # 获取网络的预测值
#     y = model(x)
#     MyConvNetVis = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
#     MyConvNetVis.format = "png"
#     # 指定文件生成的文件夹
#     MyConvNetVis.directory = "visual_model"
#     # 生成文件
#     MyConvNetVis.view()
