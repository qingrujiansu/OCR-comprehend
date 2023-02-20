import torch
from torch import nn
import torch.nn.functional as F

# CRNN采取的架构是CNN+RNN+CTC，
# cnn提取图像像素特征，
# rnn提取图像时序特征，
# ctc归纳字符间的连接特性。
class CRNN(nn.Module):
    # 定义输入图片的高，输入通道，？n_class,填充大小
    def __init__(self, img_height, input_channel, n_class, hidden_size):
        super().__init__()
        # 图片的高必须是16的倍数
        if img_height % 16 != 0:
            raise ValueError('img_height has to be a multiple of 16')
        # 卷积核的尺寸
        kernel_size = [3, 3, 3, 3, 3, 3, 2]
        # 填充的尺寸
        padding_size = [1, 1, 1, 1, 1, 1, 0]
        # 训练时候的每一步步长
        stride = [1, 1, 1, 1, 1, 1, 1]
        # 通道数
        channel = [64, 128, 256, 256, 512, 512, 512]
        # 卷积函数 batchNormalization = false默认不进行归一化处理
        def conv_relu(i, batchNormalization=False):
            # 如果 i == 0 则直接将input_channel赋值给in_channels，否则将channel[i - 1]赋值给它
            in_channels = input_channel if i == 0 else channel[i - 1]
            out_channels = channel[i]
            # 添加卷积操作
            cnn.add_module(f'conv{i}',
                           nn.Conv2d(in_channels, out_channels,
                                     kernel_size[i],
                                     stride[i],
                                     padding_size[i]))

            if batchNormalization:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(out_channels))
            # 添加激活函数relu ，直接替换原来的数据源
            cnn.add_module(f'relu{i}', nn.ReLU(True))

        # x: 1 x 32 x 320
        # 定义一个执行序列
        cnn = nn.Sequential()
        # 卷积通道数设为64
        conv_relu(0)
        # 进行2,2的池化操作
        cnn.add_module('pooling0', nn.MaxPool2d(2, 2))  # 64x16x160
        # 卷积通道数设为128？
        conv_relu(1)
        # 进行2,2的池化操作
        cnn.add_module('pooling1', nn.MaxPool2d(2, 2))  # 128x8x80
        # 卷积通道数设为128 进行归一化处理
        conv_relu(2, True)
        # 卷积通道数设为256
        conv_relu(3)
        # 添加池化操作
        cnn.add_module('pooling2',
                       nn.MaxPool2d(kernel_size=(2, 2),
                                    stride=(2, 1),
                                    padding=(0, 1)))  # 256x4x81
        # 卷积通道数设为256 进行归一化处理
        conv_relu(4, True)
        # 卷积通道数设为512
        conv_relu(5)
        # 添加池化操作
        cnn.add_module('pooling3',
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x82
        # 卷积通道数设为512 进行归一化处理
        conv_relu(6, True)  # 512x1x81
        # 将序列化出来的cnn赋值给类本身
        self.cnn = cnn
        # 定义类本身的rnn模型 运行两层rnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, n_class)
        )

    def forward(self, x):
        # cnn_feature是tensor类型
        cnn_feature = self.cnn(x)

        # 1 x 512 x 1 x 81
        h = cnn_feature.size()[2]
        if h != 1:
            raise ValueError("the height of cnn_feature must be 1")
        # 删除cnn_feature所有维度为2的维数
        cnn_feature = cnn_feature.squeeze(2)

        # 81: 序列长度 1: batch size, 512: 每个特征的维度
        # 将维数互换
        cnn_feature = cnn_feature.permute(2, 0, 1)

        output = self.rnn(cnn_feature)
        # [81, 1, 26]
        x = F.log_softmax(x, dim=2)
        return output

# 双向传播长短期神经网络Rnn
class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_feature):
        super().__init__()
        # 输入尺寸 隐藏的特征数量 双向 LSTM
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True)
        # 线性化 输入尺寸 输出尺寸
        self.embedding = nn.Linear(hidden_size * 2, out_feature)

    def forward(self, x):
        # x: [81, 1, 512] → [sequence_length, batch_size, input_size]
        recurrent, _ = self.rnn(x)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output

def weights_init(m):
    # get class name
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_crnn(config):

    model = CRNN(config.MODEL.IMAGE_SIZE.H, 1, config.MODEL.NUM_CLASSES + 1, config.MODEL.NUM_HIDDEN)
    model.apply(weights_init)

    return model

if __name__ == '__main__':
    img = torch.randn((1, 1, 32, 320))

    crnn = CRNN(32, 1, 26, 256)

    res = crnn(img)

    print(res.shape)
