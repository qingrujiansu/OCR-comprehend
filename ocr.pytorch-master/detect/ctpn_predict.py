import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np

import torch
import torch.nn.functional as F
from detect.ctpn_model import CTPN_Model
from detect.ctpn_utils import gen_anchor, bbox_transfor_inv, clip_box, filter_bbox,nms, TextProposalConnectorOriented
from detect.ctpn_utils import resize
from detect import config

prob_thresh = 0.5
height = 720
gpu = True

# 判断能否使用gpu
if not torch.cuda.is_available():
    gpu = False
device = torch.device('cuda:0' if gpu else 'cpu') # 检测设备如果可以就使用cuda：0 不可以就使用cpu 返回分配tensor的张量
weights = os.path.join(config.checkpoints_dir, 'CTPN.pth') # 权重使用CTPN.pth的预训练模型
model = CTPN_Model() # 得到CTPN模型
model.load_state_dict(torch.load(weights, map_location=device)['model_state_dict']) # 将预训练的参数权重加载到新模型之中
model.to(device) # 模型在GPU上训练
# 在模型中，我们通常会加上Dropout层和batch normalization层，在模型预测阶段，我们需要将这些层设置到预测模式，model.eval()就是帮我们一键搞定的
model.eval()

# 显示图像
def dis(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_det_boxes(image,display = True, expand = True):
    image = resize(image, height=height) # 得到指定的大小 高默认为720
    image_r = image.copy() # 复制图像
    image_c = image.copy() # 复制图像
    h, w = image.shape[:2] # 得到三维数组的前两位
    image = image.astype(np.float32) - config.IMAGE_MEAN # config.IMAGE_MEAN = [123.68, 116.779, 103.939] 尺寸相减
    # image.transpose(2, 0, 1)实现数据的布局改变（299,300,3）=>（3,299,300）unsqueeze(0)=>增加一维度，变成[[299,300,3]]
    # 转化为浮点数之后转化为tensor
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()

    with torch.no_grad():
        image = image.to(device) # 将tensor变量copy一份到device所指定的设备上面，之后的运算都在GPU或者CPU上进行
        cls, regr = model(image) # 通过ctpn模型得到结果
        cls_prob = F.softmax(cls, dim=-1).cpu().numpy() # 使用cpu来找出最大值并且转化为numpy数组
        regr = regr.cpu().numpy() # 使用cpu转化为numpy数组
        anchor = gen_anchor((int(h / 16), int(w / 16)), 16) # 返回特定的尺寸作为锚点
        bbox = bbox_transfor_inv(anchor, regr) # 得到特定的bbox格式
        bbox = clip_box(bbox, [h, w]) #  比较两个数组并返回一个包含元素最小值的新数组
        # print(bbox.shape)

        fg = np.where(cls_prob[0, :, 1] > prob_thresh)[0] # prob_thresh = 0.5 返回符合要求的数组
        # print(np.max(cls_prob[0, :, 1]))
        select_anchor = bbox[fg, :] # 截取从fg 到最后一位 获得锚点
        select_score = cls_prob[0, fg, 1] # 获得分数
        select_anchor = select_anchor.astype(np.int32) # 将锚点转化为int32
        # print(select_anchor.shape)
        keep_index = filter_bbox(select_anchor, 16) # 返回bbox格式大于16的值

        # nms
        select_anchor = select_anchor[keep_index] # 得到锚点值
        select_score = select_score[keep_index] # 得到锚点分数
        select_score = np.reshape(select_score, (select_score.shape[0], 1)) # 将锚点分数重塑
        nmsbox = np.hstack((select_anchor, select_score)) # 将数组平铺
        keep = nms(nmsbox, 0.3) # 返回满足 <= 0.3的值
        # print(keep)
        select_anchor = select_anchor[keep]
        select_score = select_score[keep]

        # text line-
        textConn = TextProposalConnectorOriented() # 得到文本建议标记链接
        text = textConn.get_text_lines(select_anchor, select_score, [h, w])

        # expand text
        if expand:
            for idx in range(len(text)):
                text[idx][0] = max(text[idx][0] - 10, 0)
                text[idx][2] = min(text[idx][2] + 10, w - 1)
                text[idx][4] = max(text[idx][4] - 10, 0)
                text[idx][6] = min(text[idx][6] + 10, w - 1)


        # print(text)
        if display:
            blank = np.zeros(image_c.shape,dtype=np.uint8)
            for box in select_anchor:
                pt1 = (box[0], box[1])
                pt2 = (box[2], box[3])
                blank = cv2.rectangle(blank, pt1, pt2, (50, 0, 0), -1) # 画方框
            image_c = image_c+blank
            image_c[image_c>255] = 255
            for i in text:
                s = str(round(i[-1] * 100, 2)) + '%'
                i = [int(j) for j in i]
                cv2.line(image_c, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 2)
                cv2.line(image_c, (i[0], i[1]), (i[4], i[5]), (0, 0, 255), 2)
                cv2.line(image_c, (i[6], i[7]), (i[2], i[3]), (0, 0, 255), 2)
                cv2.line(image_c, (i[4], i[5]), (i[6], i[7]), (0, 0, 255), 2)
                cv2.putText(image_c, s, (i[0]+13, i[1]+13),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255,0,0),
                            2,
                            cv2.LINE_AA)
            # dis(image_c)
        # print(text)
        return text,image_c,image_r

if __name__ == '__main__':
    img_path = 'images/t1.png'
    image = cv2.imread(img_path) # 读入一副图片默认是一副彩色图片 返回三维数组形式
    text,image = get_det_boxes(image) # 返回文本 画出图像的结果
    dis(image) # 显示图像