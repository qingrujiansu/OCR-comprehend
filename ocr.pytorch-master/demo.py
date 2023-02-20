import os
from ocr import ocr
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob


def single_pic_proc(image_file):
    # 打开图片，转化为npmpy数组，并且指定为3通道的rgb模式，因为rgba其中a透明通道用不上
    image = np.array(Image.open(image_file).convert('RGB'))
    result, image_framed = ocr(image) # 调用ocr返回结果
    return result,image_framed # 返回结果


# 入口
if __name__ == '__main__':
    image_files = glob('./test_images/*.*')# 读取相应的文件转化为列表形式
    result_dir = './test_result'# 定义结果目录
    if os.path.exists(result_dir): # 如果系统文件存在结果输出目录
        shutil.rmtree(result_dir) # 递归删除文件夹下的所有子文件夹和子文件
    os.mkdir(result_dir)# 创建相应的目录

    for image_file in sorted(image_files):# 循环遍历测试图片列表，并且先是升序排序
        t = time.time() # 定义开始时间
        result, image_framed = single_pic_proc(image_file) # 得到图片识别后的结果
        output_file = os.path.join(result_dir, image_file.split('/')[-1]) # 生成输出图片文件地址
        txt_file = os.path.join(result_dir, image_file.split('/')[-1].split('.')[0]+'.txt') # 生成输出结果文件地址
        print(txt_file)# 打印结果文件地址
        txt_f = open(txt_file, 'w') # 打开结果文件地址为写入模式
        Image.fromarray(image_framed).save(output_file) # 将的到的image_framed数组结果转换为图片存储在输出文件里面
        print("Mission complete, it took {:.3f}s".format(time.time() - t)) # 打印任务完成并且计算时间
        print("\nRecognition Result:\n") # 打印识别出来的结果
        for key in result: # 循环遍历数组
            print(result[key][1]) # 打印结果
            txt_f.write(result[key][1]+'\n') # 将结果写入结果文件
        txt_f.close() # 关闭文件写入流