"""
Author: fanghong
edited: 2019.5.12
"""
#coding=utf-8
from cv2 import dnn
import cv2
from hyperlpr_py3 import pipline as pp
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

Sheng = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]

plateSheng = {"京":"JING","津":"JINA","沪":"HU","渝":"YUA","蒙":"MENG","新":"XIN","藏":"ZANG","宁":"NING",
                 "桂":"GUIA","黑":"HEI","吉":"JIB","辽":"LIAO","晋":"JINB","冀":"JIA","青":"QING","鲁":"LU",
                 "豫":"YUB","苏":"SU","皖":"WAN","浙":"ZHE","闽":"MIN","赣":"GANA","湘":"XIANG","鄂":"E",
                 "粤":"YUE","琼":"QIONG","甘":"GANB","陕":"SHAN","贵":"GUIB","云":"YUN","川":"CHUAN"}

plateTypeName = ["蓝", "黄", "绿", "白", "黑 "]
fontC = ImageFont.truetype("Font/platech.ttf", 38, 0)  # 加载中文字体，38表示字体大小，0表示unicode编码
inWidth = 480  # 480  # from ssd.prototxt ，540，960,720,640,768,设置图片宽度
inHeight = 640  # 640 ，720，1280，960，480，1024
WHRatio = inWidth / float(inHeight)   # 计算宽高比
inScaleFactor = 0.007843    # 1/127.5
meanVal = 127.5

classNames = ('background',
              'plate')
net = dnn.readNetFromCaffe("model/MobileNetSSD_test.prototxt","model/lpr.caffemodel")  # 读入模型文件
net.setPreferableBackend(dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(dnn.DNN_TARGET_CPU)    # 使用cpu
# net.setPreferableTarget(dnn.DNN_TARGET_OPENCL)   # 启用GPU OPENCL 加速 ，默认FP32
# net.setPreferableTarget(dnn.DNN_TARGET_OPENCL_FP16)   # only for intel xianka test faster speed


# 画车牌定位框及识别出来的车牌字符，返回标记过的图片
def drawPred(frame, label, left, top, right, bottom):
    # 画车牌定位边框.左上点，右下点,红色，边框粗细：2
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    # 画车牌字符
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.text((left + 1, top - 38), label, (0, 0, 255), font=fontC)    # 车牌框上方红色汉字
    imagex = np.array(img)
    return imagex

# 判断车牌字符是否有效
def isValidPlate(plate,confidence):
    # 置信度大于0.8，长度等于7或8（绿牌） ， 车牌第一个字符应是省名
    if confidence > 0.8 and (len(plate) == 7 or len(plate) == 8) and plate[0]  in Sheng:
        return True
    return False


# 对输入图片进行检测，返回结果：绘制了车牌定位框的图，检测结果（车牌，车牌颜色，车牌字符置信度等）
def detect(frame):

    frame_resized = cv2.resize(frame, (inWidth, inHeight)); # 将原图缩放到指定高宽，并显示
   # cv2.imshow("test", frame_resized)
   # cv2.waitKey(0)

    heightFactor = frame.shape[0] / inHeight  # 计算高度缩放比例
    widthFactor = frame.shape[1] / inWidth   # 计算宽度缩放比例

    # t0 = time.time()
    # 读取图片，并按指定参数缩放
    blob = dnn.blobFromImage(frame_resized, inScaleFactor, (inWidth, inHeight), meanVal)
    net.setInput(blob)  # 设置好图片输出
    detections = net.forward()   # ssd神经网处理图片，返回结果
    # print("车牌定位时间:", time.time() - t0)

    cols = frame_resized.shape[1]  # 宽度，列
    rows = frame_resized.shape[0]  # 高度，行

    res_set = []  # 检测结果
    framedrawed = frame
    # 循环遍历处理定位到的车牌
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # 提取出车牌定位置信度
        if confidence > 0.2:
            # class_id = int(detections[0, 0, i, 1])

            xLeftBottom = int(detections[0, 0, i, 3] * cols) # 被实际检测图（缩放过的）中车牌框左上点横坐标
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop = int(detections[0, 0, i, 5] * cols)  # 被实际检测图中车牌框右下点横坐标
            yRightTop = int(detections[0, 0, i, 6] * rows)

            xLeftBottom_ = int(widthFactor * xLeftBottom); # 原始图中车牌框左上点横坐标
            yLeftBottom_ = int(heightFactor * yLeftBottom);
            xRightTop_ = int(widthFactor * xRightTop);
            yRightTop_ = int(heightFactor * yRightTop);
            # print("y1:",yLeftBottom_, "y2:",yRightTop_, "x1:",xLeftBottom_, "x2:", xRightTop_)  # 输出车牌在原图中位置信息
            # 适当扩大车牌定位框
            h = yRightTop_ - yLeftBottom_
            w = xRightTop_ - xLeftBottom_
            yLeftBottom_ -= int(h * 0.5)
            yRightTop_ += int(h * 0.5)
            xLeftBottom_ -= int(w * 0.14)
            xRightTop_ += int(w * 0.14)


            image_sub = frame[yLeftBottom_:yRightTop_,xLeftBottom_:xRightTop_] # 截取原图车牌定位区域

            # 调整车牌到统一大小
            plate = image_sub
            # print(plate.shape[0],plate.shape[1])
            if plate.shape[0] > 36:
                plate = cv2.resize(image_sub, (136, 36 * 2))
            else:
                plate = cv2.resize(image_sub, (136, 36 ))
          #  cv2.imshow("test", plate)
          #  cv2.waitKey(0)
            # 判断车牌颜色

            plate_type = pp.td.SimplePredict(plate)
            plate_color = plateTypeName[plate_type]

            if (plate_type > 0) and (plate_type < 5):
                plate = cv2.bitwise_not(plate)


            # 精定位，倾斜校正
            image_rgb = pp.fm.findContoursAndDrawBoundingBox(plate)
           # cv2.imshow("test", image_rgb);
           # cv2.waitKey(0)
            # 车牌左右边界修正
            image_rgb = pp.fv.finemappingVertical(image_rgb)
           # cv2.imshow("test", image_rgb);
           # cv2.waitKey(0)
            # 车牌字符识别
            # t0 = time.time()
            e2e_plate, e2e_confidence = pp.e2e.recognizeOne(image_rgb)
            # print("e2e:", e2e_plate, e2e_confidence, plate_color)   #车牌字符判断
            # print("车牌字符识别时间：",time.time()-t0)
            if isValidPlate(e2e_plate,e2e_confidence):  # 判断是否是有效车牌
                # 在原图中绘制定位框及车牌信息，传入定位框左上点和右下点xy坐标
                framedrawed = drawPred(framedrawed, e2e_plate, xLeftBottom_, yLeftBottom_, xRightTop_, yRightTop_)

                res_set.append([e2e_plate,   # 结果车牌号
                            plate_color,     # 车牌颜色
                            e2e_confidence,  # 车牌字符置信度
                            (xLeftBottom_, yLeftBottom_)])   # 车牌定位框左上点坐标


    return framedrawed, res_set  # 返回绘制的图片，检测结果




test_dir = "./test-imgs"  # 图片读入路径
fw = open("./test-results/No14007mresults.txt", 'w+')  # 以覆盖写方式打开文件，如果不存在，则新建一个

# cv2.namedWindow("test",cv2.WINDOW_NORMAL)   #cv2.WINDOW_AUTOSIZE
#  循环遍历文件夹下所有的图片文件
for f in os.listdir(test_dir):
    try:
        if f.endswith(".jpg") or f.endswith("JPG") or f.endswith("png"):
            # print("---------"+f+"----------------");
            cpp = test_dir + "/" + f    # 生成完整路径
            image = cv2.imdecode(np.fromfile(cpp, dtype=np.uint8), -1);  # 读入图片文件，支持中文名

            t0 = time.time()
            framedrawed,res = detect(image)  # 识别图片，返回的是绘制的原图片
            tlabel = '%.0f ms' % ((time.time()-t0)*1000)
           # print("识别总耗时：",tlabel)

            # 输出车牌检测信息
            info = f+"\n"   # 输出信息,文件名+换行符
            # 循环遍历检测结果，将车牌省名替换为相应拼音
            for r in res:
                py = plateSheng[r[0][0]]  # 获取结果中车牌的第一个字符省名，获取省名对应的拼音
                plate = r[0].replace(r[0][0],py)  # 将省名替换为拼音
                info = info + plate + "\n"   # 拼接结果字符串

            fw.write(info)   # 写入检测信息到结果文本文件
           # cv2.imwrite("./test-results/"+f, framedrawed.astype(np.uint8))  # 保存图片
            print(info[:-1])   # 屏幕输出结果
            print(tlabel)   # 输出处理时间

           # img2 = cv2.resize(framedrawed, (0, 0), fx=0.25, fy=0.25);
           # cv2.imshow("test",img2)
           # cv2.waitKey(0)
    except Exception as e:
        print(e)  # 输出异常信息，调试用，发布时应注释掉
        continue            # 出现异常则继续循环读取

fw.close()
cv2.destroyAllWindows()