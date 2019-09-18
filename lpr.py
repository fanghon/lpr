# Usage example:  python3 lpr.py --video=run.mp4
#                 python3 lpr.py --image=bird.jpg
# coding=utf-8

from cv2 import dnn
import cv2
import argparse
import sys
import numpy as np
import os.path
from hyperlpr_py3 import pipline as pp
import time
from PIL import Image, ImageDraw, ImageFont


# Initialize the parameters
plateTypeName = ["蓝", "黄", "绿", "白", "黑 "]

fontC = ImageFont.truetype("Font/platech.ttf", 38, 0)  # 加载中文字体，38表示字体大小，0表示unicode编码

inWidth = 480  # 480  # from ssd.prototxt ，540，960,720,640，768
inHeight = 640  # 640 ，720，1280，960，480，1024
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843    # 1/127.5
meanVal = 127.5

classNames = ('background',
              'plate')

net = dnn.readNetFromCaffe("./model/MobileNetSSD_test.prototxt","./model/lpr.caffemodel")  # 读入模型文件
net.setPreferableBackend(dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(dnn.DNN_TARGET_OPENCL)   # 启用GPU OPENCL 加速 ，默认FP32
# net.setPreferableTarget(dnn.DNN_TARGET_OPENCL_FP16)   # only for intel xianka test faster speed

parser = argparse.ArgumentParser(description='Object Detection using ssd in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()




# 画车牌定位框及识别出来的车牌字符，返回标记过的图片
def drawPred(frame, label, left, top, right, bottom):
    # 画车牌定位边框.
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 2)

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.text((left + 1, top - 28), label, (0, 0, 255), font=fontC)    # 车牌框上方红色汉字
    imagex = np.array(img)
    return imagex

# 检测结果后处理，根据置信度等，确定最合适的车牌识别结果
def PostProcess(res_set):
    i = 0
    for  plate, plate_color, confidence in res_set[:]:  # 在复制的车牌列表中遍历
        if confidence < 0.8:
            res_set.remove([plate, plate_color, confidence])
            continue
        if len(plate) != 7:
            res_set.remove([plate, plate_color, confidence])
            continue


    return res_set
# 从输入图片中定位车牌，识别车牌字符，返回标记过车牌的图片和识别结果
def detect(frame):

    frame_resized = cv2.resize(frame, (inWidth, inHeight)); # 将原图缩放到指定高宽
    #cv2.imshow("test", frame_resized);
    #cv2.waitKey(0);

    heightFactor = frame.shape[0] / inHeight;  # 计算高度缩放比例
    widthFactor = frame.shape[1] / inWidth;

    t0 = time.time()
    blob = dnn.blobFromImage(frame_resized, inScaleFactor, (inWidth, inHeight), meanVal) # 读入图片
    net.setInput(blob)
    detections = net.forward()   # 定位车牌
    # print("车牌定位时间:", time.time() - t0)

    cols = frame_resized.shape[1]  # 缩放后图片宽度
    rows = frame_resized.shape[0]

    res_set = []
    # 循环遍历定位到的车牌
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:   # 车牌定位置信度大于指定值
            class_id = int(detections[0, 0, i, 1])

            xLeftBottom = int(detections[0, 0, i, 3] * cols) # 被实际检测图中车牌框左上点横坐标
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop = int(detections[0, 0, i, 5] * cols)  # 被实际检测图中车牌框右下点横坐标
            yRightTop = int(detections[0, 0, i, 6] * rows)

            xLeftBottom_ = int(widthFactor * xLeftBottom); # 原图中车牌框左上点横坐标
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

           # cv2.rectangle(frame, (xLeftBottom_-2, yLeftBottom_-2), (xRightTop_+2, yRightTop_+2),(0, 0,255))    #车牌位置绘制红色边框

            image_sub = frame[yLeftBottom_:yRightTop_,xLeftBottom_:xRightTop_] # 截取原图车牌定位区域

            # 必须调整车牌到统一大小
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
            t0 = time.time()
            e2e_plate, e2e_confidence = pp.e2e.recognizeOne(image_rgb)
          #  print("e2e:", e2e_plate, e2e_confidence, plate_color)   #车牌字符判断
          #  print("车牌字符识别时间：",time.time()-t0)

            frame  = drawPred(frame, e2e_plate, xLeftBottom_, yLeftBottom_, xRightTop_, yRightTop_)

            res_set.append([e2e_plate,
                            plate_color,
                            e2e_confidence])
    return frame , res_set

# 将图片据图片中心旋转指定角度
def rotate(image, degree):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    # 将图像旋转180度
    M = cv2.getRotationMatrix2D(center, degree, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# Process inputs
winName = 'Deep learning object detection in OpenCV'
cv2.namedWindow(winName, cv2.WINDOW_NORMAL)

outputFile = "ssd_out_py.avi" # 定义输出文件名
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv2.VideoCapture(args.image)
    outputFile = args.image[:-4] + '_ssd_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv2.VideoCapture(args.video)
    outputFile = args.video[:-4] + '_ssd_out_py.avi'
else:
    # 摄像头输入
    cap = cv2.VideoCapture(0)

# print("w:",cap.get(cv2.CAP_PROP_FRAME_WIDTH),"h:",cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cv2.waitKey(1) < 0:

    # get frame from the video
    hasFrame, frame = cap.read()

    # Stop the program if reached end of video
    if not hasFrame:
        print("处理结束!")
        print("输出文件存储在： ", outputFile)
        cv2.waitKey(3000)
        break

    # 图片旋转90度,只针对test视频文件
    if  args.video.find("test")!=-1:
        frame = rotate(frame, -90)

    t0 = time.time();
    frame,res = detect(frame)

    # 图片左上方输出车牌检测识别时间
    label = ' %.0f ms' % ((time.time()-t0)*1000)
    cv2.putText(frame, label, (int(frame.shape[1]/4), 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255))
    # 车牌后处理,过滤掉不符合要求的车牌
    res = PostProcess(res)
    # 输出车牌检测信息
    info = ""
    for plate, plate_color, confidence in res:
        info = plate + plate_color+" "
    if len(info) > 1:
        print(info, label)



    # Write the frame with the detection boxes
    if (args.image):
        cv2.imwrite(outputFile, frame.astype(np.uint8))
    else:
        vid_writer.write(frame.astype(np.uint8))

    cv2.imshow(winName, frame)

cap.release()
vid_writer.release()
cv2.destroyAllWindows()