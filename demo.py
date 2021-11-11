from detection import Detction
import cv2

if __name__ == '__main__':
    det = Detction(3,'../logs/best.pth')
    image = cv2.imread('/home/zhanggong/disk/Extern/workspace/yolov4_tiny_ws/yolov4_tiny/VOCdevkit/VOC2007/JPEGImages/011.jpg')

    outputs = det.detect(image)[0]
    for out in outputs:
        y1 =int(out[0])
        x1 = int(out[1])
        y2 =int(out[2])
        x2 =int(out[3])
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,26))
    cv2.imshow('image',image)
    cv2.waitKey()
    cv2.destroyAllWindows()
