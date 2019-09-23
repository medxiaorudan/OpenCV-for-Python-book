# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 01:41:17 2019

@author: xiaorudan
"""

##############################################################################
##########################chr 1 OpenCV入门#####################################
###############################################################################
#1.2.1 读取图像
import cv2
lena=cv2.imread("lena.png")
print(lena)
#1.2.2 显示图像
import cv2
lena=cv2.imread("lena.jpg",-1)
cv2.imshow("before",lena)
cv2.waitKey()
cv2.destroyWindows()
#在一个窗口内显示图像，并针对按下的不同键做出不同的反应
import cv2
lena=cv2.imread("lena.jpg",-1)
cv2.imshow("demo",lena)
key=cv2.waitKey() 
if key==ord("A"):
    cv2.imshow("pressA", lena)
elif key==ord("B"):
    cv2.imshow("pressB",lena)
cv2.destroyAllWindows()
#在一个窗口内显示图像，用函数cv2.waitKey()实现程序暂停，在按下键盘的按键后程序继续运行
import cv2
lena=cv2.imread("lena.jpg",-1)
cv2.imshow("demo",lena)
key=cv2.waitKey()
if key!=-1:
    print("触发了按键")
#1.2.3 保存图像
import cv2
lena=cv2.imread("lena.jpg",-1)
r=cv2.imwrite("lena.bmp",lena)   


##############################################################################
############################chr 2 图像处理基础#################################
###############################################################################
#2.2 像素处理
#2.2.1 二值图像及灰度图像
#使用Numpy库来生成一个元素值都是0的二维数组，用来模拟一副黑色图像，并对其进行修改访问
import cv2
import numpy as np
img=np.zeros((8,8),dtype=np.uint8)
print("img=\n",img)
print("读取像素点[0，3]=",img[0,3])
img[0,3]=255
print("修改后img=\n",img)
print("读取修改后像素点[0，3]=",img[0,3])
cv2.imshow("two",img)
cv2.waitKey()
cv2.destroyAllWindows()
#读取一个灰度图像，对其像素进行修改访问
import cv2
lena=cv2.imread("lena.jpg",0)
cv2.imshow("before",lena)
for i in range(10,100):
    for j in range(80,100):
        lena[i,j]=255
cv2.imshow("after",lena)
cv2.waitKey()
cv2.destroyAllWindows()

#2.2.2 彩色图像
#使用numpy生成三维数组，用来观察三个通道的变化情况
import cv2
import numpy as np
#####------------蓝色通道--------------#######
blue=np.zeros((300,300,3),dtype=np.uint8)
blue[:,:,0]=255
print("blue=\n",blue)
cv2.imshow("blue",blue)
#####-------------绿色通道-------------####
green=np.zeros((300,300,3),dtype=np.uint8)
green[:,:,1]=255
print("green=\n",green)
cv2.imshow("green",green)
#####--------------红色通道-----------#########
red=np.zeros((300,300,3),dtype=np.uint8)
red[:,:,2]=255
print("red=\n",red)
cv2.imshow("red",red)
####---------------释放窗口--------------####
cv2.waitKey()
cv2.destroyAllWindows()

#使用numpy生成三维数组，用来观察三个通道的变化情况
import cv2
import numpy as np
img=np.zeros((300,300,3),dtype=np.uint8)
img[:,0:100,0]=255
img[:,100:200,1]=255
img[:,200:300,2]=255
print("img\n",img)
cv2.imshow("img",img)
cv2.waitKey()
cv2.destroyAllWindows()


#使用numpy生成三维数组，用来模拟一幅BGR模式的彩色图像，并对其像素进行修改访问
import numpy as np
img=np.zeros((2,4,3),dtype=np.uint8)
print("img=\n",img)
print("读取像素点img[0,3]=",img[0,3])
print("读取像素点img[1,2,2]=",img[1,2,2])
img[0,3]=255
img[0,0]=[66,77,88]
img[1,1,1]=3
img[1,2,2]=4
img[0,2,0]=5
print("修改后的img=\n",img)
print("修改后的像素点img[1,2,2]=",img[1,2,2])

#读取一幅彩色图像，并对其像素进行修改访问
import cv2
import numpy as np
img=cv2.imread("lena.png",-1)
cv2.imshow("before",img)
print("访问img[0,0]=",img[0,0])
print("访问img[0,0,0]=",img[0,0,0])
print("访问img[0,0,1]=",img[0,0,1])
print("访问img[0,0,2]=",img[0,0,2])
print("访问img[50,0]=",img[50,0])
print("访问img[100,0]=",img[100,0])
###区域1
for i in range(0,50):
    for j in range(0,100):
        for k in range(0,3):
            img[i,j,k]=255   ##白色
###区域2
for i in range(50,100):
    for j in range(0,100):
        for k in range(0,3):
            img[i,j,k]=128   ###灰色
###区域3
for i in range(100,150):
    for j in range(0,100):
        for k in range(0,3):
            img[i,j,k]=0    ####黑色
cv2.imshow("after",img)
print("修改后img[0,0]=",img[0,0])
print("修改后img[0,0,0]=",img[0,0,0])
print("修改后img[0,0,1]=",img[0,0,1])
print("修改后img[0,0,2]=",img[0,0,2])
print("修改后img[50,0]=",img[50,0])
print("修改后img[100,0]=",img[100,0])
cv2.waitKey()
cv2.destroyAllWindows()

#2.3 使用numpy.array访问像素
###########灰度图像和彩色图像########
#2.3.1 二值图像及灰度图像
#使用Numpy生成一个二维随机数组，用来模拟一幅灰度图像，并对其像素进行修改访问
import cv2
import numpy as np
img=np.random.randint(10,99,size=[5,5],dtype=np.uint8)
print("img=\n",img)
print("读取像素点img.item(3,2)=",img.item(3,2))
img.itemset((3,2),255)
print("修改后img=\n",img)
print("修改后像素点img.item(3,2)=",img.item(3,2))
#生成一个灰度图像，让其中的像素值均为随机数
import cv2
import numpy as np
img=np.random.randint(0,256,size=[256,256],dtype=np.uint8)
cv2.imshow("demo",img)
cv2.waitKey()
cv2.destroyAllWindows()

#读取一幅灰度图像，并对其像素进行修改访问
import cv2
img=cv2.imread("lena.png",0)
#测试修改、读取单个像素值
print("读取像素点img.item(3,2)=",img.item(3,2))
img.itemset((3,2),255)
print("修改后像素点img.item(3,2)=",img.item(3,2))
#测试修改一个区域的像素值
cv2.imshow("before",img)
for i in range(10,100):
    for j in range(80,100):
        img.itemset((i,j),255)
cv2.imshow("after",img)
cv2.waitKey()
cv2.destroyAllWindows()

#2.3.2彩色图像
#使用Numpy生成一个由随机数组成的三维数组，用来模拟一幅RGB彩色空间的彩色图像，并使用函数item()和itemset()来访问和修改它
import numpy as np
img=np.random.randint(10,99,size=[2,4,3],dtype=np.uint8)
print("img=\n",img)
print("读取像素点img[1,2,0]=",img.item(1,2,0))
print("读取像素点img[0,2,1]=",img.item(0,2,1))
print("读取像素点img[1,0,2]=",img.item(1,0,2))
img.itemset((1,2,0),255)
img.itemset((0,2,1),255)
img.itemset((1,0,2),255)
print("修改后img=\n",img)
print("修改后像素点img[1,2,0]=",img.item(1,2,0))
print("修改后像素点img[0,2,1]=",img.item(0,2,1))
print("修改后像素点img[1,0,2]=",img.item(1,0,2))
#生成一幅彩色图像，让其中的像素值均为随机数
import cv2
import numpy as np
img=np.random.randint(0,256,size=[256,256,3],dtype=np.uint8)
cv2.imshow("demo",img)
cv2.waitKey()
cv2.destroyAllWindows()

#2.4 感兴趣区域ROI
#获取图像lena的脸部信息，并将其显示出来
import cv2
import numpy as np
a=cv2.imread("lena.png",cv2.IMREAD_UNCHANGED)
face=a[110:250,130:230]
cv2.imshow("original",a)
cv2.imshow("face",face)
cv2.waitKey()
cv2.destroyAllWindows()
#对脸部进行打码
import cv2
import numpy as np
face=np.random.randint(0,256,size=[140,100,3])
a[110:250,130:230]=face
cv2.imshow("result",a)
cv2.waitKey()
cv2.destroyAllWindows()

##2.5 通道操作
#2.5.1 通道拆分
lena=cv2.imread("lena.png",-1)
cv2.imshow("lena",lena)
b=lena[:,:,0]
g=lena[:,:,1]
r=lena[:,:,2]
cv2.imshow("b",b)
cv2.imshow("g",g)
cv2.imshow("r",r)
lena[:,:,0]=0
cv2.imshow("lenab0",lena)
lena[:,:,1]=0
cv2.imshow("lenab0g0",lena)
cv2.waitKey()
cv2.destroyAllWindows()
#通过函数拆分通道
import cv2
lena=cv2.imread("lena.png",-1)
b,g,r=cv2.split(lena)
cv2.imshow("B",b)
cv2.imshow("G",g)
cv2.imshow("R",r)
cv2.waitKey()
cv2.destroyAllWindows()

#2.5.2 通道合并
import cv2
lena=cv2.imread("lena.png",-1)
b,g,r=cv2.split(lena)
bgr=cv2.merge([b,g,r])
rgb=cv2.merge([r,g,b])
cv2.imshow("lena",lena)
cv2.imshow("bgr",bgr)
cv2.imshow("rgb",rgb)
cv2.waitKey()
cv2.destroyAllWindows()

##2.6 获取图像属性
import cv2
grey=cv2.imread("lena.png",0)
color=cv2.imread("lena.png",-1)
print("图像grey的属性：")
print("grey.shape=",grey.shape)
print("grey.size=",grey.size)
print("grey.dtype=",grey.dtype)
print("图像color的属性：")
print("color.shape=",color.shape)
print("color.size=",color.size)
print("color.dtype=",color.dtype)

##############################################################################
########################chr 3 图像运算#########################################
##############################################################################
#3.1图像加法运算
import cv2
a=cv2.imread("lena.png",0)
b=a
result1=a+b
result2=cv2.add(a,b)
cv2.imshow("original",a)
cv2.imshow("result1",result1)
cv2.imshow("result2",result2)
cv2.waitKey()
cv2.destroyAllWindows()

#3.2图像加权和
import cv2
import numpy as np
img1=np.ones((3,4),dtype=np.uint8)*100##生成一个3*4大小的、元素值都是100的二维数组
img2=np.ones((3,4),dtype=np.uint8)*10
gamma=3
img3=cv2.addWeighted(img1,0.6,img2,5,gamma)  ###计算img1*0.6+img2*5+3
print(img3)

import cv2
import numpy as np
img1=np.random.randint(10,100,size=[3,4],dtype=np.uint8)
img2=np.random.randint(20,200,size=[3,4],dtype=np.uint8)
gamma=3
img3=cv2.addWeighted(img1,0.6,img2,5,gamma)  ###计算img1*0.6+img2*5+3
print(img1)
print(img2)
print(img3)

#3.3 按位逻辑运算
#3.3.1 按位与运算
import cv2
import numpy as np
a=np.random.randint(0,255,[5,5],dtype=np.uint8)
b=np.zeros((5,5),dtype=np.uint8)
b[0:3,0:3]=255
b[4,4]=255
c=cv2.bitwise_and(a,b)
print("a=\n",a)
print("b=\n",b)
print("c=\n",c)

import cv2
import numpy as np
a=cv2.imread("lena.png",0)
b=np.zeros(a.shape,dtype=np.uint8)
b[50:245,45:245]=255
b[50:290,45:130]=255
c=cv2.bitwise_and(a,b)
cv2.imshow("a",a)
cv2.imshow("b",b)
cv2.imshow("c",c)
cv2.waitKey()
cv2.destroyAllWindows()

#3.4 掩模
import cv2
import numpy as np
img1=np.ones((4,4),dtype=np.uint8)*3
img2=np.ones((4,4),dtype=np.uint8)*5
img3=np.ones((4,4),dtype=np.uint8)*66
mask=np.ones((4,4),dtype=np.uint8)
mask[:,0:2]=0
mask[0:2,:]=0
print("img1=\n",img1)
print("img2=\n",img2)
print("mask=\n",mask)
print("初始值img3=\n",img3)
img3=cv2.add(img1,img2,mask=mask)
print("求和后img3=\n",img3)

#3.6 位平面分解
import cv2
import numpy as np
lena=cv2.imread("lena.png",0)
cv2.imshow("lena",lena)
r,c=lena.shape
x=np.zeros((r,c,8),dtype=np.uint8)
for i in range(0,8):
    x[:,:,i]=2**i
r=np.zeros((r,c,8),dtype=np.uint8)
for i in range(0,8):
    r[:,:,i]=cv2.bitwise_and(lena,x[:,:,i])
    mask=r[:,:,i]>0
    r[mask]=255
    cv2.imshow(str(i),r[:,:,i])
cv2.waitKey()
cv2.destroyAllWindows()

#3.7 图像加密和解密
import cv2
import numpy as np
lena=cv2.imread("lena.png",0)
key=np.random.randint(0,256,lena.shape,dtype=np.uint8)
encryption=cv2.bitwise_xor(lena,key)
decryption=cv2.bitwise_xor(encryption,key)
cv2.imshow("lena",lena)
cv2.imshow("key",key)
cv2.imshow("encryption",encryption)
cv2.imshow("decryption",decryption)
cv2.waitKey()
cv2.destroyAllWindows()

##############################################################################
########################chr 4 色彩空间类型转换#################################
##############################################################################
#4.3.1 通过数组观察转换效果
#将RGB转换成灰度图像
import cv2
import numpy as np
img=np.random.randint(0,256,[2,4,3],dtype=np.uint8)
rst=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print("img=\n",img)
print("rst=\n",rst)
print("像素点(1,0)直接计算得到的值=",img[1,0,0]*0.114+img[1,0,1]*0.587+img[1,0,2]*0.299)
print("像素点(1,0)使用公式cv2.cvtColor()转换值=",rst[1,0])

import cv2
import numpy as np
img=np.random.randint(0,256,[2,4],dtype=np.uint8)
rst=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
print("img=\n",img)
print("rst=\n",rst)
#4.3.2 图像处理实例
import cv2
lena=cv2.imread("lena.png",-1)
gray=cv2.cvtColor(lena,cv2.COLOR_BGR2GRAY)
rgb=cv2.cvtColor(lena,cv2.COLOR_BGR2RGB)
#===============打印shape========================###
print("lena.shape=",lena.shape)
print("gray.shape=",gray.shape)
print("rgb.shape=",rgb.shape)
#================显示效果========================###
cv2.imshow("lena",lena)
cv2.imshow("gray",gray)
cv2.imshow("rgb",rgb)
cv2.waitKey()
cv2.destroyAllWindows()

#4.4 HSV色彩空间
#4.4.2 获取指定颜色
import cv2
import numpy as np
#========测试一下OpenCV中蓝色的HSV模式值========###
imgBlue=np.zeros([1,1,3],dtype=np.uint8)
imgBlue[0,0,0]=255
Blue=imgBlue
BlueHSV=cv2.cvtColor(Blue,cv2.COLOR_BGR2HSV)
print("Blue=\n",Blue)
print("BlueHSV",BlueHSV)
#========测试一下OpenCV中绿色的HSV模式值========###
imgGreen=np.zeros([1,1,3],dtype=np.uint8)
imgGreen[0,0,1]=255
Green=imgGreen
GreenHSV=cv2.cvtColor(Green,cv2.COLOR_BGR2HSV)
print("Green=\n",Green)
print("GreenHSV",GreenHSV)
#========测试一下OpenCV中红色的HSV模式值========###
imgRed=np.zeros([1,1,3],dtype=np.uint8)
imgRed[0,0,2]=255
Red=imgRed
RedHSV=cv2.cvtColor(Red,cv2.COLOR_BGR2HSV)
print("Red=\n",Red)
print("RedHSV",RedHSV)

#4.4.3 标记指定颜色
#使用函数cv2.inRange()将某个图像内的在[100,200]内的值标注出来
import cv2
import numpy as np
img=np.random.randint(0,256,[5,5],dtype=np.uint8)
min=100
max=200
mask=cv2.inRange(img,min,max)
print("img=\n",img)
print("mask=\n",mask)
#通过基于掩码的按位与显示ROI
import cv2
import numpy as np
img=np.ones([5,5],dtype=np.uint8)*9
mask=np.zeros([5,5],dtype=np.uint8)
mask[0:3,0]=1
mask[2:5,2:4]=1
roi=cv2.bitwise_and(img,img,mask=mask)
print("img=\n",img)
print("mask=\n",mask)
print("roi=\n",roi)
#显示特定颜色值
import cv2
import numpy as np
opencv=cv2.imread("opencv.png",-1)
hsv=cv2.cvtColor(opencv,cv2.COLOR_BGR2HSV)
cv2.imshow("opencv",opencv)
#=============指定蓝色值得范围===============###
minBlue=np.array([110,100,100])
maxBlue=np.array([130,255,255])
#确定蓝色区域
mask=cv2.inRange(hsv,minBlue,maxBlue)
#通过掩码控制得按位与运算，锁定蓝色区域
Blue=cv2.bitwise_and(opencv,opencv,mask=mask)
cv2.imshow("Blue",Blue)
#=============指定绿色值得范围===============###
minGreen=np.array([50,100,100])
maxGreen=np.array([70,255,255])
#确定绿色区域
mask=cv2.inRange(hsv,minGreen,maxGreen)
#通过掩码控制得按位与运算，锁定绿色区域
Green=cv2.bitwise_and(opencv,opencv,mask=mask)
cv2.imshow("Green",Green)
#=============指定红色值得范围===============###
minRed=np.array([0,50,50])
maxRed=np.array([30,255,255])
#确定红色区域
mask=cv2.inRange(hsv,minRed,maxRed)
#通过掩码控制得按位与运算，锁定红色区域
Red=cv2.bitwise_and(opencv,opencv,mask=mask)
cv2.imshow("Red",Red)
cv2.waitKey()
cv2.destroyAllWindows()

#实现艺术效果
import cv2
img=cv2.imread("lena.png",-1)
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
h,s,v=cv2.split(hsv)
v[:,:]=255
newHSV=cv2.merge([h,s,v])
art=cv2.cvtColor(newHSV,cv2.COLOR_HSV2BGR)
cv2.imshow("img",img)
cv2.imshow("art",art)
cv2.waitKey()
cv2.destroyAllWindows()

#4.5 alpha通道
import cv2
img=cv2.imread("lena.png",-1)
bgra=cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
b,g,r,a=cv2.split(bgra)
a[:,:]=125
bgra125=cv2.merge([b,g,r,a])
a[:,:]=0
bgra0=cv2.merge([b,g,r,a])
cv2.imwrite("bgra.png",bgra)
cv2.imwrite("bgra125.png",bgra125)
cv2.imwrite("bgra0.png",bgra0)
cv2.imshow("img",img)
cv2.imshow("bgra",bgra)
cv2.imshow("bgra125",bgra125)
cv2.imshow("bgra0",bgra0)
cv2.waitKey()
cv2.destroyAllWindows()


##############################################################################
########################chr 5 几何变换#########################################
##############################################################################
#5.1 缩放
#cv2.resize的行列属性与shape相反 
import cv2
import numpy as np
img=np.ones([2,4,3],dtype=np.uint8)
size=img.shape[:2]
rst=cv2.resize(img,size)
print("img.shape=",img.shape)
print("img=\n",img)
print("rst.shape=",rst.shape)
print("rst=\n",rst)
#完成一个简单的图形缩放
import cv2
import numpy as np
img=cv2.imread("lena.png",-1)
rows,cols=img.shape[:2]
size=(int(cols*0.9),int(rows*0.5))
rst=cv2.resize(img,size)
print("img.shape=",img.shape)
print("rst.shape=",rst.shape)
#根据函数cv2.resize()的fx参数、fy参数完成图形缩放
import cv2
img=cv2.imread("lena.png",-1)
rst=cv2.resize(img,None,fx=2,fy=0.5)
print("img.shape=",img.shape)
print("rst.shape=",rst.shape)

#5.2 翻转
import cv2
img=cv2.imread("lena.png",-1)
x=cv2.flip(img,0)
y=cv2.flip(img,1)
xy=cv2.flip(img,-1)
cv2.imwrite("lenax.png",x)
cv2.imwrite("lenay.png",y)
cv2.imshow("img",img)
cv2.imshow("x",x)
cv2.imshow("y",y)
cv2.imshow("xy",xy)
cv2.waitKey()
cv2.destroyAllWindows()

#5.3 仿射
#5.3.1 平移
#平移转换矩阵M=np.float32([[1,0,x],[0,1,y]])
import cv2
import numpy as np
img=cv2.imread("lena.png",-1)
height,width=img.shape[:2]
x=50
y=100
M=np.float32([[1,0,x],[0,1,y]])
move=cv2.warpAffine(img,M,(width,height))
cv2.imshow("original",img)
cv2.imshow("move",move)
cv2.waitKey()
cv2.destroyAllWindows()
#5.3.2 旋转
#旋转转换矩阵M=cv2.getRotationMatrix2D(center,angle,scale)
import cv2
img=cv2.imread("lena.png",-1)
height,width=img.shape[:2]
M=cv2.getRotationMatrix2D((width/2,height/2),45,0.6)
rotate=cv2.warpAffine(img,M,(width,height))
cv2.imshow("original",img)
cv2.imshow("rotate",rotate)
cv2.waitKey()
cv2.destroyAllWindows()
#5.3.3 更复杂的仿射变换
#转换矩阵M=cv2.getAffineTransform(输入图像的三个点坐标，输出图像的三个点坐标)
#三个点分别为左上角、右上角、左下角
import cv2
import numpy as np
img=cv2.imread("lena.png",-1)
rows,cols,ch=img.shape
p1=np.float32([[0,0],[cols-1,0],[0,rows-1]])
p2=np.float32([[0,rows*0.33],[cols*0.85,rows*0.25],[cols*0.15,rows*0.7]])
M=cv2.getAffineTransform(p1,p2)
dst=cv2.warpAffine(img,M,(cols,rows))
cv2.imshow("original",img)
cv2.imshow("dst",dst)
cv2.waitKey()
cv2.destroyAllWindows()

#5.4 透视
#转换矩阵M=cv2.getPerspectiveTransform(输入图像的四个顶点坐标，输出图像的四个顶点坐标)

#5.5 重映射
#5.5.1 映射参数的理解
import cv2
import numpy as np
img=np.random.randint(0,256,[4,5],dtype=np.uint8)
rows,cols=img.shape
mapx=np.ones(img.shape,np.float32)*3
mapy=np.zeros(img.shape,np.float32)
rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
print("img=\n",img)
print("mapx=\n",mapx)
print("mapy=\n",mapy)
print("rst=\n",rst)
#5.5.2 复制
import cv2
import numpy as np
img=np.random.randint(0,256,size=[4,5],dtype=np.uint8)
rows,cols=img.shape
mapx=np.zeros(img.shape,np.float32)
mapy=np.zeros(img.shape,np.float32)
for i in range(rows):
    for j in range(cols):
        mapx.itemset((i,j),j)
        mapy.itemset((i,j),i)
rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
print("img=\n",img)
print("mapx=\n",mapx)
print("mapy=\n",mapy)
print("rst=\n",rst)
#图像复制
import cv2
import numpy as np
img=cv2.imread("lena.png",-1)
rows,cols=img.shape[:2]
mapx=np.zeros(img.shape[:2],np.float32)
mapy=np.zeros(img.shape[:2],np.float32)
for i in range(rows):
    for j in range(cols):
        mapx.itemset((i,j),j)
        mapy.itemset((i,j),i)
rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
cv2.imshow("original",img)
cv2.imshow("remap",rst)
cv2.waitKey()
cv2.destroyAllWindows()
#5.5.3 绕X轴翻转
#绕X轴翻转时，map2中当前行的行号调整为“总行数-1-当前行号”
import cv2
import numpy as np
img=np.random.randint(0,256,size=[4,5],dtype=np.uint8)
rows,cols=img.shape
mapx=np.zeros(img.shape,np.float32)
mapy=np.zeros(img.shape,np.float32)
for i in range(rows):
    for j in range(cols):
        mapx.itemset((i,j),j)
        mapy.itemset((i,j),rows-1-i)
rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
print("img=\n",img)
print("mapx=\n",mapx)
print("mapy=\n",mapy)
print("rst=\n",rst)
#图像绕X轴翻转
import cv2
import numpy as np
img=cv2.imread("lena.png",-1)
rows,cols=img.shape[:2]
mapx=np.zeros(img.shape[:2],np.float32)
mapy=np.zeros(img.shape[:2],np.float32)
for i in range(rows):
    for j in range(cols):
        mapx.itemset((i,j),j)
        mapy.itemset((i,j),rows-1-i)
rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
cv2.imshow("original",img)
cv2.imshow("result",rst)
cv2.waitKey()
cv2.destroyAllWindows()
#5.5.4 绕Y轴翻转
#绕Y轴翻转时，map1中当前行的行号调整为“总列数-1-当前列号”
import cv2
import numpy as np
img=np.random.randint(0,256,size=[4,5],dtype=np.uint8)
rows,cols=img.shape
mapx=np.zeros(img.shape,np.float32)
mapy=np.zeros(img.shape,np.float32)
for i in range(rows):
    for j in range(cols):
        mapx.itemset((i,j),cols-1-j)
        mapy.itemset((i,j),i)
rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
print("img=\n",img)
print("mapx=\n",mapx)
print("mapy=\n",mapy)
print("rst=\n",rst)
#图像绕Y轴翻转
import cv2
import numpy as np
img=cv2.imread("lena.png",-1)
rows,cols=img.shape[:2]
mapx=np.zeros(img.shape[:2],np.float32)
mapy=np.zeros(img.shape[:2],np.float32)
for i in range(rows):
    for j in range(cols):
        mapx.itemset((i,j),cols-1-j)
        mapy.itemset((i,j),i)
rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
cv2.imshow("original",img)
cv2.imshow("result",rst)
cv2.waitKey()
cv2.destroyAllWindows()
#5.5.5 绕X轴、Y轴翻转
#绕Y轴翻转时，map1中当前行的行号调整为“总列数-1-当前列号”
#绕X轴翻转时，map2中当前行的行号调整为“总行数-1-当前行号”
import cv2
import numpy as np
img=np.random.randint(0,256,size=[4,5],dtype=np.uint8)
rows,cols=img.shape
mapx=np.zeros(img.shape,np.float32)
mapy=np.zeros(img.shape,np.float32)
for i in range(rows):
    for j in range(cols):
        mapx.itemset((i,j),cols-1-j)
        mapy.itemset((i,j),rows-1-i)
rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
print("img=\n",img)
print("mapx=\n",mapx)
print("mapy=\n",mapy)
print("rst=\n",rst)
#图像绕X轴、Y轴翻转
import cv2
import numpy as np
img=cv2.imread("lena.png",-1)
rows,cols=img.shape[:2]
mapx=np.zeros(img.shape[:2],np.float32)
mapy=np.zeros(img.shape[:2],np.float32)
for i in range(rows):
    for j in range(cols):
        mapx.itemset((i,j),cols-1-j)
        mapy.itemset((i,j),rows-1-i)
rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
cv2.imshow("original",img)
cv2.imshow("result",rst)
cv2.waitKey()
cv2.destroyAllWindows()
#5.5.6 X轴、Y轴互换
import cv2
import numpy as np
img=np.random.randint(0,256,size=[4,5],dtype=np.uint8)
rows,cols=img.shape
mapx=np.zeros(img.shape,np.float32)
mapy=np.zeros(img.shape,np.float32)
for i in range(rows):
    for j in range(cols):
        mapx.itemset((i,j),i)
        mapy.itemset((i,j),j)
rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
print("img=\n",img)
print("mapx=\n",mapx)
print("mapy=\n",mapy)
print("rst=\n",rst)
#图像X轴、Y轴互换
import cv2
import numpy as np
img=cv2.imread("lena.png",-1)
rows,cols=img.shape[:2]
mapx=np.zeros(img.shape[:2],np.float32)
mapy=np.zeros(img.shape[:2],np.float32)
for i in range(rows):
    for j in range(cols):
        mapx.itemset((i,j),i)
        mapy.itemset((i,j),j)
rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
cv2.imshow("original",img)
cv2.imshow("result",rst)
cv2.waitKey()
cv2.destroyAllWindows()
#5.5.7 图像缩放
import cv2
import numpy as np
img=cv2.imread("lena.png",-1)
rows,cols=img.shape[:2]
mapx=np.zeros(img.shape[:2],np.float32)
mapy=np.zeros(img.shape[:2],np.float32)
for i in range(rows):
    for j in range(cols):
        if 0.25*cols<i<0.75*cols and 0.25*rows<j<0.75*rows:
            mapx.itemset((i,j),2*(j-cols*0.25)+0.5)
            mapy.itemset((i,j),2*(i-rows*0.25)+0.5)
        else:
            mapx.itemset((i,j),0)
            mapy.itemset((i,j),0)
rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
cv2.imshow("original",img)
cv2.imshow("result",rst)
cv2.waitKey()
cv2.destroyAllWindows()


##############################################################################
########################chr 6 阈值处理#########################################
##############################################################################
#6.1 threshold函数
#6.1.1 二值化阈值处理(cv2.THRESH_BINARY)
import cv2
import numpy as np
img=np.random.randint(0,256,[4,5],dtype=np.uint8)
t,rst=cv2.threshold(img,127,256,cv2.THRESH_BINARY)
print("img=\n",img)
print("t=",t)
print("rst=\n",rst)
#处理图像
import cv2
import numpy as np
img=cv2.imread("lena.png",0)
t,rst=cv2.threshold(img,100,256,cv2.THRESH_BINARY)
cv2.imshow("img",img)
cv2.imshow("rst",rst)
cv2.waitKey()
cv2.destroyAllWindows()
#6.1.2 反二值化阈值处理(cv2.THRESH_BINARY_INV)
import cv2
import numpy as np
img=cv2.imread("lena.png",0)
t,rst=cv2.threshold(img,100,256,cv2.THRESH_BINARY_INV)
cv2.imshow("img",img)
cv2.imshow("rst",rst)
cv2.waitKey()
cv2.destroyAllWindows()
#6.1.3 截断阈值化处理(cv2.THRESH_TRUNC)
import cv2
import numpy as np
img=cv2.imread("lena.png",0)
r,rst=cv2.threshold(img,100,256,cv2.THRESH_TRUNC)
cv2.imshow("img",img)
cv2.imshow("rst",rst)
cv2.waitKey()
cv2.destroyAllWindows()
#6.1.4 超阈值零处理(cv2.THRESH_TOZERO_INV)
import cv2
import numpy as np
img=cv2.imread("lena.png",0)
t,rst=cv2.threshold(img,100,256,cv2.THRESH_TOZERO_INV)
cv2.imshow("img",img)
cv2.imshow("rst",rst)
cv2.waitKey()
cv2.destroyAllWindows()
#6.1.4 低阈值零处理(cv2.THRESH_TOZERO)
import cv2
import numpy as np
img=cv2.imread("lena.png",0)
t,rst=cv2.threshold(img,100,256,cv2.THRESH_TOZERO)
cv2.imshow("img",img)
cv2.imshow("rst",rst)
cv2.waitKey()
cv2.destroyAllWindows()

#6.2 自适应阈值处理
#cv2.adaptiveThreshold()
#分别使用cv2.Threshold()和 cv2.adaptiveThreshold()处理图像
import cv2
import numpy as np
img=cv2.imread("lena.png",0)
t,thd=cv2.threshold(img,100,255,cv2.THRESH_BINARY)
athdMEAN=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,3)
athdGAUS=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,3)
cv2.imshow("img",img)
cv2.imshow("thd",thd)
cv2.imshow("athdMEAN",athdMEAN)
cv2.imshow("athdGAUS",athdGAUS)
cv2.waitKey()
cv2.destroyAllWindows()

#Otsu处理
import cv2
import numpy as np
img=cv2.imread("lena.png",0)
t1,thd=cv2.threshold(img,100,255,cv2.THRESH_BINARY)
t2,otsu=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print("t1=",t1)
print("t2=",t2)
cv2.imshow("thd",thd)
cv2.imshow("otsu",otsu)
cv2.waitKey()
cv2.destroyAllWindows()


##############################################################################
########################chr 7 图像平滑处理#####################################
##############################################################################
#7.1 均值滤波
import cv2
o=cv2.imread("image1.png",-1)
r1=cv2.blur(o,(5,5))
r2=cv2.blur(o,(10,10))
cv2.imshow("original",o)
cv2.imshow("result1",r1)
cv2.imshow("result2",r2)
cv2.waitKey()
cv2.destroyAllWindows()
#7.2 方框滤波
#cv2.boxFilter()
import cv2
o=cv2.imread("image1.png",-1)
r1=cv2.boxFilter(o,-1,(5,5))
r2=cv2.boxFilter(o,-1,(10,10))
r3=cv2.boxFilter(o,-1,(2,2),normalize=0)
cv2.imshow("original",o)
cv2.imshow("result1",r1)
cv2.imshow("result2",r2)
cv2.imshow("result3",r3)
cv2.waitKey()
cv2.destroyAllWindows()
#7.3 高斯滤波
#dst=cv2.GaussianBlur()
import cv2
o=cv2.imread("image1.png",-1)
r1=cv2.GaussianBlur(o,(5,5),0,0)
r2=cv2.GaussianBlur(o,(7,7),0,0)
cv2.imshow("original",o)
cv2.imshow("result1",r1)
cv2.imshow("result2",r2)
cv2.waitKey()
cv2.destroyAllWindows()
#7.4 中值滤波
#dst=cv2.medianBlur()
import cv2
o=cv2.imread("image1.png",-1)
r1=cv2.medianBlur(o,3)
r2=cv2.medianBlur(o,7)
cv2.imshow("original",o)
cv2.imshow("result1",r1)
cv2.imshow("result2",r2)
cv2.waitKey()
cv2.destroyAllWindows()
#7.5 双边滤波
#cv2.bilateralFilter()
import cv2
o=cv2.imread("lena.png",-1)
r1=cv2.bilateralFilter(o,50,150,150)
cv2.imshow("original",o)
cv2.imshow("result1",r1)
cv2.waitKey()
cv2.destroyAllWindows()
#7.6 2D卷积
#自定义卷积核 cv2.filter2D()
import cv2
import numpy as np
o=cv2.imread("image1.png",-1)
kernel=np.ones((9,9),np.float32)/81
r1=cv2.filter2D(o,-1,kernel)
cv2.imshow("original",o)
cv2.imshow("result1",r1)
cv2.waitKey()
cv2.destroyAllWindows()


##############################################################################
########################chr 8 形态学操作#######################################
##############################################################################
#8.1 腐蚀
import cv2
import numpy as np
o=cv2.imread("lena.png",-1)
kernel=np.ones((5,5),np.uint8)
r1=cv2.erode(o,kernel)
r2=cv2.erode(o,kernel,iterations=5)
cv2.imshow("original",o)
cv2.imshow("result1",r1)
cv2.imshow("result2",r2)
cv2.waitKey()
cv2.destroyAllWindows()
#8.2 膨胀
import cv2
import numpy as np
o=cv2.imread("lena.png",-1)
kernel1=np.ones((9,9),np.uint8)
kernel2=np.ones((5,5),np.uint8)
r1=cv2.dilate(o,kernel1)
r2=cv2.dilate(o,kernel2)
r3=cv2.dilate(o,kernel2,iterations=9)
cv2.imshow("original",o)
cv2.imshow("result1",r1)
cv2.imshow("result2",r2)
cv2.imshow("result3",r3)
cv2.waitKey()
cv2.destroyAllWindows()
#8.4 开运算
#开运算先腐蚀再膨胀，可以去噪，cv2.morphologyEx() op=cv.MORPH_OPEN
import cv2
import numpy as np
o=cv2.imread("lena.png",-1)
k=np.ones((5,5),np.uint8)
r=cv2.morphologyEx(o,cv2.MORPH_OPEN,k)
cv2.imshow("original",o)
cv2.imshow("result1",r)
cv2.waitKey()
cv2.destroyAllWindows()
#8.5 闭运算
#闭运算先膨胀后腐蚀，有助于关闭前景物体内部小孔，cv2.morphologyEx() op=cv.MORPH_CLOSE
import cv2
import numpy as np
o=cv2.imread("lena.png",-1)
k=np.ones((5,5),np.uint8)
r=cv2.morphologyEx(o,cv2.MORPH_CLOSE,k)
cv2.imshow("original",o)
cv2.imshow("result1",r)
cv2.waitKey()
cv2.destroyAllWindows()
#8.6 形态学梯度运算
#用膨胀图像减去腐蚀图像, cv2.morphologyEx() op=cv.MORPH_GRADIENT
import cv2
import numpy as np
o=cv2.imread("lena.png",-1)
k=np.ones((5,5),np.float32)
r=cv2.morphologyEx(o,cv2.MORPH_GRADIENT,k)
cv2.imshow("original",o)
cv2.imshow("result1",r)
cv2.waitKey()
cv2.destroyAllWindows()
#8.7 礼帽运算
#原始图像减去开运算，cv2.morphologyEx() op=cv.MORPH_TOPHAT
import cv2
import numpy as np
o=cv2.imread("lena.png",-1)
k=np.ones((5,5),np.float32)
r=cv2.morphologyEx(o,cv2.MORPH_TOPHAT,k)
cv2.imshow("original",o)
cv2.imshow("result1",r)
cv2.waitKey()
cv2.destroyAllWindows()
#8.8 黑帽运算
#闭运算减去原始图像，可以获取图像内部小孔，cv2.morphologyEx() op=cv.MORPH_BLACKHAT
import cv2
import numpy as np
o=cv2.imread("lena.png",-1)
k=np.ones((5,5),np.float32)
r=cv2.morphologyEx(o,cv2.MORPH_BLACKHAT,k)
cv2.imshow("original",o)
cv2.imshow("result1",r)
cv2.waitKey()
cv2.destroyAllWindows()
# 8.9 核函数
import cv2
kernel1=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
kernel2=cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
kernel3=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
print("kernel1=\n",kernel1)
print("kernel2=\n",kernel2)
print("kernel3=\n",kernel3)
#处理图像
import cv2
o=cv2.imread("lena.png",-1)
kernel1=cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
kernel2=cv2.getStructuringElement(cv2.MORPH_CROSS,(15,15))
kernel3=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
dst1=cv2.dilate(o,kernel1)
dst2=cv2.dilate(o,kernel2)
dst3=cv2.dilate(o,kernel3)
cv2.imshow("original",o)
cv2.imshow("result1",dst1)
cv2.imshow("result2",dst2)
cv2.imshow("result3",dst3)
cv2.waitKey()
cv2.destroyAllWindows()


##############################################################################
########################chr 9 图像梯度#########################################
##############################################################################
#9.2 Sobel算子
#使用cv2.convertScaleAbs()对一个随机数组取绝对值
import cv2
import numpy as np
img=np.random.randint(-256,256,[4,5],np.int16)
rst=cv2.convertScaleAbs(img)
print("img=\n",img)
print("rst=\n",rst)
#方向
#计算X方向梯度：dx=1,dy=0 ,计算Y方向梯度：dx=0,dy=1
#计算X方向和Y方向的边缘叠加，需分别计算两个方向的，然后二者相加 cv2.addWeighted()
#实例：使用cv2.Sobel()获取图像水平方向的边缘信息
import cv2
o=cv2.imread("lena.png",0)
Sobelx=cv2.Sobel(o,cv2.CV_64F,1,0)
Sobelx=cv2.convertScaleAbs(Sobelx)
cv2.imshow("original",o)
cv2.imshow("Sobelx",Sobelx)
cv2.waitKey()
cv2.destroyAllWindows()
#实例：使用cv2.Sobel()获取图像垂直方向的边缘信息
import cv2
o=cv2.imread("lena.png",0)
Sobely=cv2.Sobel(o,cv2.CV_64F,0,1)
Sobely=cv2.convertScaleAbs(Sobely)
cv2.imshow("original",o)
cv2.imshow("Sobely",Sobely)
cv2.waitKey()
cv2.destroyAllWindows()
#实例：使用cv2.Sobel()获取图像水平和垂直方向的边缘信息
import cv2
o=cv2.imread("lena.png",0)
Sobelx=cv2.Sobel(o,cv2.CV_64F,1,0)
Sobelx=cv2.convertScaleAbs(Sobelx)
Sobely=cv2.Sobel(o,cv2.CV_64F,0,1)
Sobely=cv2.convertScaleAbs(Sobely)
Sobelxy=cv2.addWeighted(Sobelx,0.5,Sobely,0.5,0)
cv2.imshow("original",o)
cv2.imshow("Sobelx",Sobelx)
cv2.imshow("Sobely",Sobely)
cv2.imshow("Sobelxy",Sobelxy)
cv2.waitKey()
cv2.destroyAllWindows()

#9.3 Scharr算子
import cv2
o=cv2.imread("lena.png",0)
Sobelx=cv2.Sobel(o,cv2.CV_64F,1,0)
Sobelx=cv2.convertScaleAbs(Sobelx)
Sobely=cv2.Sobel(o,cv2.CV_64F,0,1)
Sobely=cv2.convertScaleAbs(Sobely)
Sobelxy=cv2.addWeighted(Sobelx,0.5,Sobely,0.5,0)
Scharrx=cv2.Scharr(o,cv2.CV_64F,1,0)
Scharrx=cv2.convertScaleAbs(Scharrx)
Scharry=cv2.Scharr(o,cv2.CV_64F,0,1)
Scharry=cv2.convertScaleAbs(Scharry)
Scharrxy=cv2.addWeighted(Scharrx,0.5,Scharry,0.5,0)
cv2.imshow("original",o)
cv2.imshow("Sobelx",Sobelx)
cv2.imshow("Sobely",Sobely)
cv2.imshow("Sobelxy",Sobelxy)
cv2.imshow("Scharrx",Scharrx)
cv2.imshow("Scharry",Scharry)
cv2.imshow("Scharrxy",Scharrxy)
cv2.waitKey()
cv2.destroyAllWindows()

#9.5 Laplacian算子
#不用分别计算X和Y方向上的梯度
import cv2
o=cv2.imread("lena.png",0)
Laplacian=cv2.Laplacian(o,cv2.CV_64F)
Laplacian=cv2.convertScaleAbs(Laplacian)
cv2.imshow("original",o)
cv2.imshow("Laplacian",Laplacian)
cv2.waitKey()
cv2.destroyAllWindows()


##############################################################################
########################chr 10 Canny边缘检测###################################
##############################################################################
import cv2
o=cv2.imread("lena.png",0)
r1=cv2.Canny(o,128,200)
r2=cv2.Canny(o,32,128)
cv2.imshow("original",o)
cv2.imshow("result1",r1)
cv2.imshow("result2",r2)
cv2.waitKey()
cv2.destroyAllWindows()


##############################################################################
########################chr 11 图像金字塔######################################
##############################################################################
#11.2 pyrDown()下采样函数
import cv2
o=cv2.imread("lena.png",-1)
r1=cv2.pyrDown(o)
r2=cv2.pyrDown(r1)
r3=cv2.pyrDown(r2)
print("o.shape=",o.shape)
print("r1.shape=",r1.shape)
print("r2.shape=",r2.shape)
print("r3.shape=",r3.shape)
cv2.imshow("original",o)
cv2.imshow("result1",r1)
cv2.imshow("result2",r2)
cv2.imshow("result3",r3)
cv2.waitKey()
cv2.destroyAllWindows()
#11.3 pyrUp()上采样函数
import cv2
o=cv2.imread("lena.png",-1)
r1=cv2.pyrUp(o)
r2=cv2.pyrUp(r1)
print("o.shape=",o.shape)
print("r1.shape=",r1.shape)
print("r2.shape=",r2.shape)
cv2.imshow("original",o)
cv2.imshow("result1",r1)
cv2.imshow("result2",r2)
cv2.waitKey()
cv2.destroyAllWindows()
#11.5 拉普拉斯金字塔
import cv2
import numpy as np
o=cv2.imread("lena512color.tif",-1)
#=====================生成高斯金字塔================================###
G0=o
G1=cv2.pyrDown(G0)
G2=cv2.pyrDown(G1)
G3=cv2.pyrDown(G2)
#===================生成拉普拉斯金字塔==============================###
L0=G0-cv2.pyrUp(G1)  #拉普拉斯金字塔第0层
L1=G1-cv2.pyrUp(G2)  #拉普拉斯金字塔第1层
L2=G2-cv2.pyrUp(G3)  #拉普拉斯金字塔第2层
#=========================复原G0====================================###
RG0=L0+cv2.pyrUp(G1) #通过拉普拉斯金字塔第复原的原始图像G0
print("G0.shape=",G0.shape)
print("RG0.shape=",RG0.shape)
result1=RG0-G0
result1=abs(result1)
print("原始图像G0与回复图像RG0差值的绝对值和：",np.sum(result1))
#=========================复原G1====================================###
RG1=L1+cv2.pyrUp(G2) #通过拉普拉斯金字塔第复原的原始图像G1
print("G1.shape=",G1.shape)
print("RG1.shape=",RG1.shape)
result2=RG1-G1
result2=abs(result2)
print("原始图像G1与回复图像RG1差值的绝对值和：",np.sum(result2))
#=========================复原G2====================================###
RG2=L2+cv2.pyrUp(G3) #通过拉普拉斯金字塔第复原的原始图像G2
print("G2.shape=",G2.shape)
print("RG2.shape=",RG2.shape)
result3=RG2-G2
result3=abs(result3)
print("原始图像G2与回复图像RG2差值的绝对值和：",np.sum(result3))


##############################################################################
########################chr 12 图像轮廓########################################
##############################################################################
#12.1 查找并绘制轮廓 cv2.findContours() cv2.drawContour()
#背景为黑
import cv2
o=cv2.imread("contours1.jpg",-1)
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
img,contours,hierarchy=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
o=cv2.drawContours(o,contours,-1,(0,0,255),5)
cv2.imshow("result",o)
cv2.waitKey()
cv2.destroyAllWindows()
#背景为白
import cv2
o=cv2.imread("contours2.jpg",-1)
cv2.imshow("original",o)
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
img,contours,hierarchy=cv2.findContours(binary,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
o=cv2.drawContours(o,contours,-1,(0,0,0),5)
cv2.imshow("result",o)
cv2.waitKey()
cv2.destroyAllWindows()
#逐个显示一幅图像中的边缘信息
import cv2
import numpy as np
o=cv2.imread("contours2.jpg",-1)
cv2.imshow("original",o)
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
n=len(contours)
contoursImg=[]
for i in range(n):
    temp=np.ones(o.shape,np.uint8)*225
    contoursImg.append(temp)
    contoursImg[i]=cv2.drawContours(contoursImg[i],contours,i,(0,0,0),5)
    cv2.imshow("contours["+ str(i)+"]",contoursImg[i])
cv2.waitKey()
cv2.destroyAllWindows()
#使用轮廓绘制功能，提取前景对象
#黑背景
import cv2
import numpy as np
o=cv2.imread("lena.jpg",-1)
cv2.imshow("original",o)
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,bindary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
image,contours,hierarchy=cv2.findContours(bindary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
mask=np.zeros(o.shape,np.uint8)
mask=cv2.drawContours(mask,contours,-1,(255,255,255),-1)
cv2.imshow("mask",mask)
loc=cv2.bitwise_and(o,mask)
cv2.imshow("location",loc)
cv2.waitKey()
cv2.destroyAllWindows()
#白背景
import cv2
import numpy as np
o=cv2.imread("contours2.jpg",-1)
cv2.imshow("original",o)
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,bindary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
image,contours,hierarchy=cv2.findContours(bindary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
mask=np.ones(o.shape,np.uint8)*255
mask=cv2.drawContours(mask,contours,-1,(0,0,0),-1)
cv2.imshow("mask",mask)
loc=cv2.bitwise_and(o,mask)
cv2.imshow("location",loc)
cv2.waitKey()
cv2.destroyAllWindows()

#12.2 矩特征
#12.2.1 矩的计算：moments函数
#使用函数cv2.moments()提取一幅图像的特征
import cv2
import numpy as np
o=cv2.imread("contours2.jpg",-1)
cv2.imshow("original",o)
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
n=len(contours)
contoursImg=[]
for i in range(n):
    temp=np.ones(o.shape,np.uint8)*255
    contoursImg.append(temp)
    contoursImg[i]=cv2.drawContours(contoursImg[i],contours,i,3)
    cv2.imshow("Contours[" + str(i)+"]",contoursImg[i])
print("观察各个轮廓的矩（moments）:")
for i in range(n):
    print("轮廓"+str(i)+"的矩：\n",cv2.moments(contours[i]))
print("观察各个轮廓的面积：")
for i in range(n):
    print("轮廓"+str(i)+"的面积:%d" %cv2.moments(contours[i])['m00'])
cv2.waitKey()
cv2.destroyAllWindows()

#12.2.2 计算轮廓的面积：contourArea 函数
import cv2
import numpy as np
o=cv2.imread("contours2.jpg",-1)
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
n=len(contours)
contoursImg=[]
for i in range(n):
    print("contours["+str(i)+"]面积:",cv2.contourArea(contours[i]))
    temp=np.ones(o.shape,np.uint8)*255
    contoursImg.append(temp)
    contoursImg[i]=cv2.drawContours( contoursImg[i],contours,i,(0,0,0),3)
    cv2.imshow("contours["+str(i)+"]",contoursImg[i])
cv2.waitKey()
cv2.destroyAllWindows()
#将面积大于23500的轮廓筛选出来
import cv2
import numpy as np
o=cv2.imread("contours2.jpg",-1)
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
n=len(contours)
contoursImg=[]
for i in range(n):
    print("contours["+str(i)+"]面积:",cv2.contourArea(contours[i]))
    temp=np.ones(o.shape,np.uint8)*255
    contoursImg.append(temp)
    contoursImg[i]=cv2.drawContours( contoursImg[i],contours,i,(0,0,0),3)
    if cv2.contourArea(contours[i])>23500:
        cv2.imshow("contours["+str(i)+"]",contoursImg[i])
cv2.waitKey()
cv2.destroyAllWindows()

#12.2.3 计算轮廓的长度：arcLength函数
import cv2
import numpy as np
#===============读取并显示原始图像==========================###
o=cv2.imread("contours2.jpg",-1)
cv2.imshow("original",o)
#================获取轮廓==================================###
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#===============计算各轮廓的长度之和、平均长度===============###
n=len(contours)   #获取轮廓的个数
cnLen=[]          #存储个轮廓的长度
for i in range(n):
    cnLen.append(cv2.arcLength(contours[i],True))
    print("第"+str(i)+"个轮廓的长度:%d" %cnLen[i])
cnLenSum=np.sum(cnLen)   #各轮廓的长度之和
cnLenAvr=cnLenSum/n      #轮廓长度的平均值
print("轮廓的总长度为：%d" %cnLenSum)
print("轮廓的平均长度为：%d" %cnLenAvr)
#==============显示长度超过平均值的轮廓======================###
contoursImg=[]
for i in range(n):
    temp=np.ones(o.shape,np.uint8)*255
    contoursImg.append(temp)
    contoursImg[i]=cv2.drawContours(contoursImg[i],contours,i,(0,0,0),3)
    if cv2.arcLength(contours[i],True)>cnLenAvr:
        cv2.imshow("contours["+str(i)+"]",contoursImg[i])
cv2.waitKey()
cv2.destroyAllWindows()

#12.3 Hu矩
#12.3.1 Hu 矩函数
#计算三幅不同图像的Hu矩，并进行比较
import cv2
import numpy as np
#=============================计算o1的 Hu矩==================###
o1=cv2.imread("lena.png",-1)
gray1=cv2.cvtColor(o1,cv2.COLOR_BGR2GRAY)
HuM1=cv2.HuMoments(cv2.moments(gray1)).flatten()
#=============================计算o2的 Hu矩==================###
o2=cv2.imread("lenax.png",-1)
gray2=cv2.cvtColor(o2,cv2.COLOR_BGR2GRAY)
HuM2=cv2.HuMoments(cv2.moments(gray2)).flatten()
#=============================计算o3的 Hu矩==================###
o3=cv2.imread("lenay.png",-1)
gray3=cv2.cvtColor(o3,cv2.COLOR_BGR2GRAY)
HuM3=cv2.HuMoments(cv2.moments(gray3)).flatten()
#=============================计算o4的 Hu矩==================###
o4=cv2.imread("opencv.png",-1)
gray4=cv2.cvtColor(o4,cv2.COLOR_BGR2GRAY)
HuM4=cv2.HuMoments(cv2.moments(gray4)).flatten()
#=========打印图像o1、o2、o3、o4 的特征值======================###
print("o1.shape=",o1.shape)
print("o2.shape=",o2.shape)
print("o3.shape=",o3.shape)
print("o4.shape=",o4.shape)
print("cv2.moments(gray1)=\n",cv2.moments(gray1))
print("cv2.moments(gray2)=\n",cv2.moments(gray2))
print("cv2.moments(gray3)=\n",cv2.moments(gray3))
print("cv2.moments(gray4)=\n",cv2.moments(gray4))
print("\nHuM1=\n",HuM1)
print("\nHuM2=\n",HuM2)
print("\nHuM3=\n",HuM3)
print("\nHuM4=\n",HuM4)
#=============计算图像o1与o2、o3、o4的Hu矩之差==================###
print("\nHuM1-HuM2=",HuM1-HuM2)
print("\nHuM1-HuM3=",HuM1-HuM3)
print("\nHuM1-HuM4=",HuM1-HuM4)
#===================显示图像===================================###
cv2.imshow("original1",o1)
cv2.imshow("original2",o2)
cv2.imshow("original3",o3)
cv2.imshow("original4",o4)
cv2.waitKey()
cv2.destroyAllWindows()

#12.3.2 形状匹配 cv2.matchShapes()
#准备三幅图像
import cv2
import numpy as np
o=cv2.imread("contours6.jpg",-1)
r=cv2.pyrDown(o)
r1=cv2.flip(r,0)
k=np.ones((20,20),np.float32)
r2=cv2.erode(r,k)
cv2.imwrite("contours6_down.jpg",r)
cv2.imwrite("contours6x.jpg",r1)
cv2.imwrite("contours6_erode.jpg",r2)
cv2.imshow("o",o)
cv2.imshow("r",r)
cv2.imshow("r1",r1)
cv2.imshow("r2",r2)
cv2.waitKey()
cv2.destroyAllWindows()
#使用cv2.matchShapes()计算三幅图像的匹配度
import cv2
o1=cv2.imread("contours6_down.jpg",-1)
o2=cv2.imread("contours6x.jpg",-1)
o3=cv2.imread("contours6_erode.jpg",-1)
o4=cv2.imread("lena.jpg",-1)
gray1=cv2.cvtColor(o1,cv2.COLOR_BGR2GRAY)
gray2=cv2.cvtColor(o2,cv2.COLOR_BGR2GRAY)
gray3=cv2.cvtColor(o3,cv2.COLOR_BGR2GRAY)
gray4=cv2.cvtColor(o4,cv2.COLOR_BGR2GRAY)
ret,binary1=cv2.threshold(gray1,127,255,cv2.THRESH_BINARY)
ret,binary2=cv2.threshold(gray2,127,255,cv2.THRESH_BINARY)
ret,binary3=cv2.threshold(gray3,127,255,cv2.THRESH_BINARY)
ret,binary4=cv2.threshold(gray4,127,255,cv2.THRESH_BINARY)
image,contours1,hierarchy=cv2.findContours(binary1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
image,contours2,hierarchy=cv2.findContours(binary2,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
image,contours3,hierarchy=cv2.findContours(binary3,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
image,contours4,hierarchy=cv2.findContours(binary4,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnt1=contours1[0]
cnt2=contours2[0]
cnt3=contours3[0]
cnt4=contours4[0]
ret0=cv2.matchShapes(cnt1,cnt1,1,0.0)
ret1=cv2.matchShapes(cnt1,cnt2,1,0.0)
ret2=cv2.matchShapes(cnt1,cnt3,1,0.0)
ret3=cv2.matchShapes(cnt1,cnt4,1,0.0)
print("相同图像的matchShape=",ret0)
print("相似图像的matchShape=",ret1)
print("相似图像的matchShape=",ret2)
print("不同图像的matchShape=",ret3)

#12.4 轮廓拟合
#12.4.1 矩形包围框 cv2.boundingRect()
#显示cv2.boundingRect()不同形式的返回值
import cv2
#=============================读取并显示原始图像=======================###
o=cv2.imread("contours3_one.jpg",-1)
#=============================提取图像轮廓=============================###
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#===========================返回顶点及边长=============================###
x,y,w,h=cv2.boundingRect(contours[0])
print("顶点及长宽的点形式：")
print("x=",x)
print("y=",y)
print("w=",w)
print("h=",h)
#========================仅有一个返回值的情况===========================###
rect=cv2.boundingRect(contours[0])
print("\n顶点及长宽的元组(tuple)形式:")
print("rect=",rect)

#使用函数cv2.drawContours()绘制矩形包围框
import cv2
import numpy as np
#=========================读取并显示原始图像============================###
o=cv2.imread("contours3_one.jpg",-1)
cv2.imshow("original",o)
#=========================提取图像轮廓===============================###
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#==========================构造矩形边界==============================###
x,y,w,h=cv2.boundingRect(contours[0])
brcnt=np.array([[[x,y]],[[x+w,y]],[[x+w,y+h]],[[x,y+h]]])
cv2.drawContours(o,[brcnt],-1,(255,255,255),2)
#==========================显示矩形边界===============================###
cv2.imshow("result",o)
cv2.waitKey()
cv2.destroyAllWindows()
#使用函数cv2.boundingRect()及cv2.rectangle()绘制矩形包围框
import cv2
#=========================读取并显示原始图像============================###
o=cv2.imread("contours3_one.jpg",-1)
cv2.imshow("original",o)
#=========================提取图像轮廓===============================###
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#==========================构造矩形边界==============================###
x,y,w,h=cv2.boundingRect(contours[0])
cv2.rectangle(o,(x,y),(x+w,y+h),(255,255,255),2)
#==========================显示矩形边界===============================###
cv2.imshow("result",o)
cv2.waitKey()
cv2.destroyAllWindows()

#12.4.2 最小包围矩形框
#使用cv2.minAreaRect() 函数计算图像的最小包围矩形框
import cv2
import numpy as np
o=cv2.imread("contours3_one.jpg",-1)
cv2.imshow("original",o)
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
rect=cv2.minAreaRect(contours[0])
print("返回值rect:\n",rect)
points=cv2.boxPoints(rect)
print("\n转换后的points:\n",points)
points=np.int0(points)   #取整
print("\n取整后的points:\n",points)
image=cv2.drawContours(o,[points],0,(255,255,255),2)
cv2.imshow("result",o)
cv2.waitKey()
cv2.destroyAllWindows()

#12.4.3 最小包围圆圈
#使用cv2.minEnclosingCircle()构造图像的最小包围圆圈
import cv2
import numpy as np
o=cv2.imread("contours3_one.jpg",-1)
cv2.imshow("original",o)
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
(x,y),radius=cv2.minEnclosingCircle(contours[0])
center=(int(x),int(y))
radius=int(radius)
cv2.circle(o,center,radius,(255,255,255),2)
cv2.imshow("result",o)
cv2.waitKey()
cv2.destroyAllWindows()

#12.4.4 最优拟合椭圆
#使用cv2.ellipse()根据函数cv2.fitEllipse()的返回值绘制最优拟合椭圆
import cv2
o=cv2.imread("contours3_one.jpg",-1)
cv2.imshow("original",o)
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
ellipse=cv2.fitEllipse(contours[0])
print("ellipse=",ellipse)
cv2.ellipse(o,ellipse,(0,255,0),3)
cv2.imshow("result",o)
cv2.waitKey()
cv2.destroyAllWindows()

#12.4.5 最优拟合直线
#使用函数cv2.fitLine()构造最优拟合曲线
o=cv2.imread("contours3_one.jpg",-1)
cv2.imshow("original",o)
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
rows,cols=image.shape[:2]
[vx,vy,x,y]=cv2.fitLine(contours[0],cv2.DIST_L2,0,0.01,0.01)
lefty=int((-x*vy/vx)+y)
righty=int(((cols-x)*vy/vx)+y)
cv2.line(o,(cols-1,righty),(0,lefty),(0,255,0),2)
cv2.imshow("result",o)
cv2.waitKey()
cv2.destroyAllWindows()

#12.4.6 最小外包三角
#使用函数cv2.minEnclosingTriangle()构造最小外包三角形
o=cv2.imread("contours3_one.jpg",-1)
cv2.imshow("original",o)
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
area,trg1=cv2.minEnclosingTriangle(contours[0])
print("area=",area)
print("trg1:",trg1)
for i in range(0,3):
    cv2.line(o,tuple(trg1[i][0]),tuple(trg1[(i+1) % 3][0]),(255,255,255),2)
cv2.imshow("result",o)
cv2.waitKey()
cv2.destroyAllWindows()

#12.4.7 逼近多边形
#使用函数cv2.approxPolyDP()构造不同精度的逼近多边形
import cv2
##=====================读取并显示原始图像==========================##
o=cv2.imread("contours3_one.jpg",-1)
cv2.imshow("original",o)
##======================获取轮廓==================================###
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
##======================epsilon=0.1*周长=========================###
adp=o.copy()    
epsilon=0.1*cv2.arcLength(contours[0],True)
approx=cv2.approxPolyDP(contours[0],epsilon,True)
adp=cv2.drawContours(adp,[approx],0,(0,0,225),2)
cv2.imshow("result0.1",adp)
##====================epsilon=0.09*周长===========================###
adp=o.copy()
epsilon=0.09*cv2.arcLength(contours[0],True)
approx=cv2.approxPolyDP(contours[0],epsilon,True)
adp=cv2.drawContours(adp,[approx],0,(0,0,255),2)
cv2.imshow("result0.09",adp)
##====================epsilon=0.055*周长===========================###
adp=o.copy()
epsilon=0.055*cv2.arcLength(contours[0],True)
approx=cv2.approxPolyDP(contours[0],epsilon,True)
adp=cv2.drawContours(adp,[approx],0,(0,0,255),2)
cv2.imshow("result0.055",adp)
##====================epsilon=0.05*周长===========================###
adp=o.copy()
epsilon=0.05*cv2.arcLength(contours[0],True)
approx=cv2.approxPolyDP(contours[0],epsilon,True)
adp=cv2.drawContours(adp,[approx],0,(0,0,255),2)
cv2.imshow("result0.05",adp)
##====================epsilon=0.02*周长===========================###
adp=o.copy()
epsilon=0.02*cv2.arcLength(contours[0],True)
approx=cv2.approxPolyDP(contours[0],epsilon,True)
adp=cv2.drawContours(adp,[approx],0,(0,0,255),2)
cv2.imshow("result0.02",adp)
##=====================等待释放窗口==================================##
cv2.waitKey()
cv2.destroyAllWindows()

#12.5 凸包
#观察函数cv2.convexHull()内参数 returnPoints的使用情况
import cv2
o=cv2.imread("finger.png",-1)
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
hull=cv2.convexHull(contours[0])   #返回坐标值
print("returnPoints为默认值True时返回值hull的值：\n",hull)
hull2=cv2.convexHull(contours[0],returnPoints=False)   #返回索引值
print("returnPoints为False时返回值hull的值：\n",hull2)
#使用函数 cv2.convexHull()获取轮廓的凸包
import cv2
o=cv2.imread("finger.png",-1)
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
hull=cv2.convexHull(contours[0])   #返回坐标值
cv2.polylines(o,[hull],True,(0,255,0),2)
cv2.imshow("result",o)
cv2.waitKey()
cv2.destroyAllWindows()

#12.5.2 凸缺陷
#使用函数 cv2.convexityDefects()计算凸缺陷
import cv2
import numpy as np
#=========================原图=========================================###
img=cv2.imread("finger.png",-1)
cv2.imshow("original",img)
#=========================构造轮廓=====================================###
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#==========================凸包=======================================###
cnt=contours[0]
hull=cv2.convexHull(cnt,returnPoints=False)
defects=cv2.convexityDefects(cnt,hull)
print("defects=\n",defects)
#=========================构造凸缺陷==================================###
for i in range(defects.shape[0]):
    s,e,f,d=defects[i,0]
    start=tuple(cnt[s][0])
    end=tuple(cnt[e][0])
    far=tuple(cnt[f][0])
    cv2.line(img,start,end,[0,0,255],2)
    cv2.circle(img,far,5,[255,0,0],-1)
#====================显示结果，释放图像================================###
cv2.imshow("result",img)  
cv2.waitKey()
cv2.destroyAllWindows()

#12.5.3 几何学测试
#使用函数 cv2.isContourConvex()来判断轮廓是否是凸形的
import cv2
o=cv2.imread("finger.png",-1)
cv2.imshow("original",o)
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#=============================凸包====================================###
image1=o.copy()
hull=cv2.convexHull(contours[0])
cv2.polylines(image1,[hull],True,(0,255,0),2)
print("使用函数cv2.convexHull()构造的多边形是否是凸形的:",cv2.isContourConvex(hull))
cv2.imshow("result1",image1)
#==========================逼近多边形================================####
image2=o.copy()
epsilon=0.01*cv2.arcLength(contours[0],True)
approx=cv2.approxPolyDP(contours[0],epsilon,True)
image2=cv2.drawContours(image2,[approx],0,(0,0,225),2)
print("使用函数cv2.approxPolyDP()构造的多边形是否是凸形的:",cv2.isContourConvex(approx))
cv2.imshow("result2",image2)
#==========================释放窗口===================================### 
cv2.waitKey()
cv2.destroyAllWindows()

#使用函数 cv2.pointPolygonTest()计算点到轮廓的最短距离，需要将参数measureDist的值设置为True
import cv2
o=cv2.imread("finger.png",-1)
cv2.imshow("original",o)
#==========================获取凸包================================###
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
hull=cv2.convexHull(contours[0])
image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
cv2.polylines(image,[hull],True,(0,255,0),2)
print(hull)  #测试边缘到底在哪里，然后再使用确定的位置点绘制文字
#=======================内部点A到轮廓的距离==========================###
distA=cv2.pointPolygonTest(hull,(300,150),True)
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image,"A",(300,150),font,1,(0,255,0),3)
print("disA=",distA)
#=======================外部点B到轮廓的距离==========================###
distB=cv2.pointPolygonTest(hull,(50,166),True)
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image,"B",(50,166),font,1,(0,255,0),3)
print("disB=",distB)
#==================正好处于轮廓上的点C到轮廓的距离======================###
distC=cv2.pointPolygonTest(hull,(417,165),True)
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image,"C",(417,165),font,1,(0,255,0),3)
print("disC=",distC)
#=================================显示=============================###
cv2.imshow("result1",image)
cv2.waitKey()
cv2.destroyAllWindows()

#12.6 利用形状场景算法比较轮廓
#12.6.1 计算形状场景距离
#使用cv2.createShapeContextDistanceExtractor() 计算形状场景距离
import cv2
#===========================原始图像o1的边缘=======================###
o1=cv2.imread("contours6_down.jpg",-1)
cv2.imshow("original1",o1)
gray1=cv2.cvtColor(o1,cv2.COLOR_BGR2GRAY)
ret,binary1=cv2.threshold(gray1,127,255,cv2.THRESH_BINARY)
image,contours1,hierarchy=cv2.findContours(binary1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnt1=contours1[0]
#===========================原始图像o2的边缘=======================###
o2=cv2.imread("contours6x.jpg",-1)
cv2.imshow("original2",o2)
gray2=cv2.cvtColor(o2,cv2.COLOR_BGR2GRAY)
ret,binary2=cv2.threshold(gray2,127,255,cv2.THRESH_BINARY)
image,contours2,hierarchy=cv2.findContours(binary2,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnt2=contours2[0]
#===========================原始图像o3的边缘=======================###
o3=cv2.imread("finger.png",-1)
cv2.imshow("origina3",o3)
gray3=cv2.cvtColor(o3,cv2.COLOR_BGR2GRAY)
ret,binary3=cv2.threshold(gray3,127,255,cv2.THRESH_BINARY)
image,contours3,hierarchy=cv2.findContours(binary3,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnt3=contours3[0]
#=========================构建距离提取算子=========================###
sd=cv2.createShapeContextDistanceExtractor()
#==========================计算距离================================###
d1=sd.computeDistance(cnt1,cnt1)
print("与自身的距离d1=",d1)
d2=sd.computeDistance(cnt2,cnt1)
print("与旋转后的自身图像的距离d2=",d2)
d3=sd.computeDistance(cnt2,cnt3)
print("与不相似对象的距离d3=",d3)
cv2.waitKey()
cv2.destroyAllWindows()

#12.6.2 计算 Hausdorff距离
#使用函数cv2.createHausdorffDistanceExtractor()计算不同图像的Hausdorff距离
import cv2
#=========================读取原始图像====================================###
o1=cv2.imread("contours6_down.jpg",-1)
o2=cv2.imread("contours6x.jpg",-1)
o3=cv2.imread("finger.png",-1)
cv2.imshow("original1",o1)
cv2.imshow("original2",o2)
cv2.imshow("original3",o3)
#=========================色彩转换========================================###
gray1=cv2.cvtColor(o1,cv2.COLOR_BGR2GRAY)
gray2=cv2.cvtColor(o2,cv2.COLOR_BGR2GRAY)
gray3=cv2.cvtColor(o3,cv2.COLOR_BGR2GRAY)
#=========================阈值处理========================================###
ret,binary1=cv2.threshold(gray1,127,255,cv2.THRESH_BINARY)
ret,binary2=cv2.threshold(gray2,127,255,cv2.THRESH_BINARY)
ret,binary3=cv2.threshold(gray3,127,255,cv2.THRESH_BINARY)
#=========================提取轮廓========================================###
image,contours1,hierarchy=cv2.findContours(binary1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
image,contours2,hierarchy=cv2.findContours(binary2,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
image,contours3,hierarchy=cv2.findContours(binary3,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnt1=contours1[0]
cnt2=contours2[0]
cnt3=contours3[0]
#=====================构造距离提取算子===================================###
hd=cv2.createHausdorffDistanceExtractor()
#=========================计算距离=======================================###
d1=hd.computeDistance(cnt1,cnt1)
print("与自身的Hausdorff距离d1=",d1)
d2=hd.computeDistance(cnt2,cnt1)
print("与旋转后的自身图像的Hausdorff距离d2=",d2)
d3=hd.computeDistance(cnt2,cnt3)
print("与不相似对象的Hausdorff距离d3=",d3)
cv2.waitKey()
cv2.destroyAllWindows()

#12.7 轮廓的特征值
#12.7.1 宽高比 AspectRation
import cv2
o=cv2.imread("finger.png",-1)
cv2.imshow("original",o)
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
x,y,w,h=cv2.boundingRect(contours[0])
cv2.rectangle(o,(x,y),(x+w,y+h),(255,255,255),3)
aspectRatio=float(w)/h
print(aspectRatio)
cv2.imshow("result",o)  
cv2.waitKey()
cv2.destroyAllWindows()

#12.7.2 Extent
#计算图像的轮廓面积与其矩形边界面积之比
import cv2
o=cv2.imread("finger.png",-1)
cv2.imshow("original",o)
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
x,y,w,h=cv2.boundingRect(contours[0])
cv2.drawContours(o,contours[0],-1,(0,0,255),3)
cv2.rectangle(o,(x,y),(x+w,y+h),(255,255,255),3)
rectArea=w*h
cntArea=cv2.contourArea(contours[0])
extend=float(cntArea)/rectArea
print(extend)
cv2.imshow("result",o)
cv2.waitKey()
cv2.destroyAllWindows()

#12.7.3 Solidity
#计算图像轮廓面积与凸包面积之比
import cv2
o=cv2.imread("finger.png",-1)
cv2.imshow("original",o)
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(o,contours[0],-1,(0,0,255),3)
cntArea=cv2.contourArea(contours[0])
hull=cv2.convexHull(contours[0])
hullArea=cv2.contourArea(hull)
cv2.polylines(o,[hull],True,(0,255,0),2)
solidity=float(cntArea)/hullArea
print(solidity)
cv2.imshow("result",o)
cv2.waitKey()
cv2.destroyAllWindows()

#12.7.4 等效直径（Equivalent Diameter）
#计算与轮廓面积相等的圆形的直径，并绘制与该轮廓等面积的圆
import cv2
import numpy as np
o=cv2.imread("finger.png",-1)
cv2.imshow("original",o)
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(o,contours[0],-1,(0,0,255),3)
cntArea=cv2.contourArea(contours[0])
equiDiameter=np.sqrt(4*cntArea/np.pi)
print(equiDiameter)
cv2.circle(o,(200,200),int(equiDiameter/2),(0,0,255),3)
cv2.imshow("result",o)
cv2.waitKey()
cv2.destroyAllWindows()

#12.7.6 掩模和像素点
#使用OpenCV函数cv2.findNonZero()获取一个图像内的轮廓点的位置
import cv2
import numpy as np
#===========================读取原始图像=================================###
o=cv2.imread("finger.png",-1)
cv2.imshow("original",o)
#===========================获取轮廓=====================================###
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnt=contours[0]
#===========================绘制空心轮廓=================================###
mask1=np.zeros(gray.shape,np.uint8)
cv2.drawContours(mask1,[cnt],0,255,2)
pixelpoints1=cv2.findNonZero(mask1)
print("pixelpoints1.shape=",pixelpoints1.shape)
print("pixelpoints1=\n",pixelpoints1)
cv2.imshow("mask1",mask1)
#===========================绘制实心轮廓=================================###
mask2=np.zeros(gray.shape,np.uint8)
cv2.drawContours(mask2,[cnt],0,255,-1)
pixelpoints2=cv2.findNonZero(mask2)
print("pixelpoints2.shape=",pixelpoints2.shape)
print("pixelpoints2=\n",pixelpoints2)
cv2.imshow("mask2",mask2)
cv2.waitKey()
cv2.destroyAllWindows()

#12.7.7 最大值和最小值及它们的位置
#使用函数 cv2.minMaxLoc() 在图像内查找掩模指定区域内的最大值、最小值及其位置
import cv2
import numpy as np
o=cv2.imread("finger.png",-1)
cv2.imshow("original",o)
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnt=contours[0]
#=======================使用掩模获取感兴趣区域的最值====================###
mask=np.zeros(gray.shape,np.uint8)
mask=cv2.drawContours(mask,[cnt],-1,255,-1)
minVal,maxVal,minLoc,maxLoc=cv2.minMaxLoc(gray,mask=mask)
print("minVal=",minVal)
print("maxVal=",maxVal)
print("minLoc=",minLoc)
print("maxLoc=",maxLoc)
#=======================使用掩模获取感兴趣区域并显示=====================###
masko=np.zeros(o.shape,np.uint8)
masko=cv2.drawContours(masko,[cnt],-1,(255,255,255),-1)
loc=cv2.bitwise_and(o,masko)
cv2.imshow("mask",loc)
#显示灰度结果
#loc=cv2.bitwise_and(gray,masko)
#cv2.imshow("mask",loc)
cv2.waitKey()
cv2.destroyAllWindows()

#12.7.8 平均颜色及平均灰度
#使用函数cv2.mean()计算一个对象的平均灰度
import cv2
import numpy as np
o=cv2.imread("finger.png",-1)
cv2.imshow("original",o)
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnt=contours[0]
#=======================使用掩模获取感兴趣区域的均值===================###
mask=np.zeros(gray.shape,np.uint8) #构建mean所使用的掩模（必须是单通道的）
cv2.drawContours(mask,[cnt],0,(255,255,255),-1)
meanVal=cv2.mean(o,mask=mask)
print("meanVal=\n",meanVal)
#=======================使用掩模获取感兴趣区域并显示=====================###
masko=np.zeros(o.shape,np.uint8)
masko=cv2.drawContours(masko,[cnt],-1,(255,255,255),-1)
loc=cv2.bitwise_and(o,masko)
cv2.imshow("mask",loc)
cv2.waitKey()
cv2.destroyAllWindows()

#12.7.9 极点
#计算一副图像内的极值点
import cv2
import numpy as np
o=cv2.imread("finger.png",-1)
cv2.imshow("original",o)
#======================获取并绘制轮廓============================###
gray=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
image,contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnt=contours[0]
mask=np.zeros(gray.shape,np.uint8)
cv2.drawContours(mask,[cnt],0,(255,255,255),-1)
#======================计算极值======================================###
leftmost=tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost=tuple(cnt[cnt[:,:,0].argmax()][0])
topmost=tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost=tuple(cnt[cnt[:,:,1].argmax()][0])
#======================打印极值=======================================###
print("leftmost=",leftmost)
print("rightmost=",rightmost)
print("topmost=",topmost)
print("bottommost=",bottommost)
#====================绘制文字说明====================================###
font=cv2.FONT_HERSHEY_COMPLEX
cv2.putText(o,"A",leftmost,font,1,(0,0,255),2)
cv2.putText(o,"B",rightmost,font,1,(0,0,255),2)
cv2.putText(o,"C",topmost,font,1,(0,0,255),2)
cv2.putText(o,"D",bottommost,font,1,(0,0,255),2)
#==================绘制图像=======================================###
cv2.imshow("result",o)
cv2.waitKey()
cv2.destroyAllWindows()


##############################################################################
########################chr 13 直方图处理######################################
##############################################################################
#13.2 绘制直方图
#13.2.1 使用Numpy绘制直方图
#使用hist()函数绘制一幅图像的直方图
import cv2
import matplotlib.pyplot as plt
o=cv2.imread("lena.png",-1)
cv2.imshow("original",o)
plt.hist(o.ravel(),256)
cv2.waitKey()
cv2.destroyAllWindows()

import cv2
import matplotlib.pyplot as plt
o=cv2.imread("lena.png",-1)
plt.hist(o.ravel(),16)

#13.2.2 使用OpenCV绘制直方图
#使用cv2.calcHist()计算一幅图的直方图结果，并观察所得到的统计直方图信息
import cv2
import numpy as np
img=cv2.imread("lena.png",0)
hist=cv2.calcHist([img],[0],None,[16],[0,255])
print(type(hist))
print(hist.shape)
print(hist.size)
print(hist)
#使用plot()函数将两组不同的值a=[0.3,0.4,2,5,3,4.5,4],b=[3,5,1,2,1,5,3]以不同的颜色绘制出来
import matplotlib.pyplot as plt
a=[0.3,0.4,2,5,3,4.5,4]
b=[3,5,1,2,1,5,3]
plt.plot(a,color="r")
plt.plot(b,color="g")
#使用函数plot()将函数cv2.calcHist()的返回值绘制为直方图
import cv2
import matplotlib.pyplot as plt
img=cv2.imread("lena.png",0)
hist=cv2.calcHist([img],[0],None,[255],[0,255])
plt.plot(hist,color="b")
plt.show()
#使用函数plot()和函数cv2.calcHist()，将彩色图像各个通道的直方图绘制在一个窗口内
import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("lena.png",-1)
histb=cv2.calcHist([img],[0],None,[255],[0,255])
histg=cv2.calcHist([img],[1],None,[255],[0,255])
histr=cv2.calcHist([img],[2],None,[255],[0,255])
plt.plot(histb,color="b")
plt.plot(histg,color="g")
plt.plot(histr,color="r")
plt.show()

#13.2.3 使用掩模绘制直方图
import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("lena.png",0)
mask=np.zeros(img.shape,np.uint8)
mask[200:400,200:400]=255
histImg=cv2.calcHist([img],[0],None,[255],[0,255])
histMask=cv2.calcHist([img],[0],mask,[255],[0,255])
plt.plot(histImg,color="r")
plt.plot(histMask,color="g")
plt.show()

#13.3 直方图均衡化
#使用函数cv2.equalizeHist()实现直方图均横化
import cv2
import matplotlib.pyplot as plt
img=cv2.imread("lena.png",0)
#=============================直方图均衡化处理=========================###
equ=cv2.equalizeHist(img)
#===========================显示均衡化前后的图像=======================###
cv2.imshow("original",img)
cv2.imshow("result",equ)
#===========================显示均衡化前后的直方图======================###
plt.figure("原始图像直方图")   #构建窗口
plt.hist(img.ravel(),256)
plt.figure("均衡化结果直方图")  #构建新窗口
plt.hist(equ.ravel(),256)
#===========================等待释放窗口==============================###
cv2.waitKey()
cv2.destroyAllWindows()

#13.4 pyplot模块介绍
#13.4.1 subplot函数 用来向当前窗口内添加一个子窗口对象
import cv2
import matplotlib.pyplot as plt
img=cv2.imread("lena.png",0)
equ=cv2.equalizeHist(img)
plt.figure("subplot 实例")
plt.subplot(121),plt.hist(img.ravel(),256)
plt.subplot(122),plt.hist(equ.ravel(),256)

#13.4.2 imshow函数显示彩色图像
import cv2
import matplotlib.pyplot as plt
img=cv2.imread("lena.png",-1)
imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.figure("显示结果")
plt.subplot(121),plt.imshow(img),plt.axis("off")
plt.subplot(122),plt.imshow(imgRGB),plt.axis("off")
#使用函数 matplotlib.pyplot.imshow()以不同的参数显示灰度图像
import cv2
import matplotlib.pyplot as plt
img=cv2.imread("lena.png",-1)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.figure("灰度图像显示演示")
plt.subplot(221);plt.imshow(gray,cmap=plt.cm.gray)
plt.subplot(222);plt.imshow(gray,cmap=plt.cm.gray_r)
plt.subplot(223);plt.imshow(gray,cmap="gray")
plt.subplot(224);plt.imshow(gray,cmap="gray_r")



##############################################################################
########################chr 14 傅里叶变换######################################
##############################################################################
#14.2 Numpy实现傅里叶变换
#14.2.1 实现傅里叶变换
#利用函数numpy.fft.fft2()实现傅里叶变换，然后numpy.fft.fftshift()将零频率成分移到中间
import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("lena.png",0)
f=np.fft.fft2(img)
fshift=np.fft.fftshift(f)
magnitude_spectrum=20*np.log(np.abs(fshift))
plt.subplot(121)
plt.imshow(img,cmap="gray")
plt.title("original")
plt.axis("off")
plt.subplot(122)
plt.imshow(magnitude_spectrum,cmap="gray")
plt.title("result")
plt.axis("off")
plt.show()

#14.2.2 实现逆傅里叶变换
#numpy.fft.ifft2可以实现逆傅里叶变换
import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("lena.png",0)
f=np.fft.fft2(img)
fshift=np.fft.fftshift(f)
ishift=np.fft.ifftshift(fshift)
iimg=np.fft.ifft2(ishift)
print(iimg)
iimg=np.abs(iimg)
print(iimg)
plt.subplot(121)
plt.imshow(img,cmap="gray")
plt.title("original"),plt.axis("off")
plt.subplot(122)
plt.imshow(iimg,cmap="gray")
plt.title("result"),plt.axis("off")
plt.show()

#14.2.3 高通滤波示例
import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("lena.png",0)
f=np.fft.fft2(img)
fshift=np.fft.fftshift(f)
rows,cols=img.shape
crow,ccol=int(rows/2),int(cols/2)
fshift[crow-30:crow+30,ccol-30:ccol+30]=0
ishift=np.fft.ifftshift(fshift)
iimg=np.fft.ifft2(ishift)
iimg=np.abs(iimg)
plt.subplot(121)
plt.imshow(img,cmap="gray")
plt.title("original"),plt.axis("off")
plt.subplot(122)
plt.imshow(iimg,cmap="gray")
plt.title("result"),plt.axis("off")
plt.show()

#14.3 OpenCV实现傅里叶变换
#14.3.1 利用函数cv2.dft()实现傅里叶变换
import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("lena.png",0)
dft=cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
dftshift=np.fft.fftshift(dft)
result=20*np.log(cv2.magnitude(dftshift[:,:,0],dftshift[:,:,1]))
plt.subplot(121)
plt.imshow(img,cmap="gray")
plt.title("original"),plt.axis("off")
plt.subplot(122)
plt.imshow(result,cmap="gray")
plt.title("result"),plt.axis("off")
plt.show()
#14.3.2 利用函数cv2.idft()实现逆傅里叶变换
import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("lena.png",0)
dft=cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
dftshift=np.fft.fftshift(dft)
ishift=np.fft.ifftshift(dftshift)
iImg=cv2.idft(ishift)
iImg=cv2.magnitude(iImg[:,:,0],iImg[:,:,1])
plt.subplot(121)
plt.imshow(img,cmap="gray")
plt.title("original"),plt.axis("off")
plt.subplot(122)
plt.imshow(iImg,cmap="gray")
plt.title("result"),plt.axis("off")
plt.show()
#14.3.3 低通滤波
import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("lena.png",0)
dft=cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
dftshift=np.fft.fftshift(dft)
rows,cols=img.shape
crow,ccol=int(rows/2),int(cols/2)
mask=np.zeros((rows,cols,2),np.uint8)
#两个通道，与频率图像匹配
mask[crow-30:crow+30,ccol-30:ccol+30]=1
fshift=dftshift*mask
ishift=np.fft.ifftshift(fshift)
iImg=cv2.idft(ishift)
iImg=cv2.magnitude(iImg[:,:,0],iImg[:,:,1])
plt.subplot(121)
plt.imshow(img,cmap="gray")
plt.title("original"),plt.axis("off")
plt.subplot(122)
plt.imshow(iImg,cmap="gray")
plt.title("inverse"),plt.axis("off")
plt.show()


##############################################################################
########################chr 15 模板匹配######################################
##############################################################################
#15.1 模板匹配基础
#使用函数cv2.matchTemplate()进行模板匹配，method=cv2.TM_SQDIFF
import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread("lena.jpg",0)
template=cv2.imread("lena_eyes.jpg",0)
th,tw=template.shape[::]
rv=cv2.matchTemplate(img,template,cv2.TM_SQDIFF)
minVal,maxVal,minLoc,maxLoc=cv2.minMaxLoc(rv)
topleft=minLoc      #值最小匹配度最好
bottomRight=(topleft[0]+tw,topleft[1]+th)
cv2.rectangle(img,topleft,bottomRight,255,2)
plt.subplot(121)
plt.imshow(rv,cmap="gray")
plt.title("Matching Result"),plt.xticks([]),plt.yticks([])
plt.subplot(122)
plt.imshow(img,cmap="gray")
plt.title("Detected Point"),plt.xticks([]),plt.yticks([])
#使用函数cv2.matchTemplate()进行模板匹配，method=cv2.TM_CCOEFF
import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread("lena.jpg",0)
template=cv2.imread("lena_eyes.jpg",0)
th,tw=template.shape[::]
rv=cv2.matchTemplate(img,template,cv2.TM_CCOEFF)
minVal,maxVal,minLoc,maxLoc=cv2.minMaxLoc(rv)
topleft=maxLoc       #值最大匹配度最好
bottomRight=(topleft[0]+tw,topleft[1]+th)
cv2.rectangle(img,topleft,bottomRight,255,2)
plt.subplot(121)
plt.imshow(rv,cmap="gray")
plt.title("Match Result")
plt.xticks([]),plt.yticks([])
plt.subplot(122)
plt.imshow(img,cmap="gray")
plt.title("Detected Point")
plt.xticks([]),plt.yticks([])


#15.2 多模板匹配
import numpy as np
am=np.array([[3,6,8,77,66],[12,88,3,9,8],[11,2,67,5,2]])
print(am)
b=np.where(am>5)
for i in zip(*b):
    print(i)
print(am[::-1])  #实现行列位置的互换
#使用模板匹配方式，标记在输入图像内匹配的多个子集图像
import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread("lena_muti.jpg",0)
template=cv2.imread("lena_eyes.jpg",0)
w,h=template.shape[::-1]
res=cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
threshold=0.9
loc=np.where(res>=threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img,pt,(pt[0]+w,pt[1]+h),255,1)
plt.imshow(img,cmap="gray")
plt.xticks([]),plt.yticks([])


##############################################################################
########################chr 16 霍夫变换########################################
##############################################################################
#16.1 霍夫直线变换
#16.1.2 HoughLines函数
import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread("lines.jpg",-1)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges=cv2.Canny(gray,50,150,apertureSize=3)
orgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
oshow=orgb.copy()
lines=cv2.HoughLines(edges,1,np.pi/180,140)
for line in lines:
    rho,theta=line[0]
    a=np.cos(theta)
    b=np.sin(theta)
    x0=a*rho
    y0=b*rho
    x1=int(x0+1000*(-b))
    y1=int(x0+1000*(a))
    x2=int(x0-1000*(-b))
    y2=int(x0-1000*(a))
    cv2.line(orgb,(x1,y1),(x2,y2),(0,0,255),2)
plt.subplot(121)
plt.imshow(oshow)
plt.axis("off")
plt.subplot(122)
plt.imshow(orgb)
plt.axis("off")
#16.1.3 概率霍夫变换 HoughLinesP函数
import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread("lines.jpg",-1)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges=cv2.Canny(gray,50,150,apertureSize=3)
orgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
oshow=orgb.copy()
lines=cv2.HoughLinesP(edges,1,np.pi/180,1,minLineLength=100,maxLineGap=10)
for line in lines:
    x1,y1,x2,y2=line[0]
    cv2.line(orgb,(x1,y1),(x2,y2),(255,0,0),5)
plt.subplot(121)
plt.imshow(oshow)
plt.axis("off")
plt.subplot(122)
plt.imshow(orgb)
plt.axis("off")

#16.2 霍夫圆环变换
#使用HoughLinesCircles()函数对一幅图像进行霍夫圆变换
import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread("circles.jpg",0)
imgo=cv2.imread("circles.jpg",-1)
o=cv2.cvtColor(imgo,cv2.COLOR_BGR2RGB)
oshow=o.copy()
img=cv2.medianBlur(img,5)
circles=cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,300,param1=50,param2=30,minRadius=50,maxRadius=100)
for i in circles[0,:]:
    cv2.circle(o,(i[0],i[1]),i[2],(0,0,255),12)
    cv2.circle(o,(i[0],i[1]),2,(0,0,255),12)
plt.subplot(121)
plt.imshow(oshow)
plt.axis("off")
plt.subplot(122)
plt.imshow(o)
plt.axis("off")


##############################################################################
########################chr 17 图像分割与提取##################################
##############################################################################
#17.1 用分水岭算法实现图像分割与提取
#使用形态学变换获取一幅图像的边界信息
import cv2
import numpy as np
from matplotlib import pyplot as plt
o=cv2.imread("border.jpg",-1)
k=np.ones((5,5),np.uint8)
e=cv2.erode(o,k)
b=cv2.subtract(o,e)
plt.subplot(131)
plt.imshow(o)
plt.axis("off")
plt.subplot(132)
plt.imshow(e)
plt.axis("off")
plt.subplot(133)
plt.imshow(b)
plt.axis("off")
cv2.imshow("original",o)
cv2.imshow("erode",e)
cv2.imshow("border",b)
cv2.waitKey()
cv2.destroyAllWindows()
#使用距离变换函数cv2.distanceTransform()计算衣服图像的确定前景
import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread("cells.jpg",-1)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
ishow=img.copy()
ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
kernel=np.ones((3,3),np.uint8)
opening=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)
dis_transform=cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret,fore=cv2.threshold(dis_transform,0.4*dis_transform.max(),255,0)
plt.subplot(131)
plt.imshow(ishow)
plt.axis("off")
plt.subplot(132)
plt.imshow(dis_transform)
plt.axis("off")
plt.subplot(133)
plt.imshow(fore)
plt.axis("off")
cv2.imshow("original",ishow)
cv2.imshow("dis_transform",dis_transform)
cv2.imshow("fore",fore)
cv2.waitKey()
cv2.destroyAllWindows()
#确定未知区域（原始图像-确定背景-确定前景）
import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread("cells.jpg",-1)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
ishow=img.copy()
ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
kernel=np.ones((3,3),np.uint8)
opening=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)
bg=cv2.dilate(opening,kernel,iterations=3)
dist=cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret,fore=cv2.threshold(dist,0.4*dist.max(),255,0)
fore=np.uint8(fore)
un=cv2.subtract(bg,fore)
plt.subplot(221)
plt.imshow(ishow)
plt.axis("off")
plt.subplot(222)
plt.imshow(bg)
plt.axis("off")
plt.subplot(223)
plt.imshow(fore)
plt.axis("off")
plt.subplot(224)
plt.imshow(un)
plt.axis("off")
cv2.imshow("original",ishow)
cv2.imshow("bg",bg)
cv2.imshow("fore",fore)
cv2.imshow("unknow",un)
cv2.waitKey()
cv2.destroyAllWindows()
#使用函数cv2.connectedComponents()标注一幅图像
import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread("cells.jpg",-1)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
ishow=img.copy()
ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
kernel=np.ones((3,3),np.uint8)
opening=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)
dist_transform=cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret,fore=cv2.threshold(dist_transform,0.4*dist_transform.max(),255,0)
fore=np.uint8(fore)
ret,marker=cv2.connectedComponents(fore)
plt.subplot(131)
plt.imshow(ishow)
plt.axis("off")
plt.subplot(132)
plt.imshow(fore)
plt.axis("off")
plt.subplot(133)
plt.imshow(marker)
plt.axis("off")
cv2.imshow("original",ishow)
cv2.imshow("fore",fore)
cv2.imshow("marker",marker)
cv2.waitKey()
cv2.destroyAllWindows()
#使用函数cv2.connectedComponents()标注一幅图像，并对其进行修正，使未知区域被标注为0
import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread("cells.jpg",-1)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
ishow=img.copy()
ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
kernel=np.ones((3,3),np.uint8)
opening=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)
bg=cv2.dilate(opening,kernel,iterations=3)
dist_transform=cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret,fore=cv2.threshold(dist_transform,0.4*dist_transform.max(),255,0)
fore=np.uint8(fore)
un=cv2.subtract(bg,fore)
ret,marker1=cv2.connectedComponents(fore)
ret,marker2=cv2.connectedComponents(fore)
marker2=marker2+1
marker[un==255]=0                                    
plt.subplot(121)
plt.imshow(marker1)
plt.axis("off")
plt.subplot(122)
plt.imshow(marker1)
plt.axis("off")
#使用函数cv2.watershed()分水岭算法对一幅图像进行分割
import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread("cells.jpg",-1)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
ishow=img.copy()
ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
kernel=np.ones((3,3),np.uint8)
opening=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)
bg=cv2.dilate(opening,kernel,iterations=3)
dist_transform=cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret,fore=cv2.threshold(dist_transform,0.4*dist_transform.max(),255,0)
fore=np.uint8(fore)
un=cv2.subtract(bg,fore)
ret,markers=cv2.connectedComponents(fore)
markers=markers+1
markers[un==255]=0
markers=cv2.watershed(img,markers)
img[markers==-1]=[0,255,0]
plt.subplot(121)
plt.imshow(ishow)
plt.axis("off")
plt.subplot(122)
plt.imshow(img)
plt.axis("off")

#17.2 交互式前景提取
#在GrabCut算法中使用模板提取图像的前景，并观察提取效果
import cv2
import numpy as np
from matplotlib import pyplot as plt
o=cv2.imread("lena.png",-1)
orgb=cv2.cvtColor(o,cv2.COLOR_BGR2RGB)
mask=np.zeros(o.shape[:2],np.uint8)
bdg=np.zeros((1,65),np.float64)
fdg=np.zeros((1,65),np.float64)
rect=(50,50,400,500)
cv2.grabCut(o,mask,rect,bdg,fdg,5,cv2.GC_INIT_WITH_RECT)
mask2=cv2.imread("lena_mask1.png",0)
mask2show=cv2.imread("lena_mask1.png",-1)
m2rgb=cv2.cvtColor(mask2show,cv2.COLOR_BGR2RGB)
mask[mask2==0]=0
mask[mask2==255]=1
mask,bgd,fgd=cv2.grabCut(o,mask,None,bdg,fdg,5,cv2.GC_INIT_WITH_MASK)
mask=np.where((mask==2)|(mask==0),0,1).astype("uint8")
ogc=o*mask[:,:,np.newaxis]
ogc=cv2.cvtColor(ogc,cv2.COLOR_BGR2RGB)
plt.subplot(121)
plt.imshow(m2rgb)
plt.axis("off")
plt.subplot(122)
plt.imshow(ogc)
plt.axis("off")
#在GrabCut算法中直接使用自定义模板提取图像前景
import cv2
import numpy as np
from matplotlib import pyplot as plt
o=cv2.imread("lena.png",-1)
orgb=cv2.cvtColor(o,cv2.COLOR_BGR2RGB)
bdg=np.zeros((1,65),np.float64)
fdg=np.zeros((1,65),np.float64)
mask2=np.zeros(o.shape[:2],np.uint8)
#先将掩模的值全部构造为0（确定背景），在后续步骤中，再根据需要修改其中部分值
mask2[15:512,20:270]=3   #lena头像的可能区域
mask2[25:300,100:150]=1  #lena头像的确定区域，如果不设置这个区域，头像的提取不完整
cv2.grabCut(o,mask2,None,bgd,fgd,5,cv2.GC_INIT_WITH_MASK)
mask2=np.where((mask2==2)|(mask2==0),0,1).astype("uint8")
ogc=o*mask2[:,:,np.newaxis]
ogc=cv2.cvtColor(ogc,cv2.COLOR_BGR2RGB)
plt.subplot(121)
plt.imshow(orgb)
plt.axis("off")
plt.subplot(122)
plt.imshow(ogc)
plt.axis("off")
cv2.imshow("original",o)
cv2.imshow("result",ogc)
cv2.waitKey()
cv2.destroyAllWindows()


##############################################################################
########################chr 19 绘图及交互######################################
##############################################################################
#19.1.1 绘制直线
import cv2
import numpy as np
n=300
img=np.zeros((n+1,n+1,3),np.uint8)
img=cv2.line(img,(0,0),(n,n),(255,0,0),3)
img=cv2.line(img,(0,100),(n,100),(0,255,0),1)
img=cv2.line(img,(100,0),(100,n),(0,0,255),6)
winname="Demo19.1.1"
cv2.namedWindow(winname)
cv2.imshow(winname,img)
cv2.waitKey()
cv2.destroyAllWindows()

#19.1.2 绘制矩形
import cv2
import numpy as np
n=300
img=np.zeros((n+1,n+1,3),np.uint8)*255
img=cv2.rectangle(img,(50,50),(n-100,n-50),(0,0,255),-1)
winname="Demo19.1.2"
cv2.namedWindow(winname)
cv2.imshow(winname,img)
cv2.waitKey()
cv2.destroyAllWindows()

#19.1.3 绘制圆形
import cv2
import numpy as np
d=400
img=np.zeros((d,d,3),np.uint8)
(centerX,centerY)=(round(img.shape[1]/2),round(img.shape[0]/2))
#将图像中心作为圆心，实际值为d/2
red=[0,0,255]   #设置白色变量
for r in range(5,round(d/2),12):
    cv2.circle(img,(centerX,centerY),r,red,3)
    #circle(载体图像，圆心，半径，颜色)
winname="Demo19.1.3.1"
cv2.namedWindow(winname)
cv2.imshow(winname,img)
cv2.waitKey()
cv2.destroyAllWindows()
#使用函数cv2.circle()在一个白色背景图像内绘制一组位置和大小均随机的实心圆
import cv2
import numpy as np
d=400
img=np.ones((d,d,3),np.uint8)*255
#生成白色背景
for i in range(0,100):
    centerX=np.random.randint(0,high=d)   #生成随机圆心centerX，确保在画布img内
    centerY=np.random.randint(0,high=d)   #生成随机圆心centerY，确保在画布img内
    radius=np.random.randint(5,high=d/5)  #生成随机半径，值范围为[5,d/5],最大半径为d/5
    color=np.random.randint(0,256,size=(3,)).tolist() #生成随机颜色，3个[0,256)的随机数
    cv2.circle(img,(centerX,centerY),radius,color,-1) #使用上述随机数在画布img内画圆
winname="Demo19.1.3.2"
cv2.namedWindow(winname)
cv2.imshow(winname,img)
cv2.waitKey()
cv2.destroyAllWindows()

#19.1.4 绘制椭圆
#使用cv2.ellipse()在白色北京图像内随机绘制一组空心椭圆
import cv2
import numpy as np
d=400
img=np.ones((d,d,3),np.uint8)*255
#生成白色背景
center=(round(d/2),round(d/2))
#注意数值类型，不可以使用语句center=(d/2,d/2)
size=(100,200) #轴的长度
for i in range(0,10):
    angle=np.random.randint(0,361) #偏移角度
    color=np.random.randint(0,256,size=(3,)).tolist() #生成随机颜色，3个[0,256)的随机数
    thickness=np.random.randint(1,9)
    cv2.ellipse(img,center,size,angle,0,360,color,thickness)
winname="Demo19.1.4"
cv2.namedWindow(winname)
cv2.imshow(winname,img)
cv2.waitKey()
cv2.destroyAllWindows()

#19.1.5 绘制多边形
#使用函数cv2.polylines()在一个白色背景图像内绘制一个多边形
import cv2
import numpy as np
d=400
img=np.ones((d,d,3),np.uint8)*255
#生成白色背景
pts=np.array([[200,50],[300,200],[200,350],[100,200]],np.int32)
#生成各个顶点，注意数据类型为 int32
pts=pts.reshape((-1,1,2))
#第一个参数为-1，表明它未设置具体值，它所表示的维度值是通过其他参数计算得到的
cv2.polylines(img,[pts],True,(0,255,0),8)
winname="Demo19.1.5"
cv2.namedWindow(winname)
cv2.imshow(winname,img)
cv2.waitKey()
cv2.destroyAllWindows()

#19.1.6在图像上绘制文字
#使用函数cv2.putText()在一个白色背景图像内绘制一段镜像的文字
import cv2
import numpy as np
d=400
img=np.ones((d,d,3),np.uint8)*255
#生成白色背景
font=cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img,"OpenCV",(0,150),font,3,(0,255,0),15)
cv2.putText(img,"OpenCV",(0,150),font,3,(0,255,255),5)
cv2.putText(img,"OpenCV",(0,250),font,3,(255,0,0),15,cv2.FONT_HERSHEY_SCRIPT_COMPLEX,True)
winname="Demo19.1.6"
cv2.namedWindow(winname)
cv2.imshow(winname,img)
cv2.waitKey()
cv2.destroyAllWindows()

#19.2 鼠标交互
#设计一个程序，对触发的鼠标事件进行判断
import cv2
import numpy as np
def Demo(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print("单击了鼠标左键")
    if event==cv2.EVENT_RBUTTONDOWN:
        print("单击了鼠标右键")
    elif flags==cv2.EVENT_FLAG_LBUTTON:
        print("按住左键拖动了鼠标")
    elif event==cv2.EVENT_MBUTTONDOWN:
        print("单击了中间键")
#创建名称为Demo的响应（回调）函数 onMouseAction
#将响应的函数Demo与窗口“Demo19.2”建立联系（实现绑定）
img=np.ones((300,300,3),np.uint8)*255        
cv2.namedWindow("Demo19.2.1")
cv2.setMouseCallback("Demo19.2.1",Demo)
cv2.imshow("Demo19.2.1",img)
cv2.waitKey()
cv2.destroyAllWindows()
#设计一个程序，当双击鼠标后，以当前位置为顶点绘制大小随机、颜色随机的矩形
import cv2
import numpy as np
d=400
def draw(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDBLCLK:
        p1x=x
        p1y=y
        p2x=np.random.randint(1,d-50)
        p2y=np.random.randint(1,d-50)
        color=np.random.randint(0,high=256,size=(3,)).tolist()
        cv2.rectangle(img,(p1x,p1y),(p2x,p2y),color,2)
img=np.ones((d,d,3),np.uint8)*255
cv2.namedWindow("Demo19.2.2")
cv2.setMouseCallback("Demo19.2.2",draw)
while(1):
    cv2.imshow("Demo19.2.2",img)
    if cv2.waitKey(20)==27:
        break
cv2.destroyAllWindows()
#进阶示例，设计一个交互程序，通过键盘与鼠标的组合控制显示不同的形状或文字
import cv2
import numpy as np
thickness=-1
mode=1
d=400
def draw_circle(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        a=np.random.randint(1,d-50)
        r=np.random.randint(1,d/5)
        angle=np.random.randint(0,361)
        color=np.random.randint(0,256,size=(3,)).tolist()
        if mode==1:
            cv2.rectangle(img,(x,y),(a,a),color,thickness)
        elif mode==2:
            cv2.circle(img,(x,y),r,color,thickness)
        elif mode==3:
            cv2.line(img,(a,a),(x,y),color,3)
        elif mode==4:
            cv2.ellipse(img,(x,y),(100,150),angle,0,360,color,thickness)
        elif mode==5:
            cv2.putText(img,"OpenCV",(0,round(d/2)),cv2.FONT_HERSHEY_SIMPLEX,2,color,5)
img=np.ones((d,d,3),np.uint8)*255
cv2.namedWindow("Demo19.2.3")
cv2.setMouseCallback("Demo19.2.3",draw_circle)
while(1):
    cv2.imshow("Demo19.2.3",img)
    k=cv2.waitKey(1)&0xFF
    if k==ord("r"):
        mode=1
    elif k==ord("c"):
        mode=2
    elif k==ord("l"):
        mode=3
    elif k==ord("e"):
        mode=4
    elif k==ord("t"):
        mode=5
    elif k==ord("t"):
        thickness=-1
    elif k==ord("u"):
        thickness=-3
    elif k==27:
        break
cv2.destroyAllWindows()
             
#19.3 滚动条
#19.3.1 用滚动条实现调色板
#设计一个滚动条交互程序，通过滚动条模拟调色板效果
import cv2
import numpy as np         
def ChangeColor(x):
    r=cv2.getTrackbarPos("R","image")
    g=cv2.getTrackbarPos('G',"image")
    b=cv2.getTrackbarPos('B',"image")
    img[:]=[b,g,r]
img=np.zeros((100,700,3),np.uint8)
cv2.namedWindow("image")
cv2.createTrackbar("R","image",0,255,ChangeColor)
cv2.createTrackbar("G","image",0,255,ChangeColor)
cv2.createTrackbar("B","image",0,255,ChangeColor)
while(1):
    cv2.imshow("image",img)
    k=cv2.waitKey(1)&0xFF
    if k==27:
        break
cv2.destroyAllWindows()
#19.3.2 用滚动条控制阈值处理参数
#设计一个滚动条交互程序，通过滚动条控制函数cv2.threshold()中的阈值和模式
import cv2
Type=0   #阈值处理方式
Value=0  #使用的阈值
def onType(a):
    Type=cv2.getTrackbarPos(tType,windowName)
    Value=cv2.getTrackbarPos(tValue,windowName)
    ret,dst=cv2.threshold(o,Value,255,Type)
    cv2.imshow(windowName,dst)
def onValue(a):
    Type=cv2.getTrackbarPos(tType,windowName)
    Value=cv2.getTrackbarPos(tValue,windowName)
    ret,dst=cv2.threshold(o,Value,255,Type)
    cv2.imshow(windowName,dst)
o=cv2.imread("lena512color.tif",0)
windowName="Demo19.3.2"   #窗体名
cv2.namedWindow(windowName)
cv2.imshow(windowName,o)
#创建两个滚动条
tType="Type"   #用来选取阈值处理方式的滚动条
tValue="Value" #用来选取阈值得滚动条
cv2.createTrackbar(tType,windowName,0,4,onType)
cv2.createTrackbar(tValue,windowName,0,255,onValue)
if cv2.waitKey(0)==27:
    cv2.destroyAllWindows()
#19.3.3 用滚动条作为开关
#设计一个滚动交互程序，用滚动条控制绘制的矩形是实心的还是空心的
import cv2
import numpy as np
d=400
global thickness
thickness=-1
def fill(x):
    pass
def draw(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDBLCLK:
        p1x=x
        p1y=y
        p2x=np.random.randint(1,d-50)
        p2y=np.random.randint(1,d-50)
        color=np.random.randint(0,256,size=(3,)).tolist()
        cv2.rectangle(img,(p1x,p1y),(p2x,p2y),color,thickness)
img=np.ones((d,d,3),np.uint8)*255
cv2.namedWindow("image")
cv2.setMouseCallback("image",draw)
cv2.createTrackbar("R","image",0,1,fill)
while(1):
    cv2.imshow("image",img)
    k=cv2.waitKey(1)&0xFF
    g=cv2.getTrackbarPos("R","image")
    if g==0:
        thickness=-1
    else:
        thickness=2
    if k==27:
        break
cv2.destroyAllWindows()
                

##############################################################################
########################chr 20 K近邻算法######################################
##############################################################################
#20.4 演示OpenCV自带的K近邻算法模块使用方法
#使用OpenCV自带的K近邻模块判断生成的随即数对test是属于rand1所在类型0，还是属于rand2所在类型1
import cv2
import numpy as np
import matplotlib.pyplot as plt
#创建两组用于训练得数据，每组包含20对随机数（20个随机数据点）
rand1=np.random.randint(0,30,(20,2)).astype(np.float32)
rand2=np.random.randint(70,100,(20,2)).astype(np.float32)
#将rand1和rand2拼接为训练数据
trainData=np.vstack((rand1,rand2))
#接下来为两组随机数分配标签,共两类：0和1
r1Label=np.zeros((20,1)).astype(np.float32)
r2Label=np.ones((20,1)).astype(np.float32)
tdLabel=np.vstack((r1Label,r2Label))
#使用绿色标注类型0
g=trainData[tdLabel.ravel()==0]
plt.scatter(g[:,0],g[:,1],80,"g","o")
#使用蓝色标注类型1
b=trainData[tdLabel.ravel()==1]
plt.scatter(b[:,0],b[:,1],80,"b","s")
#然后，生成一对值在(0,100)内的随机数对test,为用于测试的随机数
test=np.random.randint(0,100,(1,2)).astype(np.float32)
plt.scatter(test[:,0],test[:,1],80,"r","*")
#调用OpenCV内的 K 近邻模块，并进行训练
knn=cv2.ml.KNearest_create()
knn.train(trainData,cv2.ml.ROW_SAMPLE,tdLabel)
#使用 K 邻近算法分类
ret,result,neighbours,dist=knn.findNearest(test,5)
#显示处理结果
print("当前随机数可以被判定为类型：",result)
print("距离当前点最近的5个邻居是：",neighbours)
print("5个最近邻居的距离：",dist)
plt.show()


##############################################################################
########################chr 21 支持向量机######################################
##############################################################################
#已知老员工的笔试成绩、面试成绩及对应的等级表现，根据新员工的笔试成绩和面试成绩预测其可能的表现
#=============================1. 生成模拟数据===============================###
import cv2
import numpy as np
import matplotlib.pyplot as plt
#首先生成20组入职一年后变现为A级的员工入职时笔试和面试成绩，成绩均分布在[95,100)区间的数据对
a=np.random.randint(95,100,(20,2)).astype(np.float32)
#然后生成20组入职一年后变现为B级的员工入职时笔试和面试成绩，成绩均分布在[90,95)区间的数据对
b=np.random.randint(90,95,(20,2)).astype(np.float32)
#将两组数据合并，并使用numpy.array对其进行类型转换：
data=np.vstack((a,b))
data=np.array(data,dtype="float32")
#============================2. 构造分组标签================================###
#首先对表现为A级的数据构造标签“0”：
alable=np.zeros((20,1))
#接下来对表现为B级的数据构造标签“1”：
blable=np.ones((20,1))
#将两组数据合并，并使用numpy.array对其进行类型转换：
label=np.vstack((alable,blable))
label=np.array(label,dtype="int32")
#===========================3. 训练=========================================###
svm=cv2.ml.SVM_create()
result=svm.train(data,cv2.ml.ROW_SAMPLE,label)
#============================4. 预测分类=====================================###
#生成测试数据
test=np.vstack([[98,90],[90,99]])
test=np.array(test,dtype="float32")
#然后使用函数svm.predict()对随机成绩分类
(p1,p2)=svm.predict(test)
#==========================5. 显示分类结果===================================###
#可视化
plt.scatter(a[:,0],a[:,1],80,"g","o")
plt.scatter(b[:,0],b[:,1],80,"b","s")
plt.scatter(test[:,0],test[:,1],80,"r","*")
plt.show()
#打印原始数据test,预测结果
print(test)
print(p2)


##############################################################################
########################chr 22 K均值聚类#######################################
##############################################################################
#22.3.1 随机生成一组数据，使用函数cv2.kmeans()对其分类
import cv2
import numpy as np
import matplotlib.pyplot as plt
#随机生成两组数据
#生成60个值在[0,50]内的xiaoMI直径数据
xiaoMI=np.random.randint(0,50,60)
#生成60个值在[200,250]内的daMI直径数据
daMI=np.random.randint(200,250,60)
#将xiaoMI和daMI组合为MI
MI=np.vstack((xiaoMI,daMI))
#使用reshape函数将其转换成(120,1)
MI=MI.reshape(120,1)
#将MI转换成float32类型
MI=np.float32(MI)
#调用kmeans模块
#设置参数criteria的值
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,0.1)
#设置参数flags的值
flags=cv2.KMEANS_RANDOM_CENTERS
#调用函数kmeans
retval,bestLabels,centers=cv2.kmeans(MI,2,None,criteria,10,flags)
#打印返回值
print(retval)
print(bestLabels)
print(centers)
#获取分类及如果
XM=MI[bestLabels==0]
DM=MI[bestLabels==1]
#绘制分类结果
#绘制原始数据
plt.plot(XM,"rs")
plt.plot(DM,"bo")
#绘制中心点
plt.plot(centers[0],"rx")
plt.plot(centers[1],"bx")
plt.show()

#22.3.2 有一堆米粒，按照长度和宽度对它们分类，XM的长和宽都在[0,20]内，DM的长和宽都在[40,60]内
import cv2
import numpy as np
import matplotlib.pyplot as plt
xiaomi=np.random.randint(0,20,(30,2))
dami=np.random.randint(40,60,(30,2))
MI=np.vstack((xiaomi,dami))
MI=np.float32(MI)
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,0.1)
flags=cv2.KMEANS_RANDOM_CENTERS
ret,label,center=cv2.kmeans(MI,2,None,criteria,10,flags)
#打印返回值
print(ret)
print(label)
print(center)
XM=MI[label.ravel()==0]
DM=MI[label.ravel()==1]
#绘制分类结果数据及中心点
plt.scatter(XM[:,0],XM[:,1],c="g",marker="s")
plt.scatter(DM[:,0],DM[:,1],c="r",marker="o")
plt.scatter(center[0,0],center[0,1],200,c="b",marker="o")
plt.scatter(center[1,0],center[1,1],200,c="b",marker="s")
plt.xlabel("Height"),plt.ylabel("Width")
plt.show()

#22.3.3 使用函数cv2.kmeans()将灰度图像处理为只有两个灰度级的二值图像
import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("lena.png",-1)
#使用reshape将一个像素点的RGB值作为一个单元处理
data=img.reshape((-1,3))
#转换成kmens可以处理的数据类型
data=np.float32(data)
#调用Kmeans模块
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
flags=cv2.KMEANS_RANDOM_CENTERS
ret,label,center=cv2.kmeans(data,2,None,criteria,10,flags)
#转换为uint8数据类型，将每个像素点都赋值为当前分类的中心点像素
center=np.uint8(center)
#使用center内的值替换原像素点的值
res1=center[label.flatten()]
#使用reshape调整替换后的图像
res2=res1.reshape((img.shape))
#显示处理结果
plt.subplot(121)
plt.imshow(img)
plt.axis("off")
plt.subplot(122)
plt.imshow(res2)
plt.axis("off")
cv2.imshow("original",img)
cv2.imshow("result",res2)
cv2.waitKey()
cv2.destroyAllWindows()


##############################################################################
########################chr 23 人脸识别########################################
##############################################################################
#23.1 Haar级联分类器
#使用函数cv2.CascadeClassifier.detectMultiScale()检测一幅图像内的人脸
import cv2
image=cv2.imread("face_recognize.jpg",-1)
#获取XML文件，加载人脸检测器
faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
print(faceCascade)
#色彩转换，转换成灰度图像
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#调用函数detectMultiScale
faces=faceCascade.detectMultiScale(gray,scaleFactor=1.15,minNeighbors=5,minSize=(5,5))
print(faces)
#打印输出的测试结果
print("发现{0}个人脸！".format(len(faces)))
#逐个标注人脸
for(x,y,w,h) in faces:
    #cv2.rectangle(image,(x,y),(x+w,y+w),(0,255,0),2)   #矩形标注
    cv2.circle(image,(int((x+x+w)/2),int((y+y+h)/2)),int(w/2),(0,255,0),2)
#显示结果
cv2.imshow("dect",image)
cv2.imwrite("re_1.jpg",image)
cv2.waitKey()
cv2.destroyAllWindows()

#23.2 LBPH人脸识别
#利用LBPH完成一个简单的人脸识别程序
import cv2
import numpy as np
images=[]
images.append(cv2.imread("X1.jpg",0))
images.append(cv2.imread("X2.jpg",0))
images.append(cv2.imread("W1.jpg",0))
images.append(cv2.imread("W2.jpg",0))
labels=[0,0,1,1]
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images,np.array(labels))
predict_image=cv2.imread("X3.jpg",0)
label,confidence=recognizer.predict(predict_image)
print("label",label)
print("confidence",confidence)

#23.3 EigenFaces人脸识别
#利用EigenFaces模块完成一个简单的人脸识别程序(图像大小必须相等)
import cv2
import numpy as np
import cv2
import numpy as np
images=[]
images.append(cv2.imread("Y1.jpg",0))
images.append(cv2.imread("Y2.jpg",0))
images.append(cv2.imread("Z1.jpg",0))
images.append(cv2.imread("Z2.jpg",0))
labels=[0,0,1,1]
recognizer=cv2.face.EigenFaceRecognizer_create()
recognizer.train(images,np.array(labels))
predict_image=cv2.imread("Z3.jpg",0)
label,confidence=recognizer.predict(predict_image)
print("label",label)
print("confidence",confidence)

#23.4 Fisherfaces人脸识别
import cv2
import numpy as np
import cv2
import numpy as np
images=[]
images.append(cv2.imread("Y1.jpg",0))
images.append(cv2.imread("Y2.jpg",0))
images.append(cv2.imread("Z1.jpg",0))
images.append(cv2.imread("Z2.jpg",0))
labels=[0,0,1,1]
recognizer=cv2.face.FisherFaceRecognizer_create()
recognizer.train(images,np.array(labels))
predict_image=cv2.imread("Y3.jpg",0)
label,confidence=recognizer.predict(predict_image)
print("label",label)
print("confidence",confidence)










