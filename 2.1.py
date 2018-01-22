# -*- coding: utf-8 -*-
import cv2
import bluetooth
import numpy as np
import time

bd_addr = "00:10:05:25:00:01"
port = 1

sock=bluetooth.BluetoothSocket( bluetooth.RFCOMM )
sock.connect((bd_addr, port))
sock.send("T")
print sock.recv(1024)
sock.send("T")
print sock.recv(1024)
def nothing(x):
    pass
cap = cv2.VideoCapture(2)
#cap1=cv2.VideoCapture(0)
#opencv2:
#   fourcc = cv2.cv.CV_FOURCC(*'XVID')
#opencv3的话用:
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))#保存视频
xi=[0,0,0,0]
yi=[0,0,0,0]
M1=0
M2=0
cnt=0
path=[]
ans_path=[]
ans=100000
p_x=[-1,1,0,0]
p_y=[0,0,-1,1]
note=np.zeros((5,5),int)
lower_green=np.array([63,100,70])
upper_green=np.array([86,255,255])
lower_red=np.array([165, 80, 100])
upper_red=np.array([180, 255, 255])
dirs = np.zeros((5, 5, 4), dtype=int)  # 0:up 1:down 2:left 3:right
cx_green=0
cy_green=0
cx_red=0
cy_red=0
data="FINISHED"
len_x = 300
len_y = 300
block_x = len_x / 5
block_y = len_y / 5
dx=[]
dy=[]
s_x = 0
s_y = 0
angle=3
totalStep=0
step_i=0
currentDir=0
fcnt=0
attachflag=False
clock=60
skipflag=False
def dfs(x1,y1,x2,y2,step):
    global ans,dirs,path,p_x,p_y,ans_path,note
    if x1==x2 and y1==y2:
        if step<ans:
            ans=step
            ans_path=list(path)
        return
    if ans<step:
        return
    if note[x1][y1]:
        return
    note[x1][y1]=1
    for i in range(4):
        if dirs[x1][y1][i]:
            path.append(((x1,y1),i))
            dfs(x1+p_x[i],y1+p_y[i],x2,y2,step+1)
            path.pop()
    note[x1][y1]=0
    return
def extract(event,x,y,flags,param):
    global xi,yi,cnt
    if cnt<4 and event==cv2.EVENT_LBUTTONDOWN:
        xi[cnt]=x
        yi[cnt]=y
        cnt+=1
        if cnt==4:
            return
def sendMsg(st):
    global sock
    sock.send(st)
    #time.sleep(4)
    if st=="A":
        print "AHEAD"
    elif st=="L":
        print "LEFT"
    elif st=='R':
        print "RIGHT"
    elif st=='B':
        print "BACK"
def sentCommand(mod,d):
    global sock
    flagA=False
    if mod==0:
        if d==0:
            sendMsg("A")
        elif d==3:
            sendMsg("L")
            flagA=True
        elif d==2:
            sendMsg("R")
            flagA=True
        else:
            sendMsg("H")
            flagA=True
    elif mod==1:
        if d==1:
            sendMsg("A")
        elif d==3:
            sendMsg("R")
            flagA=True
        elif d==2:
            sendMsg("L")
            flagA=True
        else:
            sendMsg("H")
            flagA=True
    elif mod==2:
        if d==2:
            sendMsg("A")
        elif d==0:
            sendMsg("L")
            flagA=True
        elif d==1:
            sendMsg("R")
            flagA=True
        else:
            sendMsg("H")
            flagA=True
    elif mod==3:
        if d==3:
            sendMsg("A")
        elif d==0:
            sendMsg("R")
            flagA=True
        elif d==1:
            sendMsg("L")
            flagA=True
        else:
            sendMsg("H")
            flagA=True
    return flagA
    # sock.recv(1024)
x0=0
y0=0
xT=0
yT=0
isFinish=True
def setDestination(event,x,y,flags,param):
    global xT,yT,isFinish,note,ans_path,totalStep,step_i
    if isFinish and event==cv2.EVENT_LBUTTONDOWN:
        yT=x/block_x
        xT=y/block_y
        dfs(x0,y0,xT,yT,0)
        note -= 1
        print xT,yT
        for pnt in ans_path:
            note[pnt[0]] = pnt[1]
        print note
        print ans_path
        isFinish=False
        totalStep=len(ans_path)
        step_i=0

def getDir(dcx,dcy):
    if dcy > 0 and dcy > abs(dcx):
        return 1
    elif dcx > 0 and dcx > abs(dcy):
        return 3
    elif dcy < 0 and abs(dcy) > abs(dcx):
        return 0
    elif dcx < 0 and abs(dcx) > abs(dcy):
        return 2
inf=99999999
def distance(Xo,Yo,Xt,Yt):
    return pow((Xo-Xt),2)+pow((Yo-Yt),2)
def getCornerR(dx,dy,widx,widy,dst,col=0):
    smdis=inf
    # dx=[0,0,0,0]
    # dy=[0,0,0,0]
    ran = min(widx, widy)
    for i in range(ran):
        flag=True
        for j in range(i + 1):
            if dst[i][j] == col:
                temp = distance(0, 0, i, j)
                if temp<smdis:
                    dx[0]=i
                    dy[0]=j
                    smdis=temp
                    flag = False
        for j in range(i + 1):
            if dst[j][i] == col:
                temp = distance(0, 0, i, j)
                if temp<smdis:
                    dx[0]=j
                    dy[0]=i
                    smdis=temp
                    flag = False
        if flag and smdis!=inf:
            break
    smdis=inf
    for i in range(ran):
        flag=True
        for j in range(i + 1):
            if dst[widx - 1 - i][j] == col:
                temp=distance(widx,0,widx-1-i,j)
                if temp<smdis:
                    dx[1]=widx - 1 - i
                    dy[1]=j
                    smdis = temp
                    flag = False
        for j in range(i + 1):
            if dst[widx - 1 - j][i] == col:
                temp = distance(widx, 0, widx - 1 - j, i)
                if temp < smdis:
                    dx[1] = widx - 1 - j
                    dy[1] = i
                    smdis = temp
                    flag = False
        if flag and smdis!=inf:
            break
    smdis=inf
    for i in range(ran):
        flag=True
        for j in range(i + 1):
            if dst[i][widy - 1 - j] == col:
                temp=distance(0,widy,i,widy-1-j)
                if temp<smdis:
                    dx[2]=i
                    dy[2]=widy-1-j
                    smdis = temp
                    flag = False
        for j in range(i + 1):
            if dst[j][widy - 1 - i] == col:
                temp=distance(0,widy,j,widy-1-i)
                if temp<smdis:
                    dx[2]=j
                    dy[2]=widy-1-i
                    smdis = temp
                    flag = False
        if flag and smdis!=inf:
            break
    smdis=inf
    for i in range(ran):
        for j in range(i + 1):
            if dst[widx - 1 - i][widy - 1 - j] == col:
                temp=distance(widx,widy,widx-1-i,widy-1-j)
                if temp<smdis:
                    dx[3]=widx - 1 - i
                    dy[3]=widy - 1 - j
                    smdis=temp
                    flag = False
        for j in range(i + 1):
            if dst[widx - 1 - j][widy - 1 - i] == col:
                temp = distance(widx, widy, widx - 1 - j, widy - 1 - i)
                if temp < smdis:
                    dx[3] = widx - 1 - j
                    dy[3] = widy - 1 - i
                    smdis = temp
                    flag = False
        if flag and smdis!=inf:
            break
def getCorner(dx,dy,widx,widy,dst):
    flag = True
    ran=min(widx,widy)
    for i in range(ran):
        for j in range(i + 1):
            if dst[i][j] == 0 and flag:
                dx.append(i)
                dy.append(j)
                flag = False
                break
        if not flag:
            break
        for j in range(i + 1):
            if dst[j][i] == 0 and flag:
                dx.append(j)
                dy.append(i)
                flag = False
                break
        if not flag:
            break
    flag = True
    for i in range(ran):
        for j in range(i + 1):
            if dst[widx - 1 - i][j] == 0 and flag:
                dx.append(widx - 1 - i)
                dy.append(j)
                flag = False
                break
        if not flag:
            break
        for j in range(i + 1):
            if dst[widx - 1 - j][i] == 0 and flag:
                dx.append(widx - 1 - j)
                dy.append(i)
                flag = False
                break
        if not flag:
            break
    flag = True
    for i in range(ran):
        for j in range(i + 1):
            if dst[i][widy - 1 - j] == 0 and flag:
                dx.append(i)
                dy.append(widy - 1 - j)
                flag = False
                break
        if not flag:
            break
        for j in range(i + 1):
            if dst[j][widy - 1 - i] == 0 and flag:
                dx.append(j)
                dy.append(widy - 1 - i)
                flag = False
                break
        if not flag:
            break
    flag = True
    for i in range(ran):
        for j in range(i + 1):
            if dst[widx - 1 - i][widy - 1 - j] == 0 and flag:
                dx.append(widx - 1 - i)
                dy.append(widy - 1 - j)
                flag = False
                break
        if not flag:
            break
        for j in range(i + 1):
            if dst[widx - 1 - j][widy - 1 - i] == 0 and flag:
                dx.append(widx - 1 - j)
                dy.append(widy - 1 - i)
                flag = False
                break
        if not flag:
            break

def scanMap( dirs, dst):
    for i in range(5):
        for j in range(5):
            if i != 0:
                mid = (2 * j * block_y + block_y) / 2 + s_y
                lev = i * block_x + s_x
                flag = True
                for k in range(lev - 10, lev + 10):
                    if dst[k][mid] == 0:
                        flag = False
                        break
                if flag:
                    dirs[i][j][0] = 1
            if i != 4:
                mid = (2 * j * block_y + block_y) / 2 + s_y
                lev = (i + 1) * block_x + s_x
                flag = True
                for k in range(lev - 10, lev + 10):
                    if dst[k][mid] == 0:
                        flag = False
                        break
                if flag:
                    dirs[i][j][1] = 1
            if j != 0:
                mid = (2 * i * block_x + block_x) / 2 + s_x
                lev = j * block_y + s_y
                flag = True
                for k in range(lev - 10, lev + 10):
                    if dst[mid][k] == 0:
                        flag = False
                        break
                if flag:
                    dirs[i][j][2] = 1
            if j != 4:
                mid = (2 * i * block_x + block_x) / 2 + s_x
                lev = (j + 1) * block_y + s_y
                flag = True
                for k in range(lev - 10, lev + 10):
                    if dst[mid][k] == 0:
                        flag = False
                        break
                if flag:
                    dirs[i][j][3] = 1
    print dirs

preflag=False
def slightAdjust(dcx,dcy,px,py,supDir):
    #方向调整
    global preflag
    print supDir
    if supDir==1:
        if not preflag:
            if (py + 15) / 60 > py / 60:
                sock.send("r")
                preflag = True
                return False
            if (py - 15) / 60 < py / 60:
                sock.send("l")
                preflag = True
                return False
        if not preflag:
            if dcx > angle:
                sock.send("r")
                return False
            elif dcx < -angle:
                sock.send("l")
                return False
        if (px + 20) / 60 > px / 60:
            sock.send("b")
            return False
        elif (px - 20) / 60 < px / 60:
            sock.send("a")
            return False
    elif supDir==3:
        if not preflag:
            if (px + 15) / 60 > px / 60:
                sock.send("l")
                preflag = True
                return False
            if (px - 15) / 60 < px / 60:
                sock.send("r")
                preflag = True
                return False
        if not preflag:
            if dcy > angle:
                sock.send("l")
                return False
            elif dcy < -angle:
                sock.send("r")
                return False
        if (py + 20) / 60 > py / 60:
            sock.send("b")
            return False
        elif (py - 20) / 60 < py / 60:
            sock.send("a")
            return False
    elif supDir==0:
        if not preflag:
            if (py + 15) / 60 > py / 60:
                sock.send("l")
                preflag = True
                return False
            if (py - 15) / 60 < py / 60:
                sock.send("r")
                preflag = True
                return False
        if not preflag:
            if dcx > angle:
                sock.send("l")
                return False
            elif dcx < -angle:
                sock.send("r")
                return False
        if (px - 20) / 60 < px / 60:
            sock.send("b")
            return False
        elif (px + 20) / 60 > px / 60:
            sock.send("a")
            return False
    elif supDir==2:
        if not preflag:
            if (px + 15) / 60 > px / 60:
                sock.send("r")
                preflag = True
                return False
            if (px - 15) / 60 < px / 60:
                sock.send("l")
                preflag = True
                return False
        if not preflag:
            if dcy > angle:
                sock.send("r")
                return False
            elif dcy < -angle:
                sock.send("l")
                return False
        if (py - 20) / 60 < py / 60:
            sock.send("b")
            return False
        elif (py + 20) / 60 > py / 60:
            sock.send("a")
            return False
    preflag=False
    return True

cv2.namedWindow('binary')
cv2.namedWindow('frame')
cv2.setMouseCallback('frame',extract)
#Tracker
cv2.createTrackbar('X','binary',55,255,nothing)
cv2.createTrackbar('mod_switch','frame',0,2,nothing)
mod_switch=True
while True:
    ret,frame = cap.read()
    frame = cv2.GaussianBlur(frame, (9, 9), 0)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #Tracker
    mod=cv2.getTrackbarPos('mod_switch','frame')
    s = cv2.getTrackbarPos('X', 'binary')
    #maxi= cv2.getTrackbarPos('maxl', 'edge')
    #mini=cv2.getTrackbarPos('minl', 'edge')
   # edges = cv2.Canny(frame, mini, maxi)

    ret, th1 = cv2.threshold(gray, s, 255, cv2.THRESH_BINARY)
    # 用高斯滤波进行模糊处理，进行处理的原因：
    # 每个输入的视频都会因自然震动、光照变化或者摄像头本身等原因而产生噪声。对噪声进行平滑是为了避免在运动和跟踪时将其检测出来。
#查找轮廓
    # image, contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # img = cv2.drawContours(frame, contours,-1, (0,255,0), 3)
    #
    # approx=[]
    # hull=[]
    # for v in contours:
    #     epsilon = 0.01 * cv2.arcLength(v, True)
    #     approx.append(cv2.approxPolyDP(v, epsilon, True))
    #     hull.append(cv2.convexHull(v))
    #     x, y, w, h = cv2.boundingRect(v)
    #     leftmost = tuple(v[v[:, :, 0].argmin()][0])
    #     rightmost = tuple(v[v[:, :, 0].argmax()][0])
    #     topmost = tuple(v[v[:, :, 1].argmin()][0])
    #     bottommost = tuple(v[v[:, :, 1].argmax()][0])
    #     cv2.circle(img,leftmost,10,(255,255,0),4,-1)
    #     cv2.circle(img, rightmost, 10, (255, 255, 0), 4, -1)
    #     cv2.circle(img, topmost, 10, (255, 255, 0), 4, -1)
    #     cv2.circle(img, bottommost, 10, (255, 255, 0), 4, -1)
    #     img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    # # img = cv2.drawContours(frame, approx,-1, (0, 0, 255), 3)
    # img = cv2.drawContours(frame, hull, -1, (0, 0, 255), 3)
    # if s==61:
    #     print hull
    # print "over"
    # cv2.imshow("xx",img)
    if mod_switch:
        if mod==1 and cnt==1:
            cnt=0
            xi=[]
            yi=[]
            dx=[0,0,0,0]
            dy=[0,0,0,0]
            width_x=frame.shape[0]
            width_y=frame.shape[1]
            getCornerR(dx,dy,width_x,width_y,th1)
            print dx,dy
            pts1 = np.float32([[dy[0], dx[0]], [dy[1], dx[1]], [dy[2], dx[2]], [dy[3], dx[3]]])
            pts2 = np.float32([[0, 0], [0, 300], [300, 0], [300, 300]])
            M1 = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(th1, M1, (300, 300))
            scanMap(dirs, dst)
            cv2.imshow("map", dst)
            cv2.setMouseCallback("map", setDestination)
            mod_switch=False
        if mod==2 and cnt==1:
            cnt=0
            xi=[]
            yi=[]
            dx=[0,0,0,0]
            dy=[0,0,0,0]
            width_x=frame.shape[0]
            width_y=frame.shape[1]
            getCornerR(dx,dy,width_x,width_y,th1,255)
            # print dx,dy
            pts1 = np.float32([[dy[0], dx[0]], [dy[1], dx[1]], [dy[2], dx[2]], [dy[3], dx[3]]])
            pts2 = np.float32([[0, 0], [0, 300], [300, 0], [300, 300]])
            M1 = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(th1, M1, (300, 300))
            # dx = []
            # dy = []
            getCornerR(dx,dy,300,300,dst)
            # for i in range(4):
            #     cv2.circle(dst, (dy[i], dx[i]), 10, (0, 0, 0), -1)
            # cv2.imshow("zz", dst)
            pts1 = np.float32([[dy[0], dx[0]], [dy[1], dx[1]], [dy[2], dx[2]], [dy[3], dx[3]]])
            pts2 = np.float32([[0, 0], [0, 300], [300, 0], [300, 300]])
            M2 = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(dst, M2, (300, 300))
            scanMap(dirs, dst)
            cv2.imshow("map", dst)
            cv2.setMouseCallback("map", setDestination)
            mod_switch=False
        if mod==0 and cnt==4:
            pts1 = np.float32([[xi[0], yi[0]], [xi[1], yi[1]], [xi[2], yi[2]], [xi[3], yi[3]]])
            pts2 = np.float32([[0, 0], [0, 300], [300, 0], [300, 300]])
            M1 = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(th1, M1, (300, 300))
            dx = []
            dy = []
            getCorner(dx,dy,len_x,len_y,dst)
            print dx
            print dy
            pts1 = np.float32([[dy[0], dx[0]], [dy[1], dx[1]], [dy[2], dx[2]], [dy[3], dx[3]]])
            pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
            M2 = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(dst, M2, (300, 300))
            scanMap(dirs,dst)
            cv2.imshow("map",dst)
            cv2.setMouseCallback("map",setDestination)
            mod_switch=False
    else:
        trace = cv2.warpPerspective(frame, M1, (300, 300))
        if mod!=1:
            trace = cv2.warpPerspective(trace, M2, (300, 300))
        trace = cv2.GaussianBlur(trace, (19, 19), 0)
        hsv = cv2.cvtColor(trace, cv2.COLOR_BGR2HSV)
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        image, contours_green, hierarchy = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image, contours_red, hierarchy = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        skipflag=False
        if len(contours_green) and not skipflag:
            cnts = contours_green[0]
            M_green = cv2.moments(cnts)
            if M_green['m00'] != 0:
                cx_green = int(M_green['m10'] / M_green['m00'])
                cy_green = int(M_green['m01'] / M_green['m00'])
        else:
            skipflag=True
        if len(contours_red) and not skipflag:
            cnts = contours_red[0]
            M_red = cv2.moments(cnts)
            if M_red['m00'] != 0:
                cx_red = int(M_red['m10'] / M_red['m00'])
                cy_red = int(M_red['m01'] / M_red['m00'])
        else:
            skipflag=True
        # cx = (cx_green + cx_red) / 2
        # cy = (cy_green + cy_red) / 2
        cx=cx_green
        cy=cy_green
        cv2.circle(trace, (cx, cy), 10, (255, 0, 0), -1)
        cv2.circle(trace, (cx_red, cy_red), 10, (125, 255, 0), -1)
        cv2.imshow("trace", trace)
        y0=cx/block_x
        x0=cy/block_y
        # print x0,y0
        dcx=cx_green-cx_red
        dcy=cy_green-cy_red
        if fcnt==clock:
            fcnt = 0
            if isFinish:
                currentDir = getDir(dcx,dcy)
                if not skipflag:
                    slightAdjust(dcx, dcy, cy, cx, currentDir)
                clock=10
            else:
            #位置微调
                if skipflag or slightAdjust(dcx,dcy,cy,cx,currentDir):
                    clock=60
                # print currentDir
                    # print x0,y0,currentDir
                    # print ans_path
                    if attachflag:
                        sock.send("A")
                        attachflag=False
                    else:
                        if sentCommand(ans_path[step_i][1],currentDir):
                            attachflag=True
                        currentDir=ans_path[step_i][1]
                        step_i+=1
                        time.sleep(0.5)
                    if step_i==totalStep and attachflag==False:
                        isFinish=True
                        note = np.zeros((5, 5), int)
                        ans = 100000
                        path = []
                        # sentCommand(note[x0,y0],currentDir)
                        # print data
                        # data=""
                else:
                    clock=10
        else:
            fcnt+=1
            #dst=cv2.resize(dst,(300,300))
    #out.write(frame)#写入视频
    #print contours
    cv2.imshow('frame',frame)#一个窗口用以显示原视频
    cv2.imshow('binary',th1)
    #cv2.imshow('self',selfy)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
    # time.sleep(5)
sock.close()
cap.release()
#out.release()
cv2.destroyAllWindows()