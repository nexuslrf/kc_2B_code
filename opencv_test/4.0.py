import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread('maze.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('233',th1)
th1=cv2.resize(th1,(300,300))
print th1.shape
print th1
dx=[]
dy=[]
flag=True

def setDestination(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        yT=x/block_x
        xT=y/block_y
        print xT,yT

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


getCorner(dx,dy,300,300,th1)
print dx,dy
pts1 = np.float32([[dx[0], dy[0]], [dx[1], dy[1]], [dx[2], dy[2]], [dx[3], dy[3]]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
M2 = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(th1, M2, (300, 300))
cv2.imshow("dst",dst)
cv2.setMouseCallback("dst",setDestination)
len_x=dx[1]-dx[0]
len_y=dy[2]-dy[0]
block_x=len_x/5
block_y=len_y/5
s_x=dx[0]
s_y=dx[0]
print block_x
print block_y
dirs=np.zeros((5, 5, 4), dtype=int) # 0:up 1:down 2:left 3:right
for i in range(5):
    for j in range(5):
        if  i!=0:
            mid=(2*j*block_y+block_y)/2+s_y
            lev=i*block_x+s_x
            flag=True
            for k in range(lev-10,lev+10):
                if th1[k][mid]==0:
                    flag=False
                    break
            if flag:
                dirs[i][j][0]=1
        if i!=4:
            mid = (2 * j * block_y + block_y) / 2+s_y
            lev = (i+1) * block_x+s_x
            flag = True
            for k in range(lev - 10, lev + 10):
                if th1[k][mid] == 0:
                    flag = False
                    break
            if flag:
                dirs[i][j][1] = 1
        if j!=0:
            mid = (2 * i * block_x + block_x) / 2+s_x
            lev = j * block_y+s_y
            flag = True
            for k in range(lev - 10, lev + 10):
                if th1[mid][k] == 0:
                    flag = False
                    break
            if flag:
                dirs[i][j][2] = 1
        if j!=4:
            mid = (2 * i * block_x + block_x) / 2+s_x
            lev = (j+1) * block_y+s_y
            flag = True
            for k in range(lev - 10, lev + 10):
                if th1[mid][k] == 0:
                    flag = False
                    break
            if flag:
                dirs[i][j][3] = 1
#print dirs
path=[]
ans_path=[]
ans=100000
p_x=[-1,1,0,0]
p_y=[0,0,-1,1]
note=np.zeros((5,5),int)
def dfs(x1,y1,x2,y2,step):
    global ans,dirs,path,p_x,p_y,ans_path
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
dfs(1,0,3,4,0)
note-=1
for pnt in ans_path:
    note[pnt[0]]=pnt[1]
print note
print ans_path

cv2.waitKey(0)