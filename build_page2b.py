# Seated man's screen (same book as the couple's): cut him out and rebuild the book, town and floor behind them.
import numpy as np, cv2
from PIL import Image
A=np.asarray(Image.open('src.png').convert('RGB')).copy(); H,W=A.shape[:2]
OUT='/home/user/Quillandcolor/first-page/assets/'
box=(20,1000,480,1530)
m=np.zeros((H,W),np.uint8); m[box[1]:box[3],box[0]:box[2]]=np.asarray(Image.open('mask.png').convert('L'))
n,l,s,_=cv2.connectedComponentsWithStats((m>40).astype(np.uint8)); k=1+np.argmax(s[1:,4])
m=np.where(cv2.dilate((l==k).astype(np.uint8),np.ones((5,5)))>0,m,0).astype(np.uint8)
ys,xs=np.where(m>8); x0,x1,y0,y1=xs.min(),xs.max()+1,ys.min(),ys.max()+1
Image.fromarray(np.dstack([A,m])[y0:y1,x0:x1],'RGBA').save(OUT+'man2.webp','WEBP',quality=90,method=6)
print('man2 left %.3f%% top %.3f%% width %.3f%%'%(x0/W*100,y0/H*100,(x1-x0)/W*100))
hole=cv2.dilate((m>8).astype(np.uint8),np.ones((11,11)))>0
B=A.copy(); done=np.zeros_like(hole)
P2=np.asarray(Image.open('../p2/src.png').convert('RGB'))
big=cv2.dilate(hole.astype(np.uint8),np.ones((41,41)))>0
fill=hole.copy(); fill[1392:,:740]|=big[1392:,:740]
fill[960:1392,96:712]|=big[960:1392,96:712]

for x in range(W):
    col=np.where(fill[:,x])[0]
    if len(col)==0: continue
    top=col.min()
    for y in col:
        if x<50 and y<1392:                              # town and flowers at the far left: the same spot in the earlier open-book image
            B[y,x]=P2[y,x]
        elif 50<=x<=96 and y<1350:                       # cover edge: continue downwards
            sy=top-44+((y-top)%40); B[y,x]=A[sy,x]
        elif 96<x<325 and y<1340:                        # page stack and curled left page: continue each column downwards
            sy=top-44+((y-top)%40); B[y,x]=A[sy,x]
        elif 325<=x<=362 and y<1290:                     # the right page's left-hand border lines run straight down
            sy=top-44+((y-top)%40); B[y,x]=A[sy,x]
        elif 325<=x<=712 and y<1290:                     # right page paper: plain paper from just above the flower, same column band
            B[y,x]=A[960+((y-960)%60), 600+(x%120)] if y>=960 else A[y,x]
        elif 325<=x<=712 and y<1342:                     # right page's bottom border lines: continue them sideways from the clear stretch
            B[y,x]=A[y,545]
        elif 50<=x<=712 and y<1392:                      # the cover's bottom band
            B[y,x]=A[y,560+((x-50)%280)]
        elif y>=1392:                                    # floor
            B[y,x]=A[y,740+(x%180)] if y<1425 else (A[y+138 if y+138<H else y+75, x]*min(1,max(0,(x-100)/60)) + A[y+138 if y+138<H else y+75, x+300]*(1-min(1,max(0,(x-100)/60))))   # floor: shadow band by row, cobbles from just below
        else: continue
        done[y,x]=True
hole=fill
# the page edges run straight down: smooth the copied columns vertically so no repeat shows
reg=done.copy(); reg[1340:]=False; reg[:, :96]=False; reg[:, 713:]=False; reg[1290:, 325:]=False
vb=cv2.blur(B,(1,61))
B=np.where(reg[...,None],vb,B)
# Right page, bottom-left corner: rebuild it as the mirror of the bottom-right corner (same flower and border).
# The two vertical border lines mirror about x = 578; the bottom border drops ~0.03 px per px to the right.
Bf=B.astype(np.float32)
for x in range(328,561):
    xs=1156-x; dy=0.03*(xs-x)
    for y in range(1110,1326):
        Bf[y,x]=A[int(round(y+dy)),xs]
wx=np.clip((560-np.arange(W))/25.0,0,1); wx[:328]=1          # fade back into the page on the right
wy=np.clip((np.arange(H)-1110)/25.0,0,1)                     # and into the paper above
wy2=np.clip((1326-np.arange(H))/8.0,0,1)
wc=(wy[:,None]*wy2[:,None]*wx[None,:]); wc[:1110]=0; wc[1326:]=0; wc[:, :328]=0; wc[:, 561:]=0
Bf=Bf*wc[...,None]+B.astype(np.float32)*(1-wc[...,None])
# the gutter beside it runs straight down
for x in range(316,328):
    for y in range(1100,1342):
        Bf[y,x]=A[1040+((y-1040)%40),x]
B=Bf.astype(np.uint8)
done[1100:1342,316:561]=True
# soften the join between the left and right halves of the page edge and bottom band
bl=cv2.GaussianBlur(B,(0,0),sigmaX=4,sigmaY=.5); B[1320:1392,312:342]=bl[1320:1392,312:342]
# soften the one tiling seam in the bottom band, and feather the floor patch into the untouched floor
Bf=B.astype(np.float32); bl=cv2.GaussianBlur(Bf,(0,0),2.5)
Bf=np.where(((np.arange(H)[:,None]>=1392)&(np.arange(H)[:,None]<1425)&done)[...,None], cv2.GaussianBlur(Bf,(0,0),sigmaX=6,sigmaY=1), Bf)
fl=np.zeros((H,W),np.float32); fl[1392:]=done[1392:]
w=np.maximum(cv2.GaussianBlur(cv2.erode(fl,np.ones((9,9))),(0,0),16), hole.astype(np.float32)*(np.arange(H)[:,None]>=1392))
w=np.where(np.arange(H)[:,None]>=1392, w, 1.0)[...,None]
B=np.clip(Bf*w+A.astype(np.float32)*(1-w),0,255).astype(np.uint8)
rest=(hole&~done).astype(np.uint8)*255
B=cv2.inpaint(B,rest,9,cv2.INPAINT_TELEA)
e=cv2.dilate(done.astype(np.uint8),np.ones((3,3)))-cv2.erode(done.astype(np.uint8),np.ones((3,3)))
B=np.where(e[...,None]>0,cv2.GaussianBlur(B,(0,0),1),B)
Image.fromarray(B).save('L_bg2c.png')
Image.fromarray(B).save(OUT+'bg2c.webp','WEBP',quality=90,method=6)
