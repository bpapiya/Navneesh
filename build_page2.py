# Page 2: cut out the seated man and rebuild what sits behind him.
import numpy as np, cv2
from PIL import Image
A=np.asarray(Image.open('src.png').convert('RGB')).copy(); H,W=A.shape[:2]
box=(10,990,480,1530)
m=np.zeros((H,W),np.uint8); m[box[1]:box[3],box[0]:box[2]]=np.asarray(Image.open('man_mask.png').convert('L'))
n,l,s,_=cv2.connectedComponentsWithStats((m>40).astype(np.uint8)); k=1+np.argmax(s[1:,4])
m=np.where(cv2.dilate((l==k).astype(np.uint8),np.ones((5,5)))>0,m,0).astype(np.uint8)
ys,xs=np.where(m>8); x0,x1,y0,y1=xs.min(),xs.max()+1,ys.min(),ys.max()+1
Image.fromarray(np.dstack([A,m])[y0:y1,x0:x1],'RGBA').save('L_man2.png'); print('man2',[x0,y0,x1,y1])
hole=cv2.dilate((m>8).astype(np.uint8),np.ones((11,11)))>0
B=A.copy(); filled=np.zeros_like(hole)
EDGE=1360   # bottom of the left-hand pages
for x in range(W):
    col=np.where(hole[:,x])[0]
    if len(col)==0: continue
    top=col.min()
    for y in col:
        if x<45 and y>1180:  continue                        # flowers: leave for inpaint
        if y<EDGE and x<=400:                                # pages / cover edge: continue the column downwards
            P=36; sy=970+((y-970)%P)
            sx=x if x<125 else min(W-1,int(round(x+0.1*(y-sy))))   # the curled page's edge leans ~0.1px per row
            B[y,x]=A[sy,sx]; filled[y,x]=True
        elif y<1398:                                         # book's bottom band: borrow from the right-hand side
            xx=x+520
            if xx<W: B[y,x]=A[y,xx]; filled[y,x]=True
        else:                                                # floor
            xx=x+460
            if xx<W and not hole[y,xx]: B[y,x]=A[y,xx]; filled[y,x]=True
rest=(hole&~filled).astype(np.uint8)*255
B=cv2.inpaint(B,rest,7,cv2.INPAINT_TELEA)
# soften seams of the copied patches
e=cv2.dilate(filled.astype(np.uint8),np.ones((3,3)))-cv2.erode(filled.astype(np.uint8),np.ones((3,3)))
B=np.where(e[...,None]>0,cv2.GaussianBlur(B,(0,0),1),B)
Image.fromarray(B).save('L_bg2.png')
