# Page 5: the closed book with the woman on the left and the man on the right. Cut out both characters and rebuild what is behind them.
import numpy as np, cv2
from PIL import Image
A=np.asarray(Image.open('src.png').convert('RGB'))[:1672].copy(); H,W=A.shape[:2]
OUT='/home/user/Quillandcolor/first-page/assets/'
def full(f,box):
    m=np.zeros((H,W),np.uint8); mm=np.asarray(Image.open(f).convert('L')); h=min(mm.shape[0],H-box[1])
    m[box[1]:box[1]+h,box[0]:box[2]]=mm[:h]
    n,l,s,_=cv2.connectedComponentsWithStats((m>40).astype(np.uint8)); k=1+np.argmax(s[1:,4])
    return np.where(cv2.dilate((l==k).astype(np.uint8),np.ones((5,5)))>0,m,0).astype(np.uint8)
wm=full('mask_L.png',(0,600,260,1440)); mm=full('mask_R.png',(730,500,940,1400))   # woman on the left, man on the right
for n,m in (('man5',mm),('woman5',wm)):
    ys,xs=np.where(m>8); x0,x1,y0,y1=xs.min(),xs.max()+1,ys.min(),ys.max()+1
    Image.fromarray(np.dstack([A,m])[y0:y1,x0:x1],'RGBA').save(OUT+n+'.webp','WEBP',quality=90,method=6)
    print(n,'left %.3f%% top %.3f%% width %.3f%%'%(x0/W*100,y0/H*100,(x1-x0)/W*100))
hole=cv2.dilate(((mm>8)|(wm>8)).astype(np.uint8),np.ones((11,11)))>0
hole[540:1412,143:241]=True                           # the whole lower spine is rebuilt from its clean top section
B=A.copy(); done=np.zeros_like(hole)
P=248
for y in range(H):
    for x in range(W):
        if not hole[y,x]: continue
        if 143<=x<=246 and 292<=y<=1412:            # spine: tile the visible top section
            sy=y
            while sy>=540: sy-=P
            if sy<292: sy+=P
            B[y,x]=A[sy,x]; done[y,x]=True
        elif 760<=x<=797 and 300<y<=1400:            # cover's right edge: continue the column from above
            B[y,x]=A[560+((y-560)%40),x]; done[y,x]=True
# everything else: soft wash from the surrounding rows
rest=hole&~done
F=B.astype(np.float32); xs=np.arange(W)
for y in range(H):
    k=~rest[y]
    if rest[y].any() and k.sum()>2:
        for c in range(3): F[y,:,c]=np.where(k,F[y,:,c],np.interp(xs,xs[k],F[y,k,c]))
blur=cv2.GaussianBlur(F,(0,0),sigmaX=16,sigmaY=34)
w=cv2.GaussianBlur(rest.astype(np.float32),(0,0),3)[...,None]
B=np.clip(F*(1-w)+blur*w,0,255).astype(np.uint8)
# Left of the spine the man hid nearly all of the town. Rebuild that strip from the same town on the
# open-book image's left edge (x 0..48 there), mirrored to widen it, with its rows remapped so the
# town, water, flowers and floor line up with this image.
S=np.asarray(Image.open('../p2/src.png').convert('RGB')).astype(np.float32)[:, :48]
def band(img,dy):  # shift only the town band (rows 560..1130) so the copies don't mirror each other exactly
    o=img.copy(); o[600:1110]=np.roll(img[600:1110],dy,0); return o
S=np.concatenate([S, band(S[:, ::-1],95), band(S,190), band(S[:, ::-1],285)], 1)[:, :146]
def srcrow(y):  # this image's row -> open-book image's row
    for (a0,a1,b0,b1) in ((0,540,0,560),(540,1290,560,1130),(1290,1320,1130,1200),(1320,1420,1200,1380),(1420,1672,1380,1672)):
        if y<a1: return int(b0+(y-a0)*(b1-b0)/(a1-a0))
    return 1671
strip=np.stack([S[min(1671,srcrow(y))] for y in range(H)])
reg=np.zeros((H,W),bool); reg[:, :146]=rest[:, :146]
wr=cv2.GaussianBlur(cv2.dilate(reg.astype(np.uint8),np.ones((3,3))).astype(np.float32),(0,0),2)[...,None]
Bf=B.astype(np.float32); Bf[:, :146]=Bf[:, :146]*(1-wr[:, :146])+strip*wr[:, :146]
# soften the joins between the three copies
for xj in (48,96):
    Bf[:, xj-3:xj+3]=np.where(reg[:, xj-3:xj+3,None], cv2.GaussianBlur(Bf,(0,0),2)[:, xj-3:xj+3], Bf[:, xj-3:xj+3])
# Right of the book the woman hid the town too: same treatment, using copies with other offsets.
S2=np.asarray(Image.open('../p2/src.png').convert('RGB')).astype(np.float32)[:, :48]
S2=np.concatenate([band(S2[:, ::-1],140), band(S2,235), band(S2[:, ::-1],330)], 1)[:, :144]
strip2=np.stack([S2[min(1671,srcrow(y))] for y in range(H)])
reg2=np.zeros((H,W),bool); reg2[:, 796:]=rest[:, 796:]
wr2=cv2.GaussianBlur(cv2.dilate(reg2.astype(np.uint8),np.ones((3,3))).astype(np.float32),(0,0),2)[...,None]
Bf[:, 796:]=Bf[:, 796:]*(1-wr2[:, 796:])+strip2*wr2[:, 796:]
for xj in (844,892):
    Bf[:, xj-3:xj+3]=np.where(reg2[:, xj-3:xj+3,None], cv2.GaussianBlur(Bf,(0,0),2)[:, xj-3:xj+3], Bf[:, xj-3:xj+3])
B=np.clip(Bf,0,255).astype(np.uint8)
Image.fromarray(B).save('L_bg5.png')
Image.fromarray(B).save(OUT+'bg5.webp','WEBP',quality=90,method=6)
