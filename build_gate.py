# Gate opening: a copy of the gate image with the (blurry) book painted out, faded in under the sharp book.
import numpy as np, cv2
from PIL import Image
A=np.asarray(Image.open('src.png').convert('RGB')).copy(); H,W=A.shape[:2]
m=np.zeros((H,W),np.uint8); m[552:1172,268:668]=1
F=A.astype(np.float32); xs=np.arange(W)
for y in range(552,1172):             # row-wise fill from either side, then soften (the scene here is out of focus)
    k=m[y]==0
    for c in range(3): F[y,:,c]=np.where(k,F[y,:,c],np.interp(xs,xs[k],F[y,k,c]))
blur=cv2.GaussianBlur(F,(0,0),sigmaX=30,sigmaY=12)
w=cv2.GaussianBlur(m.astype(np.float32),(0,0),6)[...,None]
B=np.clip(F*(1-w)+blur*w,0,255).astype(np.uint8)
Image.fromarray(B).save('/home/user/Quillandcolor/first-page/assets/gate-nobook.webp','WEBP',quality=85,method=6)
Image.fromarray(B).crop((200,600,740,1200)).save('nobook.png')
