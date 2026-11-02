# Copies of each page's background with the tree (top-left) and pot (bottom-left) painted out,
# used only for the mirrored sky/floor strips above and below the picture on tall phones.
import sys, numpy as np, cv2
from PIL import Image
src,out=sys.argv[1],sys.argv[2]
A=np.asarray(Image.open(src).convert('RGB')).copy(); H,W=A.shape[:2]
r,g,b=[A[...,i].astype(int) for i in range(3)]
sky=(b>r+40)&(b>140)
tree=np.zeros((H,W),bool); tree[0:260,0:420]=True; tree&=~sky
tree=cv2.dilate(tree.astype(np.uint8),np.ones((9,9)))
A=cv2.inpaint(A,tree*255,15,cv2.INPAINT_TELEA)
A[1500:,0:150]=A[1500:,220:370]          # floor over the pot
Image.fromarray(A).save(out)
