import numpy as np, cv2
from PIL import Image, ImageDraw
A=np.asarray(Image.open('src.png').convert('RGB')).copy(); H,W=A.shape[:2]
def full(f,box):
    m=np.zeros((H,W),np.uint8); m[box[1]:box[3],box[0]:box[2]]=np.asarray(Image.open(f).convert('L')); return m
wm=full('woman_birefnet-portrait.png',(0,560,250,1510)); mm=full('man_birefnet-portrait.png',(670,510,940,1560))
r,g,b=[A[...,i].astype(int) for i in range(3)]
reg=np.zeros_like(wm,bool); reg[1270:1420,0:70]=True
wm[reg&~((r>150)&(r-g>40)&(r-b>60))]=0
def largest(m):
    n,l,s,_=cv2.connectedComponentsWithStats((m>40).astype(np.uint8)); k=1+np.argmax(s[1:,4])
    keep=cv2.dilate((l==k).astype(np.uint8),np.ones((5,5)))>0; return np.where(keep,m,0).astype(np.uint8)
wm=largest(wm); mm=largest(mm)

def save_layer(name,rgb,alpha,pad=0):
    ys,xs=np.where(alpha>6); x0,x1,y0,y1=xs.min(),xs.max()+1,ys.min(),ys.max()+1
    Image.fromarray(np.dstack([rgb,alpha])[y0:y1,x0:x1].astype(np.uint8),'RGBA').save(f'L_{name}.png')
    print(f"'{name}':[{x0},{y0},{x1},{y1}],")
save_layer('woman',A,wm); save_layer('man',A,mm)
chars=cv2.dilate(((wm>10)|(mm>10)).astype(np.uint8),np.ones((9,9)))>0
def pushpull(img,known):
    img=img.astype(np.float32); k=known.astype(np.float32)
    if min(img.shape[:2])<4:
        s=k.sum(); 
        return np.ones_like(img)*((img*k[...,None]).sum((0,1))/max(s,1))
    small=cv2.resize(img*k[...,None],(img.shape[1]//2,img.shape[0]//2),interpolation=cv2.INTER_AREA)
    ks=cv2.resize(k,(img.shape[1]//2,img.shape[0]//2),interpolation=cv2.INTER_AREA)
    sm=np.where(ks[...,None]>1e-4,small/np.maximum(ks[...,None],1e-4),0)
    filled=pushpull(sm,ks>1e-4)
    up=cv2.resize(filled,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_LINEAR)
    return np.where(known[...,None],img,up)

# book polygon
poly=[(144,252),(146,240),(160,232),(200,228),(232,231),(246,234),(796,289),(803,295),(806,305),(804,1428),(798,1434),(246,1484),(236,1491),(165,1491),(150,1484),(147,1470)]
bm=Image.new('L',(W,H),0); ImageDraw.Draw(bm).polygon(poly,fill=255); book=np.asarray(bm)>0
# --- book layer: fill char holes inside book
B=A.copy(); manbig=cv2.dilate((mm>10).astype(np.uint8),np.ones((61,61)))>0
manbig[:, :640]=False
hole=(chars|manbig)&book
for y in range(H):
    for x in range(140,200):
        if hole[y,x]:
            xm=387-x
            if not chars[y,xm]: B[y,x]=A[y,xm]; hole[y,x]=False
tilemask=np.zeros_like(hole)
for x in range(640,807):
    for y in range(300,1440):
        if hole[y,x]:
            if y>1240:
                # bottom-right corner: mirror the (uncovered) bottom-left corner, allowing for perspective
                xx=262.5+(787-x)*1.29                      # border lines: R thick 787 / thin 776.5 -> L 262.5 / 276
                Rthin=1414-(x-600)*0.0875; Lthin=1440-(xx-300)*0.083   # slanted bottom border lines
                yy=Lthin+(y-Rthin)*(13/11.5)
                xx=int(round(xx)); yy=int(round(yy))
                B[y,x]=A[min(yy,H-1),xx]
            else:
                B[y,x]=A[500+(y%20),x]
            hole[y,x]=False; tilemask[y,x]=True
# match low-frequency lighting of the tiled patch to the surrounding cover
known=book&~chars&~manbig&(A[...,0]<45)&(A[...,1]<110)
Bf=B.astype(np.float32)
low_known=pushpull(np.where(known[...,None],Bf,0),known)
low_known=cv2.GaussianBlur(low_known,(0,0),12)
plain=(Bf[...,0]<45)&(Bf[...,1]<110)
low_tile=cv2.GaussianBlur(pushpull(np.where(plain[...,None],Bf,0),plain),(0,0),12)
adj=np.clip(Bf-low_tile+low_known,0,255)
tm3=cv2.GaussianBlur(cv2.dilate(tilemask.astype(np.uint8),np.ones((9,9))).astype(np.float32),(0,0),5)[...,None]
B=(Bf*(1-tm3)+adj*tm3).astype(np.uint8)
if hole.any(): B=cv2.inpaint(B,hole.astype(np.uint8)*255,5,cv2.INPAINT_TELEA)
# ---- logo: pull it off the cover, then patch the cover underneath
Bi=B.astype(int); r_,g_,b_=Bi[...,0],Bi[...,1],Bi[...,2]
box=np.zeros((H,W),bool); box[545:945,295:700]=True
teal=np.array([6,60,51],np.float32)
la=np.clip((r_-teal[0]-25)/150.0,0,1)                       # gold = red rises far above the teal
lum=(0.3*r_+0.59*g_+0.11*b_).astype(np.float32)
nogold=np.where(la>0.02,np.nan,lum)
ref=cv2.GaussianBlur(np.nan_to_num(nogold,nan=45.0),(0,0),9)
dark=np.clip((ref-lum-5)/25.0,0,1)
dark=cv2.dilate(dark,np.ones((3,3)))*(cv2.dilate((la>0.04).astype(np.uint8),np.ones((15,15)))>0)
la=np.maximum(la,dark*0.85)
la=np.where(box,la,0).astype(np.float32)
la=cv2.morphologyEx((la*255).astype(np.uint8),cv2.MORPH_OPEN,np.ones((2,2))).astype(np.float32)/255
lm=cv2.dilate((la>0.04).astype(np.uint8),np.ones((7,7)))>0
# cover patch: copy texture from 330 rows below, then match lighting
Bf=B.astype(np.float32); P=Bf.copy()
ys,xs=np.where(lm); P[ys,xs]=Bf[960+((ys-545)%240),np.where(xs>670,xs-80,xs)]
known2=book&~lm&box
low_k=cv2.GaussianBlur(pushpull(np.where(known2[...,None],Bf,0),known2),(0,0),8)
P=np.clip(P-cv2.GaussianBlur(P,(0,0),8)+low_k,0,255)
w3=cv2.GaussianBlur(lm.astype(np.float32),(0,0),2)[...,None]
Bclean=(Bf*(1-w3)+P*w3)
# un-premultiply logo colour against the cover behind it
a3=np.maximum(la,1e-3)[...,None]
LC=np.clip((Bf-(1-a3)*Bclean)/a3,0,255)
LC=np.where(la[...,None]>0.6,Bf,LC)
save_layer('logo',LC,(la*255).astype(np.uint8))
# even out the darker shading the artwork had inside the circle
R=np.zeros((H,W),np.uint8); cv2.circle(R,(500,752),215,1,-1); R=(R>0)&book
outside=book&~R&~chars&~manbig&(Bclean[...,0]<45)&(Bclean[...,1]<110)
lowR=cv2.GaussianBlur(pushpull(np.where(outside[...,None],Bclean,0),outside),(0,0),20)
eq=np.clip(Bclean-cv2.GaussianBlur(Bclean,(0,0),20)+lowR,0,255)
wR=cv2.GaussianBlur(R.astype(np.float32),(0,0),25)[...,None]
Bclean=Bclean*(1-wR)+eq*wR
B=Bclean.astype(np.uint8)
ba=cv2.GaussianBlur(book.astype(np.uint8)*255,(3,3),0)
save_layer('book',B,ba)
# --- tree (top-left, non-sky)
skyish=(b>r+40)&(b>140)
tm=np.zeros((H,W),bool); tm[0:190,0:340]=True; tm&=~skyish
tm=cv2.morphologyEx(tm.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3)))
ta=cv2.GaussianBlur(tm*255,(3,3),0); save_layer('tree',A,ta)
# --- pot (bottom-left)
pm=np.zeros((H,W),np.uint8); pot_poly=[(0,1440),(30,1430),(60,1460),(95,1520),(120,1560),(122,1672),(0,1672)]
ImageDraw_img=Image.new('L',(W,H),0); ImageDraw.Draw(ImageDraw_img).polygon(pot_poly,fill=255); pm=np.asarray(ImageDraw_img)>0
# refine: within pot poly exclude floor-like pixels above pot rim (y<1545)
floor=(r>180)&(g>110)&(g<170)&(b<130)
pm=pm&~((np.arange(H)[:,None]<1540)&floor)
pa=cv2.GaussianBlur(pm.astype(np.uint8)*255,(3,3),0); save_layer('pot',A,pa)
# --- background: remove everything, push-pull fill
hole=chars|book|pm
hole=cv2.dilate(hole.astype(np.uint8),np.ones((5,5)))>0
A0=A.copy()
fl=np.zeros_like(hole)
for y in range(1440,H):
    for x in range(W):
        if hole[y,x]:
            for sx,sy in ((140,0),(260,0),(0,110),(0,180),(-150,0),(0,-60)):
                xx,yy=x+sx,y+sy
                if 0<=xx<W and 0<=yy<H and not hole[yy,xx]:
                    A0[y,x]=A[yy,xx]; fl[y,x]=True; break
hole=hole&~fl
A=A0
F=A.astype(np.float32).copy()
xs=np.arange(W)
for y in range(H):
    k=~hole[y]
    if k.sum()<2: continue
    for c in range(3): F[y,:,c]=np.where(k,F[y,:,c],np.interp(xs,xs[k],A[y,k,c].astype(np.float32)))
# soft, smooth colour wash (no streaks): 2-D push-pull fill, then blur
# row-wise colours keep the sky / town / river / floor bands; blur mostly sideways so it reads as distant haze
pp=0.6*cv2.GaussianBlur(F,(0,0),sigmaX=34,sigmaY=26)+0.4*cv2.GaussianBlur(pushpull(A,~hole),(0,0),16)
wh=cv2.GaussianBlur(hole.astype(np.float32),(0,0),4)[...,None]
F=F*(1-wh)+pp*wh
# add slight grain in holes so it isn't plasticky
rng=np.random.default_rng(1); F=F+np.where(hole[...,None],rng.normal(0,2.5,F.shape),0)
F=np.clip(F,0,255).astype(np.uint8)
F=cv2.GaussianBlur(F,(0,0),0.6)*hole[...,None]+F*(~hole[...,None])
F=F.astype(np.uint8)
# soften the seams of the copied floor patches
sm=cv2.GaussianBlur(F,(0,0),1.2); e=cv2.dilate(fl.astype(np.uint8),np.ones((5,5)))-cv2.erode(fl.astype(np.uint8),np.ones((5,5)))
F=np.where(e[...,None]>0,sm,F)
Image.fromarray(F).save('L_bg.png')
