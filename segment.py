from rembg import remove, new_session
from PIL import Image
im=Image.open('src.png').convert('RGB')
for model,name,box in [('birefnet-portrait','woman',(0,560,250,1510)),('isnet-general-use','man',(670,510,940,1560)),('birefnet-portrait','man',(670,510,940,1560))]:
    s=new_session(model)
    remove(im.crop(box),session=s,only_mask=True).save(f'{name}_{model}.png')
