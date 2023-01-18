VERBOSE = False

import sys, os, json, ast
from platform import python_version
if VERBOSE:
    print(sys.executable)
    print('py: ' + python_version())
try:
    import numpy as np
    if VERBOSE:
        print('np: ' + np.__version__)
except:
    print('np: ---')
try:
    import scipy as sp
    if VERBOSE:
        print('sp: ' + sp.__version__)
except:
    print('sp: ---')
try:
    import matplotlib
    if VERBOSE:
        print('mpl: ' + matplotlib.__version__)
    import matplotlib.pyplot as plt
    import matplotlib.image as img
except:
    print('mpl: ---')
try:
    import PIL
    if VERBOSE:
        print('PIL: ' + PIL.__version__)
    from PIL import Image, ImageOps, ImageChops
except:
    print('PIL: ---')
try:
    import cv2
    if VERBOSE:
        print('cv2: ' + cv2.__version__)
except:
    print('cv2: ---')
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    import tensorflow as tf
    if VERBOSE:
        print('tf: ' + tf. __version__)
except:
    print('tf: ---')
try:
    import h5py
    if VERBOSE:
        print('h5: ' + h5py. __version__)
except:
    print('h5: ---')
    

# Neural network parameters
EPOCHS = 10000
BATCH_SIZE = 1
LEARNING_RATE = 0.0005


# Image operations (pillow)
def is_greyscale(img):
    """
    Checks if RGB Image is grayscale (R==G==B).
    """
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i,j))
            if r != g != b: 
                return False
    return True

def grayscale_rescaled(img):
    """
    Rescales to 0--255 grayscale.
    """
    gray = ImageOps.grayscale(img)
    array = np.array(gray)
    min_value = np.min(array)
    max_value = np.max(array)
    array = (array-min_value).astype(np.float64)*255/(max_value-min_value)
    return Image.fromarray(np.uint8(array))    

def transform(img, p1, p2, P1, P2, new_size):
    (w, h) = img.size
    (W, H) = new_size

    p21 = p2-p1
    angle = np.arctan(p21[1]/p21[0])
    P21 = P2-P1
    Angle = np.arctan(P21[1]/P21[0])
    scale_x = P21[0]/p21[0]
    scale_y = P21[1]/p21[1]

    rotation_angle = angle-Angle
    img = img.rotate(np.rad2deg(angle-Angle))
    
    p0 = np.array([w/2, h/2], dtype=np.float64)
    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    R = np.array([[c,s],[-s,c]])
    p1 = p0 + R@(p1-p0)
    p2 = p0 + R@(p2-p0)
    
    img = img.crop((int(p1[0]-P1[0]/scale_x),
         int(p1[1]-P1[1]/scale_y),
         int(p2[0]+(W-P2[0])/scale_x),
         int(p2[1]+(H-P2[1])/scale_y)
         ))
    
    return img.resize(new_size)

def cut4img(img):
    width, height = img.size
    img00 = img.crop((0, 0, width//2, height//2))
    img10 = img.crop((0, height//2, width//2, height))
    img01 = img.crop((width//2, 0, width, height//2))
    img11 = img.crop((width//2, height//2, width, height))
    return img00, img10, img01, img11

# image operations (numpy, open-cv2)
def cut4(img):
    width, height = img.size
    img00 = img.crop((0, 0, width//2, height//2))
    img10 = img.crop((0, height//2, width//2, height))
    img01 = img.crop((width//2, 0, width, height//2))
    img11 = img.crop((width//2, height//2, width, height))
    return img00, img10, img01, img11

def plot3d(z, W=1, H=1):
    """ 
    Optinal arguments are x-range and y-range in um.
    """
    w = z.shape[0]
    h = z.shape[1]
    
    X = np.outer(np.linspace(0, W, w), np.ones(h))
    Y = np.outer(np.ones(w), np.linspace(0, H, h))
    ax = plt.figure().add_subplot(projection='3d')
    #ax.set_xlabel('x, um')
    #ax.set_ylabel('y, um')
    ax.set_zlabel('z, nm')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot_surface(X, Y, z)
    return ax


# Data operations (flat)
def load_flatdata(filename='data.hdf5'):
    data = h5py.File(filename, "r")
    data_in = data['train/in']
    data_out = data['train/out']
    
    X = np.array([])
    Y = np.array([])
    sizes = []

    for num in data_in.keys():
        x = np.array(data_in[num])
        y = np.array(data_out[num])
        size = x.shape[:-1]
        x = x.reshape((-1,3))
        y = y.reshape((-1,1))
        assert x.shape[0] == y.shape[0], f'Shapes are different for dataset#{num}.'
        sizes.append(size)
        
        if X.size == 0:
            X = x
        else:
            X = np.append(X, x, axis=0)
        
        if Y.size == 0:
            Y = y
        else:
            Y = np.append(Y, y, axis=0)
        
    data.close()
    return X, Y, sizes

def flatdata2optical(X, sizes):
    images = []
    for size in sizes:
        area = size[0]*size[1]
        x = X[:area]
        X = X[area:]
        x = x.reshape((*size,3))
        images.append(Image.fromarray(x, mode='RGB'))    
    return images
    
def flatdata2topography(Y, sizes, rescale=False, to_img=False):
    out = []
    for size in sizes:
        area = size[0]*size[1]
        y = Y[:area]
        Y = Y[area:]
        y = y.reshape((*size,))
        y_max = y.max()
        if y_max > 255:
            print(f'Max height is {y_max}!')
        if rescale:
            y = y*255/y_max
        if to_img:
            y = Image.fromarray(y.astype('uint8'), mode='L') 
        out.append(y)    
    return out


# Data operations (images)
def load_data_a(filename='data.hdf5', size=(256,256), 
              scale_x=False, scale_y=False):
    """
    Reads hdf5 datafile and rescales all images to a given size.
    """
    data = h5py.File(filename, "r")
    data_in = data['train/in']
    data_out = data['train/out']
    m = len(data_in)
    keys = list(data_in.keys())
    assert len(data_in) == len(data_out),\
    'Number of input images does not match number of output images.'

    X = np.empty((m, *size, 3), dtype='uint8')
    Y = np.empty((m, *size,), dtype='float32')

    for i in range(m):
        key = keys[i]
        x = np.array(data_in[key])
        y = np.array(data_out[key])
        assert x.shape[:2] == y.shape[:2],\
        f'Shapes are different for dataset#{key}.'
        x = cv2.resize(x, dsize=size, interpolation=cv2.INTER_CUBIC)
        y = cv2.resize(y, dsize=size, interpolation=cv2.INTER_CUBIC)

        X[i] = x
        Y[i] = y

    data.close()
    if scale_x:
        X = X.astype(np.float32)/255
    if scale_y:
        Y = Y/100
    
    return X, Y

def show_data_a(X, Y, i=None):
    assert len(X) == len(Y), f'len(X)={len(X)} != len(Y)={len(Y)}.'
    if i is None:
        i = np.random.randint(len(X))
        
    fig, axes = plt.subplots(nrows=1, ncols=2)
    im0 = axes[0].imshow(X[i])
    im = axes[1].imshow(Y[i])

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.3, 0.05, 0.4])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()
    return fig, axes

def plot_history(history):
    fig, axes = plt.subplots(nrows=2, ncols=1)
    keys = list(history.history.keys())
    im0 = axes[0].plot(history.history[keys[0]][2:])
    im = axes[1].plot(history.history[keys[1]][2:])

    axes[1].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[1].set_ylabel('error')
    plt.show()
    
def load_data(filename='model/data.hdf5', size=(256,256)):
    """
    Reads hdf5 datafile and rescales all images to a given size.
    """
    try:
        data = h5py.File(filename, "r")
    except:
        printf("Couldn't load data.")
        return
    data_in = data['train/in']
    data_out = data['train/out']
    data_bckg = data['train/background']
    m = len(data_in)
    keys = list(data_in.keys())
    assert len(data_in) == len(data_out),\
    'Number of input images does not match number of output images.'
    assert len(data_in) == len(data_bckg),\
    'Number of input images does not match number of background images.'

    X = np.empty((m, *size, 3), dtype='uint8')
    B = np.empty((m, *size, 3), dtype='uint8')
    Y = np.empty((m, *size,), dtype='float32')

    for i in range(m):
        key = keys[i]
        x = np.array(data_in[key])
        b = np.array(data_bckg[key])
        y = np.array(data_out[key])
        assert x.shape[:2] == y.shape[:2],\
        f'Shapes are different for dataset#{key}.'
        x = cv2.resize(x, dsize=size, interpolation=cv2.INTER_CUBIC)
        b = cv2.resize(b, dsize=size, interpolation=cv2.INTER_CUBIC)
        y = cv2.resize(y, dsize=size, interpolation=cv2.INTER_CUBIC)

        X[i] = x
        B[i] = b
        Y[i] = y

    data.close()
    
    X = X.astype(np.float32)/255
    B = B.astype(np.float32)/255
    Y = Y/100
    
    return X, B, Y

def show_data(X, B, Y, i=None):
    assert len(X) == len(Y), f'len(X)={len(X)} != len(Y)={len(Y)}.'
    if i is None:
        i = np.random.randint(len(X))
        
    fig, axes = plt.subplots(nrows=1, ncols=3)
    im0 = axes[0].imshow(X[i])
    im1 = axes[1].imshow(B[i])
    im = axes[2].imshow(Y[i])

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.33, 0.02, 0.3])
    fig.colorbar(im, cax=cbar_ax)
    
    for i in range(3):
        axes[i].set_axis_off()
    
    plt.show()
    return fig, axes
    
    
    
# Predict using saved model weights
def make_model(shape=(256,256,6)):
    model_name = 'cv2d_v1'
    inputs = tf.keras.Input(shape=shape)
    conv = tf.keras.layers.Conv2D(12, 3, padding='same', activation='relu')(inputs)
    conv = tf.keras.layers.Conv2D(6, 1, padding='same', activation='relu')(conv)
    conv = tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')(conv)
    model = tf.keras.Model(inputs=inputs, outputs=conv, name=model_name)
    if os.path.exists('model/'+model_name+'.index'):
        model.load_weights('model/'+model_name).expect_partial()
    else:
        print("Couldn't locate file with model parameters.")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])
    return model

def make_background(color, shape=(256,256,3)):
    return np.full(shape, np.array(color))

def predict(img='input/img1.jpeg', bkg='input/bkg1.jpeg', size=None, model=None):
    if type(img) == str:
        img = Image.open(img, 'r')
        img = np.array(img)
    try:
        img = np.array(img)
    except:
        print("Wrong imput data type.")
        return
    if size is None:
        side = min(img.shape[0:2])
        size = (side,side)
    
    if type(bkg) == str:
        bkg = np.array(Image.open(bkg, 'r'))
    if type(bkg) == list:
        bkg = make_background(bkg, shape=img.shape)  
    try:
        bkg = np.array(bkg)
    except:
        print("Wrong imput data type.")
        return
            
    img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_CUBIC)
    bkg = cv2.resize(bkg.astype(img.dtype), dsize=size, interpolation=cv2.INTER_CUBIC)
    X = np.concatenate((img, bkg), axis=-1)
    X = X.astype(np.float32)/255

    if model is None:
        model = make_model()

    out = np.array(model(X.reshape((1,size[0],size[1],6)))).reshape(size)
    
    return img, bkg, out

def plot_data(img, out):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    im0 = axes[0].imshow(img)
    im = axes[1].imshow(out)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.3, 0.02, 0.4])
    fig.colorbar(im, cax=cbar_ax)
    
    for i in range(2):
        axes[i].set_axis_off()
    
    return fig, axes

def make_foldername():
    if not os.path.exists('output'):
        os.makedirs('output')
        return 'output'
    i = 1
    while os.path.exists('output'+repr(i)):
        i = i+1
    os.makedirs('output'+repr(i))
    return 'output'+repr(i)
        

if __name__ == "__main__":
    if len(sys.argv) == 3:
        img = sys.argv[1]
        bkg = sys.argv[2]
        if bkg[0] == '[':
            bkg = ast.literal_eval(bkg)
        img, bkg, out = predict(img, bkg)
    else:
        print("Running algorithm using test images.")
        img, bkg, out = predict()
        
    foldername = make_foldername();
    
    fig, axes = plot_data(img, 100*out)
    plt.savefig(foldername+"/colorbar.jpeg")
    
    axes = plot3d(out*100)
    plt.savefig(foldername+"/3d.jpeg")
    
    min_value = np.min(out)
    max_value = np.max(out)
    Image.fromarray(np.uint8((out-min_value)*255/(max_value-min_value))).save(foldername+'/grayscale.tiff')
    
    Image.fromarray(img).save(foldername+'/img.jpeg')
    Image.fromarray(bkg).save(foldername+'/bkg.jpeg')
    
    print(f'Range: {round(100*min_value,3), round(100*max_value,3)} nm.')
    print(f'Z-scale (gwyddion): {round((max_value-min_value)*100/255,5)} x 10e-9.')
    print(f'Saved results to {foldername}.')