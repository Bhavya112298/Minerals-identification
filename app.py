
import tensorflow
 
from tensorflow.keras.models import Sequential
 
from tensorflow.keras.layers import Convolution2D
 
from tensorflow.keras.layers import MaxPooling2D
 
from tensorflow.keras.layers import Flatten
 
from tensorflow.keras.layers import Dense
 
from tensorflow.keras.models import model_from_json
 
#note: here no data preprocessing as it is manually done by #using directory structure
 
 
# generate image augmentation to avoids overfitting 
 
# Use ImagedataGenerator to create lot of batches
 
# Gives random transformations (like flipped, rotated, etc)
 
 
batch_size = 5 # no. of images allowed to algorithm
 
 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
 
 
train_datagen = ImageDataGenerator(rescale=1/255) 
 
# images rescaled by 1./255
 
 
 
# Flow training images in batches  using train_datagen generator
 
train_generator = train_datagen.flow_from_directory(
 
        r'C:\Users\canara\Desktop\minerals_dataset\train_set',  # This is the source directory for training images
 
        target_size=(200, 200),#images are resized to 200 x 200
 
        batch_size=batch_size,
 
        # mention the classes
 
        classes = ['calcite','diamond','dioptase','magnetite','quartz','ruby','sulfur'],
 
        # categorical mode because many classes
 
        class_mode='categorical')
 
 
# building cnn model
 
model = tensorflow.keras.models.Sequential([
 
        # The first convolution & maxPooling
 
        # 32 and (3,3) means 32 feature detectors will be used and each filter will be 3rows- 3cols which will give rise to 32 feature maps 
 
 
    tensorflow.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
 
    tensorflow.keras.layers.MaxPooling2D(2, 2),
 
    # The second convolution & maxpooling
 
    tensorflow.keras.layers.Conv2D(32, (3,3), activation='relu'),
 
    tensorflow.keras.layers.MaxPooling2D(2,2),
 
    # The third convolution & maxpooling
 
    tensorflow.keras.layers.Conv2D(64, (3,3), activation='relu'),
 
    tensorflow.keras.layers.MaxPooling2D(2,2),
 
    # The fourth convolution & Maxpooling
 
    tensorflow.keras.layers.Conv2D(64, (3,3), activation='relu'),
 
    tensorflow.keras.layers.MaxPooling2D(2,2),
 
    # The fifth convolution & maxpooling
 
    tensorflow.keras.layers.Conv2D(64, (3,3), activation='relu'),
 
    tensorflow.keras.layers.MaxPooling2D(2,2),
 
    # Flatten the results to feed into ann
 
   tensorflow.keras.layers.Flatten(),
 
    # 128 neuron in the fully-connected layer
 
    tensorflow.keras.layers.Dense(128, activation='relu'),
 
    # 7 output neurons for 7 classes with the softmax activation
 
   tensorflow.keras.layers.Dense(7, activation='softmax')
 
])
 
 
 
# Compiling the CNN
 
from tensorflow.keras.optimizers import RMSprop
 
 
model.compile(loss='categorical_crossentropy',
 
              optimizer=RMSprop(lr=0.001),
 
              metrics=['acc'])
 
 
total_sample=train_generator.n # total images for training
 
 
 
n_epochs = 5 # 1 epoch means one complete dataset
 
 
# fit generator using training set
 
# define number of iterations per epoch
 
history = model.fit_generator(
 
        train_generator, 
 
        steps_per_epoch=int(total_sample/batch_size),  
 
        epochs=n_epochs,
 
        verbose=1)
 
 
# predicting class of new image
 
import numpy as np
 
from tensorflow.keras.preprocessing import image
 
 
 
def predic_mineral(fname):
    test_image = image.load_img(fname, target_size = (200, 200))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    return result





#flask
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)
#app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)
        result = "testing.."
        res= predic_mineral(f.filename)
       
        if res[0][0] == 1:
            result="CALCITE"

        elif res[0][1] == 1:
            result ="DIAMOND"

        elif res[0][2] == 1:
            result="DIOPTASE"

        elif res[0][3] == 1:
            result = 'MAGNETITE'

        elif res[0][4] ==1:
            result ='QUARTZ'

        elif res[0][5] ==1:
            result ='RUBY'

        elif res[0][6] ==1:
            result='SULFUR'
          
        return render_template("success.html", name = result)
    


app.run()    

