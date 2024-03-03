import numpy as np
import tensorflow as tf
from tensorflow.keras import models,layers,datasets
from tensorflow.keras.preprocessing import image
import numpy
#cifar10 is a part of datasets, just like mnist.datasets come from keras.keras is main library
(X_train,Y_train),(X_test,Y_test)=datasets.cifar10.load_data()
X_train=X_train/255
X_test=X_test/255
# #there are 4 arguments in layers.Conv2D(1st is number of filters(only 32),2nd is (3,3) size of filter,3rd is activation="relu,4th is input_shape
#layers.Conv2D selects/highlight but it doesnt drop unimportant the most important features, it has 32 filters, and each filter has size of 3,3, the input shape(32,32) shrinks the image to 32,32 size, and 3 means it has 3 channels (red,blue,green).black and white has only 1 channel
#layers.MaxPooling2D filters the more important and drops the rest of unimportant ones.(()), it has 2 brackets because it wants 1 argument, and (2,2) is just the size of filter
model=models.Sequential([layers.Conv2D(32,(3,3),activation="relu",input_shape=(32,32,3)),layers.MaxPooling2D((2,2)),layers.Conv2D(64,(3,3),activation="relu"),layers.MaxPooling2D((2,2)),layers.Conv2D(64,(3,3),activation="relu"),layers.Flatten(),layers.Dense(128,activation="relu"),layers.Dense(10)])
model.compile(optimizer="adam",loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=["accuracy"])
model.fit(X_train,Y_train,epochs=10)
model.save("cifar_10.keras")
test_loss,test_accuracy=model.evaluate(X_test,Y_test,verbose=2)
print(test_loss,test_accuracy)
list_images=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
loaded_model=models.load_model("cifar_10.keras")
image_variable=image.load_img("ciphar10.jpg",target_size=(32,32))
image_array=image.img_to_array(image_variable)
expand_dimenstions=np.expand_dims(image_array,axis=0)
prediction=loaded_model.predict(expand_dimenstions)
predict_digit=np.argmax(prediction)
predict_image=list_images[predict_digit]
print(predict_image)