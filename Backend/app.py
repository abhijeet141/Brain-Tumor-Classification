from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from keras.models import load_model
from keras.utils import load_img, img_to_array
from io import BytesIO

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = load_model('./vgg19_model.h5')
@app.get('/')
def welcome():
    return {
        'success': True,
        'message': 'server of "brain tumor classification using 4 classes" is up and running successfully.'
    }
@app.post('/predict')
async def predict_disease(fileUploadedByUser: UploadFile = File(...)):
    contents = await fileUploadedByUser.read()
    imageOfUser = load_img(BytesIO(contents), target_size=(224, 224))
    image_to_arr = img_to_array(imageOfUser)
    image_to_arr_preprocess_input = image_to_arr/255.0
    image_to_arr_preprocess_input_expand_dims = np.expand_dims(image_to_arr_preprocess_input, axis=0)
    prediction = model.predict(image_to_arr_preprocess_input_expand_dims)
    class_names = ['glioma', 'meningioma','notumor','pituitary']
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    confidence = np.max(prediction) * 100
    print(predicted_class)
    return {
        'success': True,
        'predicted_result': predicted_class,
        'confidence': f'{confidence:.2f}%',
        'message': f'Status of the Brain Image: {predicted_class} with a confidence of {confidence:.2f}%'
    }