# Required imports
import streamlit as st
from fastai.vision.all import *
from io import BytesIO
from PIL import Image
import urllib.request as urllib2
from st_click_detector import click_detector

# Loading the trained model
model = load_learner('model.pkl')

# Definition of functions for prediction and image processing
def get_image_from_bytes(b):
    stream = BytesIO(b)
    image = Image.open(stream).convert("RGB")
    stream.close()
    return image

def classify_image(img):
    img = get_image_from_bytes(img)
    categories = ('Car without damage', 'Damaged car')
    pred, idx, probs = model.predict(img)
    return dict(zip(categories, map(float, probs)))

# Body of app
st.header('🚗 Damaged car detection app')
st.markdown('This is a damaged car detection app. '
            'You can upload a photo of your car and receive a probability that the car in your photo is damaged.')
# 'Upload a file' method
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    photo_bytes = uploaded_file.getvalue()
    st.image(photo_bytes, caption='Your uploaded image')
    st.success(f'💥 The probability its damaged car is {round(classify_image(photo_bytes).get("Damaged car")*100, 2)}% \n\n'
               f'🚗 The probability its car without damage is {round(classify_image(photo_bytes).get("Car without damage")*100, 2)}%')

# 'Enter a link' method
image_link = st.text_input('Or enter a link to an image', value='https://')
if image_link is not None:
    try:
        photo_bytes = urllib2.urlopen(image_link).read()
        st.image(photo_bytes, caption='Image from your link')
        st.success(
            f'💥 The probability its damaged car is {round(classify_image(photo_bytes).get("Damaged car") * 100, 2)}% \n\n'
            f'🚗 The probability its car without damage is {round(classify_image(photo_bytes).get("Car without damage") * 100, 2)}%')
    except:
        pass

# 'Click on the image' method
st.markdown("If you don't have a photo or link, click on one of the following images")
content = """
    <a href='#' id='https://images.theconversation.com/files/465292/original/file-20220525-22-ufebo7.jpg?ixlib=rb-1.1.0&q=45&auto=format&w=1200&h=1200.0&fit=crop'><img width=49%' 
    src='https://images.theconversation.com/files/465292/original/file-20220525-22-ufebo7.jpg?ixlib=rb-1.1.0&q=45&auto=format&w=1200&h=1200.0&fit=crop'></a>
    
    <a href='#' id='https://thumbsnap.com/sc/QKhXFyi2.jpg'><img width='49%'
     src='https://thumbsnap.com/sc/QKhXFyi2.jpg'></a>
     
    <a href='#' id='https://www.injurylawyerflorenceky.com/img/replacing-damaged-vehicle.jpg'><img width='49%' 
    src='https://www.injurylawyerflorenceky.com/img/replacing-damaged-vehicle.jpg'></a>
    
    <a href='#' id='https://carrentalcyprus.co.uk/wp-content/uploads/2016/01/image-bfbf7f92-980x550.jpg'><img width='49%' 
    src='https://carrentalcyprus.co.uk/wp-content/uploads/2016/01/image-bfbf7f92-980x550.jpg'></a>

    """
clicked = click_detector(content)

if clicked is not None:
    try:
        photo_bytes = urllib2.urlopen(clicked).read()
        st.success(
            f'💥 The probability its damaged car is {round(classify_image(photo_bytes).get("Damaged car") * 100, 2)}% \n\n'
            f'🚗 The probability its car without damage is {round(classify_image(photo_bytes).get("Car without damage") * 100, 2)}%')
    except:
        pass

# Final message
st.info('Thank you for interacting with this model. '
			 'You can find the source code on [my GitHub 👾](https://github.com/VasilyBakshaev/Damaged_car_detection)')