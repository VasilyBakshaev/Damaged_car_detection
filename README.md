## 🚗 Damaged car detection app
### Description
An application that determines the probability that a damaged car is depicted in a photo. 
The `resnet18` model was used as a basis and fine-tuned on examples of damaged and undamaged cars from the internet. 
The model was prepared for deployment as a streamlit application and as a bot for the popular messenger Telegram.

Main files:

- `damaged_car_detection.ipynb` - file with the development and training process of the model
- `model.pkl` - file with the model in `.pkl` format
- `streamlit_app.py` - streamlit application
- `telegram_app.py` - Telegram bot application

You can interact with deployed model [on streamlit 👾](https://damaged-car.streamlit.app/)

### App work example
|streamlit|telegram|
|--|--|
|![Work example](./images/st_gif.gif)|![Work example](./images/tg_gif.gif)|