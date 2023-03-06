# pip install telebot
import telebot

import numpy as np

from io import BytesIO
# pip install pillow
from PIL import Image

# pip install fastai
from fastai.vision.all import *


bot_token = '-------------YOUR_TOKEN-------------'
bot = telebot.TeleBot(bot_token, parse_mode='html')

model = load_learner('model.pkl')


def get_image_from_bytes(b):
    '''
    Gets the PIL.Image object from the bytes of photo
    :param b: Bytes of the photo
    :type b: bytes
    '''
    stream = BytesIO(b)
    image = Image.open(stream).convert("RGB")
    stream.close()
    return image


def classify_image(img):
    img = get_image_from_bytes(img)
    categories = ('Car without damage', 'Damaged car')
    pred, idx, probs = model.predict(img)
    return dict(zip(categories, map(float, probs)))



@bot.message_handler(commands=['start'])
def send_welcome(message):
    send = bot.send_message(message.from_user.id, '😃Hello! Please send me a photo of the car so I can determine if it is damaged or not.')
#    bot.register_next_step_handler(send, get_user_pics)

@bot.message_handler(content_types=['text', 'file'])
def send_error(message):
    send = bot.send_message(message.from_user.id, '🙁Invalid input, send a photo of the car, please')

@bot.message_handler(content_types=['photo'])
def get_user_pics(message):
# Получаем id фотографии в Telegram
    photo_id = message.photo[-1].file_id
# Достаём картинку
    photo_file = bot.get_file(photo_id) # <class 'telebot.types.File'>
    photo_bytes = bot.download_file(photo_file.file_path) # <class 'bytes'>
# Отправить в дальнейшем можно таким образом
#    send = bot.send_photo(message.from_user.id, photo=photo_bytes)
    send = bot.send_message(message.from_user.id, f'💥The probability its damaged car is {round(classify_image(photo_bytes).get("Damaged car")*100, 2)}% \n'
                                                  f'🚗The probability its car without damage is {round(classify_image(photo_bytes).get("Car without damage")*100, 2)}%')

bot.polling()