import os
import json
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime as dt

import numpy as np
import cv2
from PIL import Image
import telebot
from telebot import types

from inference import convert_from_json_response
from utils import plot_color_gradients,\
    KEY_MODEL, KEY_APPLY_MASK, KEY_PALETTE, \
    VALUE_STABLE, VALUE_EXPERIMENTAL, VALUE_YES, VALUE_NO, \
    DEFAULT_MODEL, DEFAULT_APPLY_MASK, DEFAULT_CMAP, \
    CMAP_CIVIDIS, CMAP_VIRIDIS, CMAP_INFERNO, CMAP_MAGMA, CMAP_PURD, \
    CMAP_HOT, CMAP_AFMHOT, CMAP_GISTHEAT, CMAP_COOLWARM, CMAP_BWR, CMAP_JET, ALL_CMAPS


def create_app_params():
    with open('credentials.json', 'r') as fp:
        d = json.load(fp)

    token = d["token"]
    server_url = d["server_url"]
    username = d["username"]
    password = d["password"]

    if username is not None:
        auth = HTTPBasicAuth(username=username, password=password)
    else:
        print('\nNo authentification')
        auth = None

    return token, server_url, auth


TOKEN, SERVER_URL, AUTH = create_app_params()


bot = telebot.TeleBot(TOKEN, parse_mode=None) # You can set parse_mode by default. HTML or MARKDOWN


SHOULD_REMOVE_TEMP_IMAGES = True
IMAGES_DIR = 'bot_temp_images'
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR, exist_ok=True)


COLORMAPS_FNAME = 'colormaps.jpeg'
if not os.path.exists(COLORMAPS_FNAME):
    print(f'Created colormaps image {COLORMAPS_FNAME}')
    plot_color_gradients(ALL_CMAPS, fname=COLORMAPS_FNAME)


ADV_RUN_KEY = 'adv_run'
SETTINGS = {}


# ---------- Markups ----------

def generate_model_type_markup():
    markup_model_type = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    button_model_type1 = types.KeyboardButton(VALUE_STABLE)
    button_model_type2 = types.KeyboardButton(VALUE_EXPERIMENTAL)
    markup_model_type.add(button_model_type1, button_model_type2)
    return markup_model_type


def generate_face_region_markup():
    markup_apply_mask = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    button_apply_mask_true = types.KeyboardButton(VALUE_YES)
    button_apply_mask_false = types.KeyboardButton(VALUE_NO)
    markup_apply_mask.add(button_apply_mask_true, button_apply_mask_false)
    return markup_apply_mask


def generate_palettes_markup():
    markup_palette = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    markup_palette.row(CMAP_CIVIDIS, CMAP_VIRIDIS, CMAP_INFERNO, CMAP_MAGMA)
    markup_palette.row(CMAP_PURD, CMAP_HOT, CMAP_AFMHOT, CMAP_GISTHEAT)
    markup_palette.row(CMAP_COOLWARM, CMAP_BWR, CMAP_JET)
    return markup_palette


# ---------- Init commands ----------


@bot.message_handler(commands=['start'])
def send_welcome(message):
    start_message = 'Hello, I can tell you how beautiful a face is.\n' \
                    'To start, just send an image with one or more faces.\n' \
                    'Please, keep in mind that it works best person looks directly into the camera'
    bot.send_message(message.chat.id, start_message)


@bot.message_handler(commands=['help'])
def send_help(message):
    help_message = f"List of commands:\n" \
                   f"/start — start messaging\n" \
                   f"/help — see help message\n" \
                   f"/adv_run — run model in advanced mode\n" \
                   f"/set_model — set model type\n" \
                   f"/set_face_region — set face region for heatmap\n" \
                   f"/set_palette — set palette for heatmap\n" \
                   f"/palettes — show available palettes for heatmaps\n" \
                   f"/settings — show current settings\n" \
                   f"/reset — reset all settings\n\n" \
                   f"To process image just send it. If you would like more control " \
                   f"(like trying different models) first run /adv_run command (note that settings are not saved)\n\n" \
                   f"A model was trained mostly on faces, " \
                   f"while people also pay attention to hair and clothes, so results may be quite rough.\n" \
                   f"And always keep in mind that the real beauty is not what you see in a photo, " \
                   f"but what you have inside"
    bot.send_message(message.chat.id, help_message)


# ---------- The most important part ----------

def create_time_postfix():
    return dt.now().strftime('d%Y%m%d_t%H%M%S')


def save_image(message):
    cid = message.chat.id
    fid = message.photo[-1].file_id

    file_info = bot.get_file(fid)
    downloaded_file = bot.download_file(file_info.file_path)

    time_postfix = create_time_postfix()
    image_path = os.path.join(IMAGES_DIR, f'upload_{cid}_{time_postfix}_{fid}.jpeg')
    with open(image_path, 'wb') as fp:
        fp.write(downloaded_file)

    return image_path


def prepare_image(p):
    image = np.array(Image.open(p))
    success, encoded_image = cv2.imencode('.jpeg', image)
    bytes_image = encoded_image.tobytes()
    return bytes_image


def send_image_for_processing(data, postfixes=None):
    url = SERVER_URL
    if postfixes is not None:
        for p in postfixes:
            url += f'/{p}'
    r = requests.post(url, data=data, headers={'Content-Type': 'application/octet-stream'}, auth=AUTH)
    return r


def remove_files(paths):
    for p in paths:
        # Don't print anything which might be saved in logs but should not be
        # print(f'Removing file {p}...')
        os.remove(p)


@bot.message_handler(content_types=['photo'])
def handle_photos(message):
    temp_paths = []

    global SETTINGS
    cid = message.chat.id
    if cid in SETTINGS.keys():
        if ADV_RUN_KEY in SETTINGS[cid].keys():
            bot.send_message(cid, 'Note that settings for models are not saved')
            postfixes = SETTINGS[cid][KEY_MODEL], SETTINGS[cid][KEY_APPLY_MASK], SETTINGS[cid][KEY_PALETTE]
        else:
            model = SETTINGS[cid].get(KEY_MODEL, DEFAULT_MODEL)
            apply_mask = SETTINGS[cid].get(KEY_APPLY_MASK, DEFAULT_APPLY_MASK)
            palette = SETTINGS[cid].get(KEY_PALETTE, DEFAULT_CMAP)
            postfixes = model, apply_mask, palette
    else:
        postfixes = DEFAULT_MODEL, DEFAULT_APPLY_MASK, DEFAULT_CMAP

    image_path = save_image(message)
    temp_paths.append(image_path)
    image_data = prepare_image(image_path)
    r = send_image_for_processing(image_data, postfixes)

    # Reset params
    if cid in SETTINGS.keys():
        if ADV_RUN_KEY in SETTINGS[cid].keys():
            SETTINGS.pop(cid, None)

    bot.reply_to(message, 'Processing uploaded photo...')

    if r.status_code == 200:
        bot.send_message(cid, 'Photo processed. Preparing output images...')
        r_outputs = convert_from_json_response(r)
        for idx, out in enumerate(r_outputs):
            time_postfix = create_time_postfix()
            out_image_path = os.path.join(IMAGES_DIR, f'out_{idx}_{cid}_{time_postfix}.jpeg')
            temp_paths.append(out_image_path)
            # out[0] is a np.array with dtype=uint8
            Image.fromarray(out[0]).save(out_image_path)
            with open(out_image_path, 'rb') as out_image:
                bot.send_photo(cid, out_image, caption=out[1])
    else:
        bot.reply_to(message, 'Sorry, something went wrong. Maybe try another photo?')

    if SHOULD_REMOVE_TEMP_IMAGES:
        remove_files(list(set(temp_paths)))


# ---------- Advanced processing of images ----------

@bot.message_handler(commands=['adv_run'])
def choose_model_type(message):
    # First select model type
    markup_model_type = generate_model_type_markup()
    message = bot.send_message(message.chat.id, 'Select model type:', reply_markup=markup_model_type)
    bot.register_next_step_handler(message, choose_mask_option)


@bot.callback_query_handler(func=lambda c: True)
def choose_mask_option(message):
    global SETTINGS
    cid = message.chat.id
    SETTINGS[cid] = {KEY_MODEL: message.text}
    # Second choose if heatmap should only be applied to face
    face_region_markup = generate_face_region_markup()
    message = bot.send_message(cid, 'Apply heatmap only to face?', reply_markup=face_region_markup)
    bot.register_next_step_handler(message, choose_palette)


@bot.callback_query_handler(func=lambda c: True)
def choose_palette(message):
    global SETTINGS
    cid = message.chat.id
    SETTINGS[cid][KEY_APPLY_MASK] = message.text
    # Third choose palettes for heatmap
    palettes_markup = generate_palettes_markup()
    message = bot.send_message(cid, 'Select palette for heatmap:', reply_markup=palettes_markup)
    bot.register_next_step_handler(message, ask_to_send_image)


def ask_to_send_image(message):
    cid = message.chat.id
    global SETTINGS
    SETTINGS[cid][KEY_PALETTE] = message.text
    SETTINGS[cid][ADV_RUN_KEY] = True
    bot.send_message(cid, 'Now upload image with faces')


# ---------- Small and useful commands ----------

@bot.message_handler(commands=['palettes'])
def show_palettes(message):
    with open(COLORMAPS_FNAME, 'rb') as image:
        bot.send_photo(message.chat.id, image)


@bot.message_handler(commands=['set_model'])
def set_model(message):
    model_type_markup = generate_model_type_markup()
    message = bot.send_message(message.chat.id, 'Select model type:', reply_markup=model_type_markup)
    bot.register_next_step_handler(message, set_model_handler)


@bot.callback_query_handler(func=lambda c: True)
def set_model_handler(message):
    global SETTINGS
    cid = message.chat.id
    if cid not in SETTINGS.keys():
        SETTINGS[cid] = {}
    SETTINGS[cid][KEY_MODEL] = message.text
    bot.send_message(cid, f"Model set to '{message.text}'")
    print(SETTINGS)


@bot.message_handler(commands=['set_face_region'])
def set_face_region(message):
    face_region_markup = generate_face_region_markup()
    message = bot.send_message(message.chat.id, 'Apply heatmap only to face?', reply_markup=face_region_markup)
    bot.register_next_step_handler(message, set_face_region_handler)


@bot.callback_query_handler(func=lambda c: True)
def set_face_region_handler(message):
    global SETTINGS
    cid = message.chat.id
    if cid not in SETTINGS.keys():
        SETTINGS[cid] = {}
    SETTINGS[cid][KEY_APPLY_MASK] = message.text
    bot.send_message(cid, f"Apply heatmap only to face set to '{message.text}'")


@bot.message_handler(commands=['set_palette'])
def set_palette(message):
    palettes_markup = generate_palettes_markup()
    message = bot.send_message(message.chat.id, 'Select palette for heatmap:', reply_markup=palettes_markup)
    bot.register_next_step_handler(message, set_palette_handler)


@bot.callback_query_handler(func=lambda c: True)
def set_palette_handler(message):
    global SETTINGS
    cid = message.chat.id
    if cid not in SETTINGS.keys():
        SETTINGS[cid] = {}
    SETTINGS[cid][KEY_PALETTE] = message.text
    bot.send_message(cid, f"Palette for heatmap set to '{message.text}'")


@bot.message_handler(commands=['settings'])
def show_settings(message):
    cid = message.chat.id

    def_dict = {
        KEY_MODEL: DEFAULT_MODEL,
        KEY_APPLY_MASK: DEFAULT_APPLY_MASK,
        KEY_PALETTE: DEFAULT_CMAP
    }
    text_dict = {
        KEY_MODEL: 'Model: ',
        KEY_APPLY_MASK: 'Apply heatmap only to face? ',
        KEY_PALETTE: 'Heatmap palette: '
    }

    out_text = ''
    for k in [KEY_MODEL, KEY_APPLY_MASK, KEY_PALETTE]:
        if cid in SETTINGS.keys():
            v = SETTINGS[cid].get(k, def_dict[k])
        else:
            v = def_dict[k]
        out_text += text_dict[k] + v + '\n'

    bot.send_message(cid, out_text)


@bot.message_handler(commands=['reset'])
def set_palette(message):
    global SETTINGS
    cid = message.chat.id
    if cid in SETTINGS.keys():
        SETTINGS.pop(cid, None)
    bot.send_message(cid, 'SAll settings have been reset to default values')


# ---------- Other types of messages ----------

@bot.message_handler(content_types=['document'])
def handle_docs(message):
    bot.reply_to(message, 'Sorry, but documents are not supported')


@bot.message_handler(content_types=['audio'])
def handle_audio(message):
    bot.reply_to(message, 'Sorry, but audio files are not supported')


@bot.message_handler(content_types=['video'])
def handle_video(message):
    video_message = 'Sorry, but video files are not supported. '\
                    'You can manually extract a frame and send it to me'
    bot.reply_to(message, video_message)


@bot.message_handler(content_types=['sticker'])
def handle_sticker(message):
    emoji_message = "Hope, it's a good sign :)"
    bot.reply_to(message, emoji_message)


GOOD_EMOJI = [
    '\xF0\x9F\x98\x82',
    '\xF0\x9F\x98\x83',
    '\xF0\x9F\x98\x84',
    '\xF0\x9F\x98\x86',
    '\xF0\x9F\x98\x89',
    '\xF0\x9F\x98\x8A',
    '\xF0\x9F\x98\x8C',
    '\xF0\x9F\x98\x8F',
    '\xF0\x9F\x98\x9C',
    '\xF0\x9F\x98\x9D'
]
GOOD_EMOJI = [x.encode('raw_unicode_escape') for x in GOOD_EMOJI]

@bot.message_handler(func=lambda msg: msg.text.encode('utf-8') in GOOD_EMOJI)
def handle_emoji(message):
    emoji_message = "Hope, it's a good sign :)"
    bot.reply_to(message, emoji_message)


@bot.message_handler(func=lambda m: True)
def echo_all(message):
    reply_message = "Sorry, but I don't understand you. Type /help to get information about how to use this bot"
    bot.reply_to(message, reply_message)


bot.polling()
