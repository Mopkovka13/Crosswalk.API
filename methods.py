import cv2
import numpy as np
import torch.nn.functional as F
import torch
from model import model
from model_type import model_type
from PIL import Image


def get_color(light_arr, thresh=0.09):
    h, w = light_arr.shape[:2]
    S = h * w

    # crop image
    h_cut = int(h * 0.1)
    w_cut = int(w * 0.1)
    if h_cut > 0 and w_cut > 0:
        light_arr = light_arr[h_cut:-h_cut, w_cut:-w_cut]
    elif h_cut < 0 and w_cut > 0:
        light_arr = light_arr[:, w_cut:-w_cut]
    elif h_cut > 0 and w_cut < 0:
        light_arr = light_arr[h_cut:-h_cut, :]

    # Image.fromarray(np.uint8(light_arr)).show()

    # convert to HSV format
    light_arr = cv2.cvtColor(light_arr, cv2.COLOR_RGB2HSV)

    # green range
    lower_green = (36, 40, 40)
    upper_green = (88, 255, 255)

    # red ranges
    lower_red_1 = (0, 40, 50)
    upper_red_1 = (20, 255, 255)
    lower_red_2 = (165, 40, 50)
    upper_red_2 = (180, 255, 255)

    # calculating green
    green_per = np.clip(cv2.inRange(light_arr, lower_green, upper_green), 0, 1).sum() / S

    # calculating red
    red_1 = cv2.inRange(light_arr, lower_red_1, upper_red_1)
    red_2 = cv2.inRange(light_arr, lower_red_2, upper_red_2)
    red_per = np.clip(red_1 + red_2, 0, 1).sum() / S
    if max(green_per, red_per) > thresh:
        if green_per > red_per:
            return 3
        else:
            return 4
    else:
        return 5


def detect(img_path):
    img = np.asarray(Image.open(img_path))

    # Детектирование объктов дорожной инфрастурктуры
    results = model(img)
    res_tech = results.pandas().xyxy[0]
    res_tech = res_tech[res_tech['class'] == 3]
    res = results.pandas().xywhn[0][['class', 'xcenter', 'ycenter', 'width', 'height']]

    # Выделение светофоров на фото и определение света
    for ind, vals in res_tech.iterrows():
        xmin, ymin, xmax, ymax = list(map(int, vals[:4].tolist()))
        light_arr = img[ymin:ymax, xmin:xmax]
        cls = get_color(light_arr)
        res.loc[ind, ['class']] = cls

    if 5 not in res['class'].tolist():
        return res


def get_result(img_path):
    res = detect(img_path)

    img_data = [0] * 25  # Создание пустого списка размера 25, по 5 критериев для 5 видов объектов
    for obj in res.iloc:  # Проходмися по каждой строке в таблице
        obj = list(obj)
        S = obj[3] * obj[4]  # Вычисление площади рамки объекта
        t = 5 * int(obj[0])  # Вычисление номера в списке с которого начинаются критерии объекта данного типа
        if S > img_data[
            t + 4]:  # Если площадь текущего объекта больше площади объекта этого же типа на этой же фотографии, то
            # Текущий объект записывается на место предыдущего
            img_data[t:t + 4] = obj[1:5]  # Записывание признаков
            img_data[t + 4] = S  # Записывание площади

    return int(F.softmax(model_type.forward(torch.tensor(img_data)), dim=-1).argmax())
