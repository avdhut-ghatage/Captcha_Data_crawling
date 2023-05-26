from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


driver= webdriver.Chrome()
driver.maximize_window()
driver.delete_all_cookies()
driver.implicitly_wait(5)
driver.get("https://tnreginet.gov.in/portal/webHP?requestType=ApplicationRH&actionVal=homePage&screenId=114&UserLocaleID=en&_csrf=0cab44e9-38ad-4ec3-9de6-5b97acf94a97")

def elem(xpath):
    ele = driver.find_element(By.XPATH,xpath)
    ele.click()

elem('//*[@id="8500009"]/a')
elem('//*[@id="8400001"]/a')
elem('//*[@id="8400010"]/a')

elem('//*[@id="DOC_WISE"]')
time.sleep(2)

def dropdown(xpath,option):
    drop = driver.find_element(By.XPATH,xpath)
    select = Select(drop)
    select.select_by_visible_text(option)

dropdown('//*[@id="cmb_SroName"]','Chennai Central Joint I')
dropdown('//*[@id="cmb_Year"]','2023')
dropdown('//*[@id="cmb_doc_type"]','Regular Document')

s= driver.find_element(By.XPATH,'//*[@id="captcha"]') # identifying the element to capture the screenshot
location = s.location # to get the element location
size = s.size # to get the dimension the element
driver.save_screenshot("screenshot_tutorialspoint.png") #to get the screenshot of complete page
x = location['x'] #to get the x axis
y = location['y'] #to get the y axis
height = location['y']+size['height'] # to get the length the element
width = location['x']+size['width']  # to get the width the element
imgOpen = Image.open("screenshot_tutorialspoint.png") # to open the captured image
imgOpen = imgOpen.crop((int(x), int(y), int(width), int(height)))  # to crop the captured image to size of that element
imgOpen.save("scraped.png")  # to save the cropped image

def prediction(file):
    characters = ['2', '3', '4', '5', '6', '7', '8', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'M', 'N', 'P', 'R', 'W', 'X', 'Y', 'Z']
    batch_size = 16
    img_width = 200
    img_height = 50
    max_length = 5

    char_to_num = layers.StringLookup(
        vocabulary=list(characters), mask_token=None
    )

    # Mapping integers back to original characters
    num_to_char = layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )

    def encode_single_sample(img_path, label):
        # 1. Read image
        img = tf.io.read_file(img_path)
        # 2. Decode and convert to grayscale
        img = tf.io.decode_png(img, channels=1)
        # 3. Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. Resize to the desired size
        img = tf.image.resize(img, [img_height, img_width])
        # 5. Transpose the image because we want the time
        # dimension to correspond to the width of the image.
        img = tf.transpose(img, perm=[1, 0, 2])

        # 7. Return a dict as our model is expecting two inputs
        return {"image": img }
    class CTCLayer(layers.Layer):
        def __init__(self, name=None):
            super().__init__(name=name)
            self.loss_fn = keras.backend.ctc_batch_cost

        def call(self, y_true, y_pred):
            # Compute the training-time loss value and add it
            # to the layer using `self.add_loss()`.
            batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
            input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
            label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

            input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
            label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

            loss = self.loss_fn(y_true, y_pred, input_length, label_length)
            self.add_loss(loss)

            # At test time, just return the computed predictions
            return y_pred

    custom_objects = {"CTCLayer": CTCLayer}

    reconstructed_model = keras.models.load_model("ocr_model_100_epoch.h5", custom_objects=custom_objects)

    def decode_batch_predictions(pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
            :, :max_length]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text
    
    test_img_path =[file]

    validation_dataset = tf.data.Dataset.from_tensor_slices((test_img_path[0:1], ['']))
    validation_dataset = (
        validation_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    for batch in validation_dataset.take(1):
        #print(batch['image'])
        
        preds = reconstructed_model.predict(batch['image']) # reconstructed_model is saved trained model
        pred_texts = decode_batch_predictions(preds)

    return pred_texts[0]

text = prediction('scraped.png')
inputElement = driver.find_element(By.XPATH,'//*[@id="txt_Captcha"]')
inputElement.send_keys(text)
inputElement = driver.find_element(By.XPATH,'//*[@id="txt_DocumentNo"]')
inputElement.send_keys('1')
inputElement = driver.find_element(By.XPATH,'//*[@id="btn_SearchDoc"]').click()
inputElement = driver.find_element(By.XPATH,'//*[@id="divGeneratePdf"]/h3/a/b').click()
inputElement = driver.find_element(By.XPATH,'//*[@id="successPage"]/div[2]/div/div[2]/h2[2]/a').click()
time.sleep(2)
driver.close()    #to close the browser