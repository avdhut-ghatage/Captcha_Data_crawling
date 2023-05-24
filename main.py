from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time
from bs4 import BeautifulSoup


driver= webdriver.Chrome()
driver.get("https://tnreginet.gov.in/portal/webHP?requestType=ApplicationRH&actionVal=homePage&screenId=114&UserLocaleID=en&_csrf=0cab44e9-38ad-4ec3-9de6-5b97acf94a97")

def elem(xpath):
    ele = driver.find_element(By.XPATH,xpath)
    ele.click()

elem('//*[@id="8500009"]/a')
elem('//*[@id="8400001"]/a')
elem('//*[@id="8400010"]/a')
time.sleep(2)

elem('//*[@id="DOC_WISE"]')
time.sleep(2)

def dropdown(xpath,option):
    drop = driver.find_element(By.XPATH,xpath)
    select = Select(drop)
    select.select_by_visible_text(option)

dropdown('//*[@id="cmb_SroName"]','Chennai Central Joint I')
dropdown('//*[@id="cmb_Year"]','2023')
dropdown('//*[@id="cmb_doc_type"]','Regular Document')

time.sleep(5)



#dropdown.click()
#driver.close()
#driver.quit() 