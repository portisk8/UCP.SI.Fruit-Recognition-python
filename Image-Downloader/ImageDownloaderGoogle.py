from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import requests
import urllib.request
import time
from datetime import datetime
import sys
import os


currentPath = os.getcwdb().decode("utf-8") 
folderImagePath = os.path.join(currentPath, 'ImagesDownloaded')
driver = webdriver.Chrome(executable_path = currentPath + '\\chromedriver.exe')

site = 'https://www.google.com/search?tbm=isch&q='+"\"banana+fruit\"+fruta"
scroller = 0
driver.get(site)
i = 0
while scroller<2:  
    driver.execute_script("window.scrollBy(0,document.body.scrollHeight)")
    try:
        driver.find_element(by=By.XPATH, value="/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div/div[5]/input").click()
        # driver.find_element_by_xpath("/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div/div[5]/input").click()
    except Exception as e:
        pass
    time.sleep(5)
    scroller+=1
soup = BeautifulSoup(driver.page_source, 'html.parser')
driver.close()

img_tags = soup.find_all("img", class_="rg_i")

image_no = 0
for i in img_tags:
    try:
        if image_no == 500:
            break
        nameImg = str(image_no)+'_'+datetime.now().strftime("%Y%m%d_%H%M%S")
        urllib.request.urlretrieve(i['src'], folderImagePath+'\\'+nameImg+".jpg")
        image_no+=1
        print("downloaded images = "+str(image_no),end='\r')
    except Exception as e:
        pass
