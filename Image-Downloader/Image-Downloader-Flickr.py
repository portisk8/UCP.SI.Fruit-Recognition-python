from matplotlib import style
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import requests
import urllib.request
import time
from datetime import datetime
import sys
import os

site = 'https://www.flickr.com/search/?text=banana%20fruit'
currentPath = os.getcwdb().decode("utf-8") 
folderImagePath = os.path.join(currentPath, 'ImagesDownloaded')
driver = webdriver.Chrome(executable_path = currentPath + '\\chromedriver.exe')
scroller = 0
driver.get(site)
i = 0
while scroller<4:  
    driver.execute_script("window.scrollBy(0,document.body.scrollHeight)")
    try:
        driver.find_element(by=By.XPATH, value="/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div/div[5]/input").click()
    except Exception as e:
        pass
    time.sleep(5)
    scroller+=1
soup = BeautifulSoup(driver.page_source, 'html.parser')
driver.close()
image_no = 0
for items in soup.select(".photo-list-photo-view"):
    if image_no == 500:
        break
    image= "https:" + items['style'].split("url(\"")[1].split("\");")[0]
    nameImg = str(image_no)+'_'+datetime.now().strftime("%Y%m%d_%H%M%S")
    urllib.request.urlretrieve(image,folderImagePath+'\\'+nameImg+".jpg")
    image_no+=1
    print("downloaded images = "+str(image_no),end='\r')