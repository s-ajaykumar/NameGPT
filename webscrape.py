from selenium import webdriver
from selenium.webdriver.common.by import By


driver = webdriver.Chrome()

boy_names, girl_names = [], []

for i in range(1, 239):
    url = f"https://www.looktamil.com/babynames/show/boy-names-{i}"
    driver.get(url)
    spans = driver.find_elements(By.CSS_SELECTOR, "span.fw-500.babyname-color-M")
    for span in spans:
        boy_names.append(span.text)
 
for i in range(1, 289):
    url = f"https://www.looktamil.com/babynames/show/girl-names-{i}"
    driver.get(url)
    spans = driver.find_elements(By.CSS_SELECTOR, "span.fw-500.babyname-color-F")
    for span in spans:
        girl_names.append(span.text)

driver.quit()

with open("Dataset/boy_names.txt", "w") as f:
    for name in boy_names:
        f.write('~' + name + '.' + "\n")

with open("Dataset/girl_names.txt", "w") as f:
    for name in girl_names:
        f.write('!' + name + '.' + "\n")
       
with open("Dataset/boy_names.txt", "r") as f1, open("Dataset/girl_names.txt", "r") as f2, open("Dataset/names.txt", "w") as f3:
    f3.write(f1.read())
    f3.write(f2.read())
