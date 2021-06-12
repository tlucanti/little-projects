from selenium import webdriver

browser = webdriver.Chrome()
browser.set_window_size(1000, 1000)
browser.get("http://google.com")
browser.execute_script("document.body.style.zoom='75%'")
