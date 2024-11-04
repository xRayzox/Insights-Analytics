from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

import streamlit as st
import chromedriver_autoinstaller

chromedriver_autoinstaller.install()

import warnings
warnings.filterwarnings('ignore')
driver = webdriver.Chrome()

driver.get("https://www.whoscored.com/Matches/1821249/Live/England-Premier-League-2024-2025-Bournemouth-Manchester-City")

st.code(driver.page_source)