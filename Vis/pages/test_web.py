import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import chromedriver_autoinstaller
import warnings

# Auto install ChromeDriver
chromedriver_autoinstaller.install()

# Set up Chrome options for headless mode (optional)
options = Options()
options.headless = True  # Run in headless mode (without opening a window)

# Create a Service object for ChromeDriver
service = Service()

# Initialize the WebDriver
driver = webdriver.Chrome(service=service, options=options)

# Streamlit application
st.title("Selenium Web Scraper")

# Button to trigger the scraping action
if st.button("Fetch Data"):
    try:
        driver.get("https://www.whoscored.com/Matches/1821249/Live/England-Premier-League-2024-2025-Bournemouth-Manchester-City")
        page_source = driver.page_source
        
        # Display the page source in Streamlit
        st.code(page_source)
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        driver.quit()  # Close the driver after scraping

