# scraper.py
import os
import time
import re
from tqdm import tqdm
import pandas as pd

# Selenium Imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, TimeoutException, ElementClickInterceptedException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scrape_and_save_reviews(app_name, app_id, country, review_count=200):
    """
    Scrapes reviews for a given app using Selenium.
    DEFINITIVE FINAL VERSION: Built with user-provided, ground-truth class names.
    """
    # --- 1. Setup ---
    sanitized_app_name = app_name.lower().replace(" ", "_")
    output_dir = os.path.join("reviews", sanitized_app_name)

    if os.path.exists(output_dir):
        print(f"Reviews directory for '{app_name}' already exists. Skipping scraping.")
        return output_dir

    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting Selenium scraper for '{app_name}'...")

    url = f"https://apps.apple.com/{country}/app/{sanitized_app_name}/id{app_id}?see-all=reviews"
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    driver.get(url)
    print(f"Browser opened. Navigating to App Store reviews page...")

    # --- 2. Load Review Batches ---
    print("Loading review batches...")
    while True:
        try:
            reviews_on_page = driver.find_elements(By.CSS_SELECTOR, "div.we-customer-review")
            if len(reviews_on_page) >= review_count:
                break
            show_more_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button.we-modal__show-more-button"))
            )
            driver.execute_script("arguments[0].click();", show_more_button)
            time.sleep(2)
        except (TimeoutException, NoSuchElementException):
            print("All review batches loaded.")
            break

    # --- 3. Extract Data by Handling Modals One-by-One ---
    review_elements = driver.find_elements(By.CSS_SELECTOR, "div.we-customer-review")
    all_reviews_data = []
    print(f"\nFound {len(review_elements)} reviews. Extracting data by opening each modal...")

    for i in tqdm(range(min(len(review_elements), review_count)), desc="Processing reviews"):
        try:
            current_review_card = driver.find_elements(By.CSS_SELECTOR, "div.we-customer-review")[i]

            # --- Step A: Find the '...more' button, scroll to it, and click ---
            try:
                more_button = current_review_card.find_element(By.CSS_SELECTOR, "button.we-truncate__button")
                driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", more_button)
                time.sleep(0.5)
                more_button.click()
            except NoSuchElementException:
                # This is a short review, extract directly from the card
                title = current_review_card.find_element(By.CSS_SELECTOR, "h3.we-customer-review__title").text.strip()
                rating_element = current_review_card.find_element(By.CSS_SELECTOR, "figure.we-star-rating")
                aria_label = rating_element.get_attribute("aria-label")
                rating = int(re.search(r'\d+', aria_label).group())
                body = current_review_card.find_element(By.CSS_SELECTOR, "div.we-customer-review__body").text.strip()
                all_reviews_data.append({"title": title, "rating": rating, "review": body})
                continue

            # --- Step B: Wait for the CORRECT modal pop-up to appear ---
            modal_wait = WebDriverWait(driver, 10)
            # THIS IS THE CORRECTED SELECTOR BASED ON YOUR HTML
            modal = modal_wait.until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, "div.we-modal__content"))
            )

            # --- Step C: Scrape the data FROM THE MODAL ---
            title = modal.find_element(By.CSS_SELECTOR, "h3.we-customer-review__title").text.strip()
            rating_element = modal.find_element(By.CSS_SELECTOR, "figure.we-star-rating")
            aria_label = rating_element.get_attribute("aria-label")
            rating = int(re.search(r'\d+', aria_label).group())
            body = modal.find_element(By.CSS_SELECTOR, "p[data-test-bidi]").text.strip()
            
            all_reviews_data.append({"title": title, "rating": rating, "review": body})

            # --- Step D: Close the modal ---
            close_button = modal.find_element(By.CSS_SELECTOR, "button.we-modal__close")
            close_button.click()

            # --- Step E: Wait for the CORRECT modal to disappear ---
            modal_wait.until(
                EC.invisibility_of_element_located((By.CSS_SELECTOR, "div.we-modal__content"))
            )
            time.sleep(0.5)

        except Exception as e:
            print(f"\nSkipping a review due to an error: {type(e).__name__} - {str(e).splitlines()[0]}")
            try:
                driver.find_element(By.CSS_SELECTOR, "button.we-modal__close").click()
                time.sleep(1)
            except:
                pass
            continue

    driver.quit()

    # --- 4. Saving the Reviews ---
    if not all_reviews_data:
        print(f"No reviews were successfully extracted for '{app_name}'.")
        return None

    reviews_df = pd.DataFrame(all_reviews_data)
    for index, row in tqdm(reviews_df.iterrows(), total=reviews_df.shape[0], desc="Saving reviews"):
        review_text = f"Title: {row['title']}\nRating: {row['rating']}/5\n\n{row['review']}"
        file_path = os.path.join(output_dir, f"review_{index + 1}.txt")
        with open(file_path, "w", encoding="utf-8", errors="ignore") as f:
            f.write(review_text)
            
    print(f"\nSuccessfully saved {len(reviews_df)} reviews to '{output_dir}'")
    return output_dir