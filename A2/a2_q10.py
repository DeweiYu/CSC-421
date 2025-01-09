from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import csv

# Scrape categories and facts from songfacts.com
def scrape_songfacts(categories_url):
    # Set up Selenium with WebDriver Manager
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)

    driver.get(categories_url)
    time.sleep(3)  # Allow page to load

    # Get list of categories
    categories = []
    try:
        category_elements = driver.find_elements(By.CSS_SELECTOR, '.categories a')
        for elem in category_elements[:3]:  # Scrape first three categories
            categories.append((elem.text, elem.get_attribute('href')))
    except Exception as e:
        print(f"Error finding categories: {e}")
        driver.quit()
        return []

    # Scrape facts for each category
    facts = []
    for category_name, category_url in categories:
        driver.get(category_url)
        time.sleep(3)  # Allow page to load

        try:
            fact_elements = driver.find_elements(By.CSS_SELECTOR, '.fact-title a')
            for fact_elem in fact_elements:
                song_title = fact_elem.text
                facts.append((category_name, song_title))
        except Exception as e:
            print(f"Error finding facts for category {category_name}: {e}")

    driver.quit()
    return facts

# Write facts to Prolog syntax
def write_prolog_facts(facts, output_file='facts.pl'):
    with open(output_file, 'w') as f:
        for category, song in facts:
            f.write(f'category_song("{category}", "{song}").\n')

# Write sample queries
def write_prolog_queries(output_file='queries.pl'):
    queries = [
        'category_song(C, "Bohemian Rhapsody").',  # Find categories for a specific song
        'category_song("Rock", S).',  # Find all songs in a specific category
        'category_song(C, S), category_song(C2, S), C \\= C2.',  # Songs appearing in multiple categories
        'category_song(C, S), category_song(C, S2), S \\= S2.'  # Multiple songs in the same category
    ]
    with open(output_file, 'w') as f:
        for query in queries:
            f.write(f'?- {query}\n')

# Fallback: Read from CSV and convert to Prolog facts
def fallback_csv_to_prolog(csv_file, output_file='facts.pl'):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        facts = [(row[0], row[1]) for row in reader]
    write_prolog_facts(facts, output_file)

# Main function
def main():
    categories_url = "http://www.songfacts.com/categories"
    try:
        facts = scrape_songfacts(categories_url)

        if facts:
            print(f"Scraped {len(facts)} facts.")
            write_prolog_facts(facts)
            write_prolog_queries()
        else:
            print("No facts scraped. Using fallback CSV.")
            fallback_csv_to_prolog('songfacts.csv')
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
