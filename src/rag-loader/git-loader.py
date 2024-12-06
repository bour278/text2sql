import requests
from bs4 import BeautifulSoup

class GitHubRepoScraper:
    def __init__(self, url):
        self.url = url

    def fetch_html(self):
        response = requests.get(self.url)
        return response.text

    def parse_html(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        return soup

    def extract_repo_links(self, soup):
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if "github.com" in href:
                links.append(href)
        return links

    def scrape(self):
        html = self.fetch_html()
        soup = self.parse_html(html)
        return self.extract_repo_links(soup)