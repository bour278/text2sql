import os
import re
import time
import random
import redis
from git import Repo
from colorama import Fore, Style

class GitHubScraper:
    def __init__(self, ip_list_path):
        self.ip_list_path = ip_list_path
        self.ip_list = self.load_ip_list()
        self.base_path = '../../code-data/'
        self._git_internal_storage = ('.idx', '.rev', '.pack') #just skip those
        self.redis_client = redis.Redis(host='redis', port=6379, db=0)

    def load_ip_list(self):
        with open(self.ip_list_path, 'r') as file:
            return [line.strip() for line in file.readlines()]

    def mask_ips(self, content):
        pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, self.ip_list)) + r')\b')
        return pattern.sub("0.0.0.0", content)

    def scrape_repository(self, repo_url):
        repo_name = repo_url.split('/')[-1]
        local_path = os.path.join(self.base_path, repo_name)
        if os.path.exists(local_path):
            print(Fore.MAGENTA + f"Repository {repo_name} has already been cloned." + Style.RESET_ALL)
            return
        os.makedirs(local_path)
        Repo.clone_from(repo_url, local_path)
        self.process_files(local_path)
        self.redis_client.hset(repo_url, "cloned", "true")

    def process_file(self, file_path):

        if not file_path.endswith('.q'): os.remove(file_path); return
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
        masked_content = self.mask_ips(content)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(masked_content)


    def process_files(self, path):
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                if file_path.endswith(self._git_internal_storage): continue
                try:
                    self.process_file(file_path)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    def save_scraped_data(self):
        for key in self.redis_client.keys():
            if self.redis_client.hget(key, "cloned") == b"false":
                self.scrape_repository(key.decode('utf-8'))
                time.sleep(max(3, random.gauss(2, 0.2)))

if __name__ == "__main__":
    scraper = GitHubScraper('../../resources/ip-addresses.txt')
    scraper.save_scraped_data()