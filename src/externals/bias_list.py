from bs4 import BeautifulSoup
import requests
import time
import tldextract
from random import randrange
import tqdm
import pickle

def make_bias_list(DATA_PATH):

    domains_dict = get_bias_domains()
    tlds = get_bias_tlds(domains_dict)

    DATA_EXTERNAL_PATH = DATA_PATH + 'external/'

    pickle_out = open(DATA_EXTERNAL_PATH + "tld.pickle","wb")
    pickle.dump(tlds, pickle_out)
    pickle_out.close()

# Get a list of TLDs and bias labels from mediabiasfactcheck.com
def get_bias_domains():

    domains_dict = {}
    root_url = 'https://mediabiasfactcheck.com/'
    bias_list = ['left', 'right', 'leftcenter', 'center', 'right-center']
    for bias in tqdm(bias_list):
        r = requests.get(root_url + bias + '/')
        soup = BeautifulSoup(r.text, 'html.parser')
        capture_flag = False
        domains = []
        for a in tqdm(soup.find_all('a')):

            if a.get('class') == ['share-twitter', 'sd-button', 'share-icon', 'no-text']:
                capture_flag = False
                break

            if capture_flag == True:
                domains.append(a['href'])

            if a.get('class') == ['a2a_dd', 'a2a_counter', 'addtoany_share_save', 'addtoany_share']:
                capture_flag = True 
                
        domains_dict[bias] = domains

    return domains_dict

def get_bias_tlds(domains_dict):
    tld_dict = {}
    for bias in tqdm(domains_dict.keys()):
        for url in tqdm(domains_dict[bias]):
            r = requests.get(url)
            soup = BeautifulSoup(r.text, 'html.parser')
            for p in soup.find_all('p'):
                if 'Source:' in p.text:
                    for child in p.findChildren('a'):
                        tld = tldextract.extract(child['href'])[1]
                        tld_dict[tld] = bias
            time.sleep(randrange(1,2))
    return tld_dict