"""
@author: Jonathan
"""

from bs4 import BeautifulSoup
import requests
import time
import tldextract
from random import randrange
import tqdm
import pickle

# Create and save a dictionary of bias-labeled news sources, scraping info from
# http://mediabiasfactcheck.com
def make_bias_list(DATA_PATH):

    # Create dictionary
    domains_dict = get_bias_domains()
    tlds = get_bias_tlds(domains_dict)

    # Save file to externals data folder
    DATA_EXTERNAL_PATH = DATA_PATH + 'external/'

    pickle_out = open(DATA_EXTERNAL_PATH + "tld.pickle","wb")
    pickle.dump(tlds, pickle_out)
    pickle_out.close()

# Get a list of TLDs and bias labels from mediabiasfactcheck.com
def get_bias_domains():

    # Create empty dictionary
    domains_dict = {}

    # Store root url and listed biases
    root_url = 'https://mediabiasfactcheck.com/'
    bias_list = ['left', 'right', 'leftcenter', 'center', 'right-center']

    # Iterate over biases
    for bias in tqdm(bias_list):

        # Get HTML for each bias page
        r = requests.get(root_url + bias + '/')
        soup = BeautifulSoup(r.text, 'html.parser')

        # Create flag and empty list for domains
        capture_flag = False
        domains = []

        # Iterate over links on page
        for a in tqdm(soup.find_all('a')):

            # Stop capturing and exit
            if a.get('class') == ['share-twitter', 'sd-button', 'share-icon', 'no-text']:
                capture_flag = False
                break

            # Append mediabiasfactcheck URL for each listed source's page to list
            if capture_flag == True:
                domains.append(a['href'])

            # Begin capturing
            if a.get('class') == ['a2a_dd', 'a2a_counter', 'addtoany_share_save', 'addtoany_share']:
                capture_flag = True 
                
        domains_dict[bias] = domains

    return domains_dict

# Get the TLD for each listed source on http://mediabiasfactcheck.com
def get_bias_tlds(domains_dict):

    # Create empty dictionary
    tld_dict = {}

    # Iterate over biases
    for bias in tqdm(domains_dict.keys()):

        # Iterate over page URLS
        for url in tqdm(domains_dict[bias]):

            # Get HTML for URL
            r = requests.get(url)
            soup = BeautifulSoup(r.text, 'html.parser')

            # Iterate over paragraph tags
            for p in soup.find_all('p'):

                # Find "Source: " in HTML
                if 'Source:' in p.text:

                    # Extract TLD from Source link
                    for child in p.findChildren('a'):
                        tld = tldextract.extract(child['href'])[1]
                        tld_dict[tld] = bias

            # Wait for a random time so as not to overload page with GET requests
            time.sleep(randrange(1,2))
            
    return tld_dict