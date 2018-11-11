"""
@author: Jonathan
"""

import pickle
import time
from random import randrange

import requests
import tldextract
import tqdm
from bs4 import BeautifulSoup


# Create and save a dictionary of bias-labeled news sources, scraping info from
# http://mediabiasfactcheck.com
def make_bias_list(data_path):
    # Create dictionary
    domains_dict = get_bias_domains()
    tlds = get_bias_tlds(domains_dict)

    # Get labelled domains from data which are not on mediabiasfactcheck
    tlds = get_df_tlds(tlds)

    # Save file to externals data folder
    util_path = 'util/'

    pickle_out = open(util_path + "tld.pickle", "wb")
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
            time.sleep(randrange(1, 3))

    return tld_dict


# Get labelled TLDs from data not in MBFC scrape
def get_df_tlds(tlds):

    # Load data
    data_interim_path = 'data/interim/'

    train = pd.read_csv(data_interim_path + 'train_p.csv')
    val = pd.read_csv(data_interim_path + 'val_p.csv')

    # Get sets of tlds from data
    train_domains = set(train['domain'])
    val_domains = set(val['domain'])
    df_domains = train_domains.union(val_domains)

    # Get domains from scrape
    scraped_domains = set(tlds.keys())

    diff = df_domains.difference(scraped_domains)

    # Iterate over difference and find their bias
    diff_dict = {}
    for domain in diff:
        if domain in train_domains:
            diff_dict[domain] = train[train['domain'] == domain]['bias'].values[0]
        elif domain in val_domains:
            diff_dict[domain] = val[val['domain'] == domain]['bias'].values[0]

        # Correct bias to match those labels from scrape
        if diff_dict[domain] == 'left-center':
            diff_dict[domain] = 'leftcenter'
        elif diff_dict[domain] == 'least':
            diff_dict[domain] = 'center'

    # Merge and return
    merged_dict = {**tlds, **diff_dict}

    return merged_dict