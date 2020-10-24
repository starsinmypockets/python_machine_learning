from bs4 import BeautifulSoup
import requests
import ipdb

html = requests.get('https://raw.githubusercontent.com/joelgrus/data/master/getting-data.html').text
soup = BeautifulSoup(html, 'html5lib')
ipdb.set_trace()

first_p = soup.find(p)
first_p_words = soup.p.text.split()
first_p_id = soup.p['id']
all_ps = soup.findAll('p')
p_with_ids =  [p for p in soup.p if p.get('id')]
spans_in_divs = [span
                 for div in soup('div')
                 for span in div('span')
                 ]
