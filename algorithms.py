from urllib.request import urlopen
import re
from bs4 import BeautifulSoup


# page = 'https://github.com/niolabs/nio-cli/issues/74'
# html = urlopen(page).read()
# soup = BeautifulSoup(html, 'html.parser')

# convo = soup.find(
    # 'div', {'class': 'discussion-timeline js-quote-selection-container '})
# posts = convo.find_all(
    # 'td', {'class': 'd-block comment-body markdown-body js-comment-body'})

# for post in posts:
    # lines = []
    # labels = []
    # for line in post:
        # code = True if line.name == 'pre' else False
        # lines.append(line)
        # labels.append(code)
    # print([(a, b) for a, b in zip(labels, lines)])

    
    
page='http://www.raspberrypi.org/forums/viewtopic.php?f=32&t=204231'
html = urlopen(page).read()
soup = BeautifulSoup(html, 'html.parser')

foo=soup('div', {'class': 'content'})

# for q in ''.join([str(f) for f in foo]).split('\n'):
    # if q:
            # print(q)
            # print('---------------')

for bar in foo:
    for f in bar:
        if hasattr(f, 'attrs'):
                if f.attrs.get('class', [''])[0] == 'codebox':
                        print('CODE')
                        print(f)
                        print('/CODE')
                else:
                        print(f)
        else:
                print(f)
