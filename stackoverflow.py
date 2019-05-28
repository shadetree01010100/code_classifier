from urllib.request import urlopen
from time import sleep
from bs4 import BeautifulSoup


class StackOverflowScraper():

    lines_per_post = 120
    line_length = 80
    url_base = 'https://stackoverflow.com{}'
    section = '/tagged/python'
    # section = ''  # all sections

    page = 1

    def __init__(self, delay=None):
        self.delay = delay

    def next_batch(self, rate=None):
        """ Returns a matrix of labeled thread contents with shape:
        `[~100, <= lines_per_post, 1]`. The inner-most array contains
        a tuple: `(True|False, <string>[:line_length])`.

        The dimension of `~100` assumes that each page has 50 threads,
        and that for each thread the question and accepted answer
        are labeled for use. Page size (threads) appears to be
        non-negotiable, and not every thread has an answer.

        The first value of each line tuple indicates if the line is
        inside `<code></code>` tags.
        """

        url = '/questions{}?page={}&sort=votes'.format(self.section, self.page)
        links = []
        output = []

        full_url = self.url_base.format(url)
        # print('scraping threads', full_url)
        contents = self._read_webpage(full_url)
        questions = contents.find_all('div', {'class': 'summary'})
        for question in questions:
            link = question.find('a', {'class': 'question-hyperlink'})
            links.append(link.attrs['href'])
        for link in links:
            full_url = self.url_base.format(link)
            # print('getting posts', full_url)
            thread = self._read_webpage(full_url)
            question = thread.find('div', {'class' : 'question'})
            accepted_answer = thread.find('div', {'class': 'answer accepted-answer'})
            question_text = question.find('div', {'class': 'post-text'})
            try:
                answer_text = accepted_answer.find('div', {'class': 'post-text'})
            except AttributeError:
                answer_text = None
            # print('labelling question')
            lines = self._label_lines(question_text)[:self.lines_per_post]
            question = []
            for line in lines:
                question.append(line)
            output.append(question)
            if answer_text:
                # print('labelling answer')
                lines = self._label_lines(answer_text)[:self.lines_per_post]
                answer = []
                for line in lines:
                    answer.append(line)
                output.append(answer)
            # else:
                # print('no accepted answer')
            if self.delay is not None:
                # print('sleeping for {} seconds...'.format(self.delay))
                sleep(self.delay)
        self.page += 1
        return output

    def _read_webpage(self, full_url):
        html_doc = urlopen(full_url).read()
        html_object = BeautifulSoup(html_doc, 'html.parser')
        return html_object

    def _label_lines(self, text):
        out = []
        for t in text:
            is_code = int(t.name == 'pre')
            try:
                if is_code:
                    lines = [str(w).split('\n') for w in t.strings][0]
                else:
                    # a stupid idea that works isn't stupid
                    lines = [''.join([str(w) for w in t.contents])]
            except AttributeError:
                # empty page space, not part of actual post contents
                lines = []
            for line in lines:
                out.append((is_code, line[:self.line_length]))
        return out
