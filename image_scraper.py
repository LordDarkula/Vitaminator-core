import hashlib
from HTMLParser import HTMLParser
import urllib2
import urllib
from urllib2 import urlopen
import traceback
import os

import re


class ImageScraper(HTMLParser):
    num_recursions = 100
    links = set()

    def __init__(self, base_url):
        HTMLParser.__init__(self)
        self.base_url = base_url

    def error(self, message):
        print(message)

    def handle_starttag(self, tag, attrs):
        if tag == 'div':
            for (attr, value) in attrs:
                if attr == 'data-item-id':
                    print "data item found"
                    new_url = self.base_url[:-1] \
                        if self.base_url[-1] == '_' \
                        else self.base_url

                    new_url += value

                    self.num_recursions -= 1
                    if self.num_recursions > 0:
                        print("recursion " + self.num_recursions)
                        gather_image_links(new_url)

        if tag == 'a':
            for (attr, value) in attrs:
                if attr == 'href' or attr == 'data-href':
                    if len(value) > 4:

                        is_image = value[-4:] == '.jpg' or \
                                   value[-4:] == '.png' or \
                                   value[-5:] == '.jpeg'
                        if is_image:
                            ImageScraper.links.add(value)

        if tag == 'img':
            for (attr, value) in attrs:
                if attr == 'src':
                    ImageScraper.links.add(value)
                if attr == 'name':
                    print "data item found"
                    new_url = self.base_url[:-1] \
                        if self.base_url[-1] == '_' \
                        else self.base_url

                    new_url += value

                    self.num_recursions -= 1
                    if self.num_recursions > 0:
                        print("recursion " + self.num_recursions)
                        gather_image_links(new_url)

    def image_links(self):
        return self.links


def check_validity(html, allowed):
    html = html.lower()
    regex = ""
    for word in allowed:
        word = word.lower()
        regex += "(?=" + word + ")"
    filted_words = [m.start() for m in re.finditer(regex, html)]
    return len(filted_words) > 0


def gather_image_links(base_url):
    html_string = ''
    try:
        req = urllib2.Request(base_url, headers={'User-Agent': 'Mozilla/5.0'})

        response = urlopen(req)
        # if response.getheader('content-type') == 'text/html':
        html_bytes = response.read()
        html_string = html_bytes.decode('utf-8')

        finder = ImageScraper(base_url=base_url)
        finder.feed(data=html_string)
        return finder.image_links()
    except Exception:

        print('generic exception: ' + traceback.format_exc())
        return set()


def download_images(image_links, folder_name):
    opener = urllib2.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib2.install_opener(opener)

    if not os.path.exists('raw/{}'.format(folder_name)):
        os.makedirs('raw/{}'.format(folder_name))

    for link in image_links:
        full_name = 'raw/{}/{}'.format(folder_name,
                                       hashlib.md5(os.path.splitext(link)[0]).hexdigest())

        if full_name[-4:] == '.png':
            full_name += '.png'
        else:
            full_name += '.jpg'

        try:
            urllib.urlretrieve(link, full_name)
        except Exception:
            print('generic exception: ' + traceback.format_exc())


if __name__ == '__main__':
    url = raw_input("Image url: ")
    my_set = gather_image_links(url)
    print(my_set)
    print(len(my_set))

    download_images(my_set, 'healthy_nails')
