<<<<<<< HEAD
from HTMLParser import HTMLParser
import urllib2
import urllib
from urllib2 import urlopen
import traceback
import os
=======
from html.parser import HTMLParser
from urllib import parse
import urllib.request
from urllib.request import urlopen
import traceback
>>>>>>> 765aa6f844ac505236ed15a24bfd619e616d8365


class ImageScraper(HTMLParser):

    num_recursions = 0
    links = set()

    def __init__(self, base_url):
<<<<<<< HEAD
        HTMLParser.__init__(self)
=======
        super(ImageScraper, self).__init__()
>>>>>>> 765aa6f844ac505236ed15a24bfd619e616d8365
        self.base_url = base_url

    def error(self, message):
        print(message)

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for (attr, value) in attrs:
                if attr == 'href' or attr == 'data-href':
                    if len(value) > 4:
<<<<<<< HEAD
                        print(value)
                        is_image = value[-4:] == '.jpg' or \
                            value[-4:] == '.png' or \
                            value[-5:] == '.jpeg'
                        if is_image:
                            ImageScraper.links.add(value)
                        elif ImageScraper.num_recursions < 10:
=======
                        print(value[-4:])
                        if value[-4:] == '.jpg' or value[-4:] == '.png' or value[-5:] == '.jpeg':
                            ImageScraper.links.add(value)
                        elif ImageScraper.num_recursions < 100:
>>>>>>> 765aa6f844ac505236ed15a24bfd619e616d8365
                            ImageScraper.num_recursions += 1
                            print(ImageScraper.num_recursions)
                            gather_image_links(value)

        if tag == 'img':
            for (attr, value) in attrs:
                if attr == 'src':
                    ImageScraper.links.add(value)
<<<<<<< HEAD
=======
                # if attr == 'data-href':
                #     if value[-4] == '.jpg' or value[-5] == '.jpeg':
                #         url = parse.urljoin(self.base_url, value)
                #         self.links.add(url)
>>>>>>> 765aa6f844ac505236ed15a24bfd619e616d8365

    def image_links(self):
        return self.links


def gather_image_links(base_url):
    html_string = ''
    try:
        print("TRYING")
<<<<<<< HEAD
        req = urllib2.Request(base_url, headers={'User-Agent': 'Mozilla/5.0'})
=======
        req = urllib.request.Request(base_url, headers={'User-Agent': 'Mozilla/5.0'})
>>>>>>> 765aa6f844ac505236ed15a24bfd619e616d8365
        response = urlopen(req)
        # if response.getheader('content-type') == 'text/html':
        html_bytes = response.read()
        html_string = html_bytes.decode('utf-8')
        print('<a' in html_string)

        finder = ImageScraper(base_url=base_url)
        finder.feed(data=html_string)
        return finder.image_links()
    except Exception:

        print('generic exception: ' + traceback.format_exc())
        return set()


<<<<<<< HEAD
def download_images(image_links, folder_name):
    opener = urllib2.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib2.install_opener(opener)

    if not os.path.exists('raw/{}'.format(folder_name)):
        os.makedirs('raw/{}'.format(folder_name))

    for i, link in enumerate(image_links):
        full_name = 'raw/{}/image{}'.format(folder_name, i)
=======
def download_images(image_links):
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    for i, link in enumerate(image_links):
        full_name = 'image{}'.format(i)
>>>>>>> 765aa6f844ac505236ed15a24bfd619e616d8365

        if full_name[-4:] == '.png':
            full_name += '.png'
        else:
            full_name += '.jpg'

        try:
<<<<<<< HEAD
            urllib.urlretrieve(link, full_name)
=======
            urllib.request.urlretrieve(link, full_name)
>>>>>>> 765aa6f844ac505236ed15a24bfd619e616d8365
        except Exception:
            print('generic exception: ' + traceback.format_exc())


<<<<<<< HEAD
if __name__ == '__main__':
    my_set = gather_image_links(
        'http://www.bing.com/images/search?q=leukonychia+nail+disease')
    print(len(my_set))
    download_images(my_set, 'white_spots')
=======
my_set = gather_image_links(
    'https://www.bing.com/images/search?q=healthy+nails&FORM=HDRSC2')
print(len(my_set))

download_images(my_set)
>>>>>>> 765aa6f844ac505236ed15a24bfd619e616d8365
