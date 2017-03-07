from html.parser import HTMLParser
from urllib import parse
import urllib.request
from urllib.request import urlopen
import traceback


class ImageScraper(HTMLParser):

    num_recursions = 0
    links = set()

    def __init__(self, base_url):
        super(ImageScraper, self).__init__()
        self.base_url = base_url

    def error(self, message):
        print(message)

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for (attr, value) in attrs:
                if attr == 'href' or attr == 'data-href':
                    if len(value) > 4:
                        print(value[-4:])
                        if value[-4:] == '.jpg' or value[-4:] == '.png' or value[-5:] == '.jpeg':
                            ImageScraper.links.add(value)
                        elif ImageScraper.num_recursions < 100:
                            ImageScraper.num_recursions += 1
                            print(ImageScraper.num_recursions)
                            gather_image_links(value)

        if tag == 'img':
            for (attr, value) in attrs:
                if attr == 'src':
                    ImageScraper.links.add(value)
                # if attr == 'data-href':
                #     if value[-4] == '.jpg' or value[-5] == '.jpeg':
                #         url = parse.urljoin(self.base_url, value)
                #         self.links.add(url)

    def image_links(self):
        return self.links


def gather_image_links(base_url):
    html_string = ''
    try:
        print("TRYING")
        req = urllib.request.Request(base_url, headers={'User-Agent': 'Mozilla/5.0'})
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


def download_images(image_links):
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    for i, link in enumerate(image_links):
        full_name = 'image{}'.format(i)

        if full_name[-4:] == '.png':
            full_name += '.png'
        else:
            full_name += '.jpg'

        try:
            urllib.request.urlretrieve(link, full_name)
        except Exception:
            print('generic exception: ' + traceback.format_exc())


my_set = gather_image_links(
    'https://www.bing.com/images/search?q=healthy+nails&FORM=HDRSC2')
print(len(my_set))

download_images(my_set)
