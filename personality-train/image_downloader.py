from bing_image_downloader import downloader
import os
import unidecode
import urllib3
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

with open('personalities.txt') as f:
    content = f.readlines()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

personalities = [x.strip().lower() for x in content]
for personality in personalities:
    normalizedName = personality.replace(' ', '-')
    normalizedName = unidecode.unidecode(normalizedName).lower()
    downloader.download(personality, limit=20,  output_dir='temp', adult_filter_off=False, force_replace=False, timeout=30,)
    os.rename('temp/' + personality, 'temp/' + normalizedName)
