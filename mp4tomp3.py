from __future__ import unicode_literals
import youtube_dl

url = []
with open('obama_addresses.txt', 'r') as f:
    for line in f.readlines():
        url.append(line.strip())

for i in range(len(url)):
    ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
                       'key': 'FFmpegExtractAudio',
                       'preferredcodec': 'mp3',
                       'preferredquality': '192',
                       }],
    'outtmpl': '{:05d}.mp3'.format(i+1)
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url[i]])
