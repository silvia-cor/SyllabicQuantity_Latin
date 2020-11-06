import urllib.request, urllib.error, urllib.parse
import re
import os, sys

def download (authors, dir_path):
    print('----- DOWNLOADING TEXTS -----')

    #creating 'dataset' directory
    try:
        os.mkdir(dir_path)
    except OSError:
        print("Creation of the directory %s failed :(" % dir_path)
        sys.exit(1)

    #get list of titles for each author
    authors_titles = []
    for author in authors:
        #create link to the author's mainpage
        author_link = "http://www.mlat.uzh.ch/MLS/verzeichnis4.php?tabelle=" + \
                    author + '_cps5&id=&nummer=&lang=0&corpus=5&inframe=1'
        author_webpage = urllib.request.urlopen(author_link).read().decode('utf-8')
        #get list of titles for the single author...
        author_titles = re.findall(r'target=\"verz\">([A-Z]+.*?)</a>', author_webpage)
        #... and add it to the list of lists
        authors_titles.append(author_titles)

    #keep the number of downloads failed or suceeded
    n_fails = 0
    n_oks = 0
    for index, author_titles in enumerate(authors_titles):
        for title in author_titles:
            #create text file (where the download will be saved)
            text_file = open(dir_path + '/' + authors[index] + '_' + title + '.txt', 'w')
            title = title.replace(" ", "%20") #the url doesn't work with whitespaces
            #create link to the title's mainpage
            title_link= 'http://www.mlat.uzh.ch/MLS/work_header.php?lang=0&corpus=5&table=' + \
                    authors[index] + '_cps5&title=' + title + '&id=' + authors[index] + '_cps5,' + title
            title_webpage = urllib.request.urlopen(title_link).read().decode('utf-8')
            #get the download link (xml format)
            download_link = re.search(r'download(.*?)\.xml\&xml=1', title_webpage).group(0)
            download_link = 'http://www.mlat.uzh.ch/' + download_link
            textpage = urllib.request.urlopen(download_link).read().decode('utf-8')
            #if the link is wrong, it will simply give a 'could not open' kind of webpage
            if 'could not open' in textpage:
                n_fails += 1
                print(download_link)
            else:
                n_oks +=1
                text_file.write(textpage)
            text_file.close()

    print('----- DOWNLOAD COMPLETE -----')
    print('Texts downloaded:', n_oks)
    print('Failed attempts:', n_fails)

