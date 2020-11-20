# IMDB scraper

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub watchers](https://img.shields.io/badge/Watchers-1-blue)](https://github.com/Anghille/imdb_scraping/watchers)
[![Pull-Requests Welcome](https://img.shields.io/badge/Pull%20Request-Welcome-blue)](https://github.com/Anghille/imdb_scraping/pulls)

[![python](https://img.shields.io/badge/Made%20with-Python-blue)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-v0.3.2-blue)](https://github.com/Anghille/imdb_scraping#versioning)

If you prefer to access their official database, you can do so by following this link : [IMDB](https://www.imdb.com/interfaces/)

This tool is used as a training webscraping project and ML project.


## Introduction

This tool is used to scrap the www.imdb.com website using requests, [BeautifoulSoup](https://www.crummy.com/software/BeautifulSoup/)/[Scrapy](https://github.com/scrapy/scrapy) and Pandas.

The database contains:
* **Movie Name** *(str)*: the name of the movie/Serie
* **Movie Date** *(date)*: the date when the movie came out  
* **Serie Name** *(str)*: the name of the serie season (if any)  
* **Serie Date** *(date)*: The date when the serie season came out (if any)  
* **Movie type** *(str)*: type of the movie (action,drama, sci-fi...)
* **Number of votes** *(int)*: number of people who voted for the metascore
* **Movie Revenue in millions of $** *(int)*: revenue (box-office) the movie made in million-$
* **Score** *(float)*: the mean-score attributed to the movie from 1 to 10 by the journalists
* **Metascore** *(int)*: the mean-score attributed to the movie/serie from 1 to 100 by the viewers
* **Time Duration** *(int)*: the duration of the movie in minutes
* **Director** *(list)*: list of director(s) that directed the movie/serie/season
* **Actors** *(list)*: list of main actor(s) that played in the movie/serie/season 
* **Restriction** *(str)*: Age restriction and warning (all public, all public with warning, 12, 12 with warnings, 16...)
* **Description** *(str)*: A short abstract of the movie

## Installation

1. Clone the repo and navigate into `imdb_scraping` folder.
```
$ git clone https://github.com/Anghille/imdb_scraping.git
$ cd imdb_scraping/
```
2. Create and activate a virtual environment.
```
(imdb_scraping) $ pipenv shell
```
3. Install all dependencies.
```
(imdb_scraping) $ pipenv install
```

## To Do

* Add metascore given to almost every (but not all) movies (Done v0.2.1)
* multithread the scrapping to fasten the process (Done v0.2.0)
* Change scraper method (from BeautifulSoup to Scrapy)(work in progress)
  * add pipeline to mongoDB localy

## Versioning
### v.0.3.2
Added Director *feature*  
Added Actors *feature*  
Added cinema_analysis *jupyter notebook*

### v0.3.1
Added docstring to each functions  
Added list_url.txt file that contains a list of 3740 urls from IMDB  
Added the imdb_database.csv file (189.000 movies)  
Fixed a bug where duration of movies where higher than 999 minutes  

### v0.3.0 (deprecated)
Modified threading to multiprocessing (increased speed)  
Added progress bar  
Added url-list downloader function

### v0.2.1 (deprecated)

Added Metascore *feature*  
Added Restriction *feature*  
Modified Time Duration *feature* (bug correction)  
Modified Movie Type *feature* (bug correction)  
Modified Movie Revenue (M$) *feature* (str -> int)  
Improved speed (from 1 movie to 50 movie processing for the same time with less I/O head)  


### v0.2.0 (deprecated)

Modified *internal code*  
Improved Stability and speed (for-loop -> list comprehension, multithreading, improved bs4 script)  


### v0.1.0 (deprecated)

Creation date (14/10/2020)  
Implemented bone-code structure  
Added Movie Name *feature*  
Added Movie Date *feature*  
Added Serie Name *feature*  
Added Serie Date *feature*  
Added Score *feature*  
Added Time Duration (min) *feature*  
Added Description *feature*  
