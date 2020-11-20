from datetime import date
from bs4 import BeautifulSoup
from random import randint
from multiprocessing import Pool
from tqdm import tqdm

import requests
import re
import time
import pandas as pd


def get_links(url, nb_pages):
    """[Scrap the url list of IMDB starting from the first page]

    Args:
        url (str, optional): [first page of "search" section in IMDB]".

    Returns:
        List: Return a list of all url found
    """
    # Header for parse tool
    headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'}

    # Access to the URL page with html.parser
    webpage = requests.get(url[-1], headers=headers, timeout=10)
    time.sleep(randint(2,4))
    soup = BeautifulSoup(webpage.content, "html.parser")
    parsed = soup.find("div",{"class":"desc"})

    # Recursive call or return url_list
    if nb_pages > 0:
        url.append("https://www.imdb.com" + parsed.find("a",{"class":"lister-page-next next-page"}).attrs["href"])
        with open("D:/list_url.txt", 'a') as file:
            file.write("%s\n" %url[-1])
        return get_links(url, nb_pages-1)
    else:
        return url


def imdb_downloader(url):
    """[Scrap multiple information like movie name, etc. from the "url" arg.]
    Args:
        url (list): a list of url in string format 

    Returns:
        DataFrame: Return a Pandas DataFrame with data for n imdb pages
    """
    
    headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'}
    # Access to the URL page with html.parser
    webpage = requests.get(url, headers=headers, timeout=10)
    time.sleep(randint(5,15))

    if webpage.status_code == 200:
        print("Downloading ", url, ".....")
        parser = BeautifulSoup(webpage.content, "html.parser")

        # Extract Movie name/Date/serie name (if any) and serie year date (if any) from the page for each movies/serie
        classement = parser.find_all("h3", {"class":"lister-item-header"})
        movie_name = [x.find_all('a')[0].text if x else "none" for x in classement]
        date = [x.find_all('span')[1].text if x else "none" for x in classement]
        serie_name = ["None" if len(x.find_all('a')) <= 1 else x.find_all('a')[1].text for x in classement]
        serie_date = ["None" if len(x.find_all('span')) <= 2 else x.find_all('span')[2].text for x in classement]
        
        # Extract score of each movie
        all_score = parser.find_all("div", {"class":"inline-block ratings-imdb-rating"})
        score = [float(x.attrs["data-value"]) if x.attrs["data-value"] else "None" for x in all_score]

        # Extract Pegi of each movie
        paragraph = parser.find_all("div", {"class":"lister-item-content"})
        pegi = [x.p.find("span",{"class":"certificate"}).text if "certificate" in str(x.p) else "None" for x in paragraph]

        # Extract duration of movie
        duration = [int(x.p.find("span",{"class":"runtime"}).text.replace(",","").strip(" min")) if "runtime" in str(x.p) else "None" for x in paragraph]

        # movie_type
        movie_type = [x.p.find("span",{"class":"genre"}).text.strip() if "genre" in str(x.p) else "None" for x in paragraph]


        # Extract description of each movie
        paragraph = parser.find_all("p", {"class":"text-muted"}) 
        description = [x.text.strip() for x in paragraph][1::2]

        # Extract number of votes
        votes_revenues = parser.find_all("p",{"class":"sort-num_votes-visible"})
        votes = [int(x.find_all("span")[1].attrs["data-value"]) for x in votes_revenues]
        revenue_movie = ["None" if len(x.find_all("span")) <= 4 else int(x.find_all("span")[4].attrs["data-value"].replace(",","")) for x in votes_revenues]


        # Extract metascore of each movie
        metascore = parser.find_all("div",{"class":"ratings-bar"})
        metascore_i = [int(x.find("div",{"class":"inline-block ratings-metascore"}).span.text) if "metascore" in str(x.find_all("div",{"class":"inline-block"})) else "None" for x in metascore]

        # Extract directors names
        figures = parser.find_all("p",{"class":""})
        directors = [x.find_all("a",{"href":re.compile(r'/name/.*/\?ref_=adv_li_dr_\d')}) for x in figures]
        directors = [[y[i].text for i in range(0,len(y))] for y in directors]

        # Extract actors name
        actors = [x.find_all("a",{"href":re.compile(r'/name/.*/\?ref_=adv_li_st_\d')}) for x in figures]
        actors = [[y[i].text for i in range(0,len(y))] for y in actors]

        data = {"Movie Name":movie_name, "Movie Date":date,"Serie Name":serie_name,"Serie Date":serie_date,"Movie Type":movie_type,
                  "Number of Votes":votes, "Movie Revenue (M$)":revenue_movie, "Score":score,"Metascore":metascore_i,
                  "Time Duration (min)":duration,"Director":directors, "Actors":actors, "Restriction":pegi,"Description":description}

        df = pd.DataFrame.from_dict(data)

        return df


def threading(url, n_pool):
    """[Multiprocessing function for url scraping]

    Args:
        url ([list]): [List of url-str]
        n_pool ([int]): [Number of process executed by the function]

    Returns:
        [pd.Core.DataFrame]: [Return a pandas DataFrame]
    """
    df = pd.DataFrame()
    with Pool(n_pool) as pool, tqdm(total=len(url)) as pbar:
        for data in pool.imap_unordered(imdb_downloader, url):
            df = pd.concat([df, data], ignore_index=True)
            pbar.update()

    return df


def save_data(data, output="csv", path="../cinema_prediction/input/imdb_database"):
    """[Save DataFrame as CSV or JSON file]

    Args:
        data ([DataFrame]): [pandas DataFrame to be saved]
        output (str, optional): [Type of saved file (json or csv)]. Defaults to "csv".
        path (str, optional): [path where the file will be saved]. Defaults to "D:/imdb_database".
    """
    if "csv" in output:
        data.to_csv(path+".csv", index=False)
    else:
        data.to_json(path+".json", force_ascii=True)



def main():
    # Access all movies from 1900 to TODAYS DATE
    today = date.today()
    d1 = today.strftime("%Y-%m-%d")
    
    # Extract and save urls in a txt file 
    imdb_url = 'https://www.imdb.com/search/title/?release_date=1900-01-01,' + d1 + '&sort=num_votes,desc&start='
    movie_url = [imdb_url+str(page) for page in range(1, 10000, 50)] # Output 200 links, each containing 50 movies
    with open("../cinema_prediction/input/list_url.txt", 'w') as file:
        for item in movie_url:
            file.write("%s\n" %item)

    # If the number of movies you want to extract is higher than 10.000, use this function to extract other url (they are randomly generated)
    # Change nb_movie (=total movie you want)
    nb_movies = 50000
    nb_pages = (nb_movies-10000)//50
    movie_url = get_links(movie_url, nb_pages=nb_pages)

    # Read the url-txt file and save it as a list of url
    with open("../cinema_prediction/input/list_url.txt", 'r') as file:
        movie_url = file.readlines()

    # Number of Pool (multiprocess)
    n_pool = 13

    # Execute multithreading + show time it took to download n URL.
    t0 = time.time()
    data = threading(url=movie_url, n_pool=n_pool)
    t1 = time.time()
    print(f'\n{round(t1-t0, 4)} seconds to download {len(movie_url)*50} movies critics.')

    # Save data as csv or json (change output to csv or json to change the type)
    path = "../cinema_prediction/input/IMDB_database"
    save_data(data, output="csv", path=path)

if __name__ == "__main__":
    main()


