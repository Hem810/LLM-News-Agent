import langchain_google_genai as lgg

GOOGLE_API_KEY="" #Enter your apikey here

model = lgg.ChatGoogleGenerativeAI(model = "gemini-1.5-flash", google_api_key = GOOGLE_API_KEY)
embeddings = lgg.GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key = GOOGLE_API_KEY)
import json
import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime
# LINK EXTRACTION MODULE

# Extract links from Google News search results for a query to a text file for further scraping
def extract_links_to_file(query, output_file_path, num_results = 25,):
    # Prepare the search URL
    search_url = (
        f"https://www.google.com/search?q={query}&gl=us&tbm=nws&num={num_results}"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/101.0.4951.54 Safari/537.36"
        )
    }
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    links = []
    # Each news article is inside a div with class 'SoaBEf'
    for el in soup.select("div.SoaBEf"):
        a_tag = el.find("a")
        if a_tag and a_tag.has_attr("href"):
            links.append(a_tag["href"])
    return links

# main function call, making links folders of to be scraped websites outlining the topics listed below from 
s  = "World, India, Politics, Sports, Science, Technology, Entertainment, Business, Finance, Health, IndiaTrending"
m = s.split(", ")
topics = m
Date = datetime.now().strftime("%Y-%m-%d")
if not os.path.exists(Date):
    os.makedirs(Date)
if not os.path.exists(f'./{Date}/links'):
    os.makedirs(f'./{Date}/links')
for i in topics:
  output_file_path = f'./{Date}/links/{i}.txt'
  query = f'{i}'
  g = extract_links_to_file(query,output_file_path, 25)
  with open(output_file_path, "w") as f:
    for link in g:
      f.write(link + "\n")

from newspaper import Article
from newspaper import Config

# Scrape single article data individually and save to text file in topic folder
def extract_article_data(url, i, topic, Date):
    config = Config()
    config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'

    article = Article(url, config=config)

    # Download and parse the article
    try:
      article.download()
      article.parse()
    except:
      return
    # Extract relevant data from the article
    data = {
        "title": article.title,
        "authors": article.authors,
        "published_date": article.publish_date,
        "text": article.text,
        "summary": article.summary,
        "keywords": article.keywords
    }
    with open(f"./{Date}/{topic}/"+str(i)+".txt", "w") as f:
      f.write("Title:"+ data["title"]+"\n")
      f.write("Authors:"+ ",".join(data["authors"])+"\n")
      f.write("Published Date:"+ str(data["published_date"])+"\n")
      f.write("Text:"+ data["text"]+"\n")
      f.write("Summary:"+ data["summary"]+"\n")
      f.write("Keywords:"+ ",".join(data["keywords"])+"\n")

# Call extract_article_data function for each link in the topic link file
def articleSave(topic, Date):
  for i in topic:
    if not os.path.exists(f'./{Date}/{i}'):
      os.makedirs(f'./{Date}/{i}')
    with open(f"{Date}/links/{i}.txt", "r") as f:
      links = f.readlines()
      for j, link in enumerate(links):
        try:
          extract_article_data(link, j, i, Date)
        except:
          pass
import datetime
Date = datetime.datetime.now().strftime("%Y-%m-%d")
import os
# Create folders for each topic by calling articleSave function
articleSave(topics, Date)
# Create rag directory for vector database
import shutil
import time
shutil.rmtree('./Basetoday', ignore_errors=True)
time.sleep(1)
os.makedirs(f'./Basetoday', exist_ok=True)
from langchain_community.vectorstores import Chroma
# Create a vector store using Chroma for the scraped articles
Bases = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./Basetoday",  # Directory to save data locally
    )

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from uuid import uuid4


from datetime import timedelta, datetime
import datetime
import os
# Function to add documents to the vector store
def add_document(path, topic, j, Date = datetime.datetime.today().strftime("%Y-%m-%d")):
  loader = TextLoader(path)
  docs = loader.load()  # returns a list of Document objects
  # Generate unique IDs for each document
  ids = [f"{Date}/{topic}/{j}" for _ in docs]
  # Add documents to the vector store
  Bases.add_documents(documents=docs, ids=ids)
s  = "World, India, Politics, Sports, Science, Technology, Entertainment, Business, Finance, Health, IndiaTrending"
m = s.split(", ")
topics = m
Date = datetime.datetime.today()
Date = Date.strftime("%Y-%m-%d")
for i in topics:
  for j in range(len(os.listdir(f"./{Date}/{i}"))): 
    try:
      add_document(f"./{Date}/{i}/{str(j)}.txt", i, j)
      print(f"Added document {j} of {i} to stRag")
    except:
      pass
del Bases
