from datetime import timedelta
import datetime
import shutil
import os
import chromadb
import langchain_google_genai as lgg
GOOGLE_API_KEY="" #Enter your apikey here
embeddings = lgg.GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key = GOOGLE_API_KEY)

topics = "World, India, Politics, Sports, Science, Technology, Entertainment, Business, Finance, Health".split(", ")
Dateyes = datetime.datetime.today()-timedelta(days=1)
Dateyes = Dateyes.strftime("%Y-%m-%d")
shutil.rmtree("./Basetoday")
for i in topics:
    client = chromadb.PersistentClient(path=f"./Base/{i}")
    collection = client.get_collection(name = "example_collection")
    for j in range(len(os.listdir(f"./{Dateyes}/{i}"))):
        try:
            with open(f"./{Dateyes}/{i}/{j}.txt", "r") as f:
                    text = f.read()
        except:
            continue
        collection.add(
            documents = [text],
            metadatas = [{"source": f"./{Dateyes}/{i}/{j}"}],
            ids = [f"{datetime.datetime.today()}/{i}/{j}"],
            embeddings = [embeddings.embed_documents([text])[0]]
        )
        print(f"Added {j} to {i}")
    del client 
    del collection
