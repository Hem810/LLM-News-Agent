import streamlit as st
import time
import requests
import json
from typing import List, Dict, Any
import langchain_google_genai as lgg
import chromadb
import json

GOOGLE_API_KEY=""  #Enter your API key here


history = """"""
model = lgg.ChatGoogleGenerativeAI(model = "gemini-1.5-flash", google_api_key =GOOGLE_API_KEY)
st.set_page_config(layout="wide")
Bias_User=0
# --- Modern Fonts via Google Fonts ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Poppins:wght@700&family=Raleway:wght@500&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"]  {
        font-family: 'Inter', 'Raleway', 'Poppins', sans-serif !important;
    }
    .gradient-border-inner {
        border-radius: 15px;
        padding: 18px 16px 16px 16px;
        width: 100%;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State for Theme and Backend URL ---
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "backend_url" not in st.session_state:
    st.session_state.backend_url = "http://localhost:8000"  # Default backend URL

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

# --- Custom CSS for Gradient Borders and Modern Look ---
def inject_css(theme):
    if theme == "dark":
        st.markdown("""
    <style>
    body, .stApp { background: #181825 !important; color: #ffffff !important; }
    .gradient-border {
        background: linear-gradient(#232336, #232336) padding-box,
                    linear-gradient(90deg, #3b82f6, #a21caf, #ef4444) border-box;
        border: 3px solid transparent;
        border-radius: 18px;
        min-height: 170px;
        box-shadow: 0 4px 24px 0 #0002;
        margin-bottom: 0.7rem;
    }
    .gradient-border-inner {
        background: #232336;
        color: #ffffff;
        font-size: 1.05rem;
        line-height: 1.6;
        font-weight: 600;
    }
    .gradient-border-inner h4 {
        color: #ffffff !important;
        font-weight: 700;
    }
    .gradient-border-inner p {
        color: #ffffff !important;
        font-weight: 600;
    }
    .gradient-border-inner a {
        color: #ffffff !important;
        text-decoration: underline;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <style>
        body, .stApp { background: #f7f7fa !important; color: #22223b !important; }
        .gradient-border {
            background: linear-gradient(#fff, #fff) padding-box,
                        linear-gradient(90deg, #f472b6, #facc15) border-box;
            border: 3px solid transparent;
            border-radius: 18px;
            min-height: 170px;
            box-shadow: 0 4px 24px 0 #0001;
            margin-bottom: 0.7rem;
        }
        .gradient-border-inner {
            background: #fff;
            color: #22223b;
        }
        .gradient-border-inner h4, .gradient-border-inner p, .gradient-border-inner a {
            color: #22223b !important;
        }
        .gradient-border-inner a {
            text-decoration: underline;
        }
        </style>
        """, unsafe_allow_html=True)

inject_css(st.session_state.theme)

st.markdown("""
<style>
.view-selector {
    position: absolute;
    top: 0.8rem;
    right: 1.5rem;
    z-index: 1000;
    width: 200px;
    background: transparent;
}
.view-selector .stSelectbox div[data-baseweb="select"] {
    background: transparent;
}
.view-selector .stSelectbox div[data-baseweb="select"] > div {
    background: transparent;
    border: none;
    box-shadow: none;
}
</style>
""", unsafe_allow_html=True)

# Create a container for the slider in the top right
view_selector = st.container()

# Use columns to position the slider at the top right
with view_selector:
    st.markdown('<div class="view-selector">', unsafe_allow_html=True)
    view_value = st.select_slider(
        "Indicate your preference",
        options=["Left", "Center", "Right"],
        value="Center",
        key="view_mode"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Convert slider value to numeric value for backend
view_numeric = {"Left": -1, "Center": 0, "Right": 1}[view_value]

# Modify your query_backend function to include the view parameter


# --- State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []




#Function to generate optimal query for the rag
def optiraggen(user, model = model):
    t = 1
    while t == 1:
        try:
            rag_opti_prompt = """
            You are an expert language model specializing in generating precise semantic search summaries. Given a user prompt, produce a ~100-word description that concisely captures the core information need, key entities, events, and context of the query. The output will be embedded and used to query a vector database in a Retrieval-Augmented Generation (RAG) system focused on news analysis.
            Your task is to preserve the intent and specificity of the user input while abstracting away unnecessary verbosity. The summary must be general enough for relevant matches in the database, yet specific enough to maintain fidelity to the user’s query.
            Use the following steps:
            Identify the main topic or event the user is interested in.
            Extract relevant entities, such as names, locations, dates, or organizations.
            Highlight any analytical lens, e.g., bias, comparison, timeline, sentiment, geopolitical implications.
            Rephrase the content into a succinct, clear, and context-rich 50-word summary suitable for vector embedding
            Examples:
            User prompt: "What are the differences in media narratives between Western and Chinese news outlets about the recent Taiwan Strait tensions?"
            Summary: Comparative analysis of Western vs. Chinese media framing of Taiwan Strait tensions, focusing on narrative divergence, geopolitical stakes, and rhetorical strategies post-incident.
            User prompt: "Give me insights into how the 2024 U.S. elections are being portrayed differently in conservative vs. liberal media sources."
            Summary: Exploration of partisan media portrayals of the 2024 U.S. elections, emphasizing narrative framing, ideological bias, and differences in issue prioritization between conservative and liberal outlets.
            Begin with the user prompt below.
            User prompt:
            {Input}
            Output: Generate a 100 word summary, return output in a json format with the following structure:
            jsonstart
                summary: "summary"
            jsonend
            where "summary" is a string 
            """.format(Input = user)
            model = lgg.ChatGoogleGenerativeAI(model = "gemini-1.5-flash", google_api_key = GOOGLE_API_KEY, model_kwargs={"response_format": "json"} )
            t = model.invoke(rag_opti_prompt).content.strip(r"`\\njso")
        except:
            pass
    r = json.loads(t,strict = False)
    return r["summary"]
import langchain_google_genai as lgg
embeddings  = lgg.GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key = GOOGLE_API_KEY)




#Query the rags for related documents
def callrag(prompt, fileloc):
    client = chromadb.PersistentClient(path=f"./Base")
    client2 = chromadb.PersistentClient(path=f"./Basetoday")
    collection = client.get_collection(name = "example_collection")
    collections = client2.get_collection(name = "example_collection")
    results_old = collection.query(query_embeddings=embeddings.embed_documents([prompt]), n_results=5)
    results_new = collections.query(query_embeddings=embeddings.embed_documents([prompt]), n_results=5)
    return results_old, results_new

model = lgg.ChatGoogleGenerativeAI(model = "gemini-1.5-flash", google_api_key = GOOGLE_API_KEY, model_kwargs={"response_format": "json"} )


#Funtion to provide the summary of the retreieved articles
def summariser(resultsold, resultsnew, model = model):
    old = resultsold["documents"][0]
    new = resultsnew["documents"][0]
    old = [" "] + [i for i in old if i != ""]
    new = [" "] + [i for i in new if i != ""]
    old = "next Article\n".join(old)
    new = "next Article\n".join(new)
    deaf = 1
    while deaf == 1:
        try:
            promptsummaryold = f"""You are a factual and concise summarization assistant. You will be given the full text of 5 news articles. Your task is to extract 3 to 5 bullet points for each article, summarizing the most important and explicitly stated facts only. The output should be a single JSON object.

        Requirements:
        Faithful Summarization: Only include information that is explicitly stated in the article text. Do not guess, infer, or hallucinate.

        Clarity and Conciseness: Each bullet point should be 1–2 sentences max.

        Structured Output: For each article, begin with Article X Summary: followed by its bullet points.

        Separation: Separate each article’s summary with two newline characters (\n\n) in the final JSON value.

        Format Output as JSON with a single key: "key_points".

        Output Format (JSON):
        json

        jsonstart
        "key_points": "Article 1 Summary:\n• Bullet point 1\n• Bullet point 2\n• ...\n\nArticle 2 Summary:\n• Bullet point 1\n• ...\n\n... (up to Article 5)"
        jsonend
        Input Articles: {old}"""
            promptsummarynew = f"""You are a factual and concise summarization assistant. You will be given the full text of 5 news articles. Your task is to extract 3 to 5 bullet points for each article, summarizing the most important and explicitly stated facts only. The output should be a single JSON object.

        Requirements:
        Faithful Summarization: Only include information that is explicitly stated in the article text. Do not guess, infer, or hallucinate.

        Clarity and Conciseness: Each bullet point should be 1–2 sentences max.

        Structured Output: For each article, begin with Article X Summary: followed by its bullet points.

        Separation: Separate each article’s summary with two newline characters (\n\n) in the final JSON value.

        Format Output as JSON with a single key: "key_points".

        Output Format (JSON):
        json
        jsonstart
        "key_points": "Article 1 Summary:\n• Bullet point 1\n• Bullet point 2\n• ...\n\nArticle 2 Summary:\n• Bullet point 1\n• ...\n\n... (up to Article 5)"
        jsonend
        Input Articles: {new}"""
            old = json.loads(model.invoke(promptsummaryold).content.strip(r"`\\njso"),strict = False)

            new = json.loads(model.invoke(promptsummarynew).content.strip(r"`\\njso"),strict = False)
            deaf = 0
        except:
            pass
    return old, new




#Generate the final answer to show the user
def callfinal(old, new, history, model = model):
    old = old["key_points"]
    new = new["key_points"]
    deaf = 1
    while deaf == 1:
        try:
            promptfinal = f"""
        You are a news intelligence analyst assistant. Your job is to analyze and reason about evolving news based on factual summaries. You are given two sets of bullet-point summaries:

        old_summary: earlier or background developments

        new_summary: recent updates or current developments

        Your task is to produce a logically sound and context-aware interpretation that:

        Draws conclusions and insights using both the old and new summaries.

        Prioritizes recent developments (new_summary) while using old_summary for relevant background or context.

        Uses logical reasoning and general knowledge to explain implications or trends, but only if clearly supported.

        Remains faithful to the content of the summaries — no fabrication, no speculation, and no hallucinated facts.

        Avoids direct comparison unless naturally relevant — the goal is interpretation, not difference detection.

        You will also recieve chat history with the user to use relevant queues and update annswer accordingly.
        Input:

        old_summary = 
        {old}


        new_summary = 
        {new}

        history = 
        {history}
        Output Format:

        Return your reasoned interpretation as a JSON object with a single key "output".

        json
        jsonstart
        "output": "A logically sound, fact-grounded interpretation of the situation that uses both summaries and gives priority to recent updates. No hallucinated facts. Reason through implications and what this might mean going forward, but stay grounded in the information provided."
        jsonend

        """
            
            final = model.invoke(promptfinal).content.strip(r"`\\njso")
            final = json.loads(final)
            final = final['output']
            tapppm = f"""
        You are an expert analyst. You’ll receive two variables:

        • sources: a list of articles, each with a title and full text  
        • answer: a draft write‑up describing what happens  

        Your job is to verify and enrich “answer” against “sources” across five areas:
        1. Effects on Financial Sector  
        2. Effects on Government Sector  
        3. Effects on Service Sector  
        4. Effects on Services‑Based Sector  
        5. Reaction of Other Countries  

        For each area, include:
        – Evidence: key facts from sources  
        – Verification: Supported/Unsupported/Contradicted (cite as [Title, ¶#])  
        – Analysis: insights, mechanisms, uncertainties  

        Also add:
        – Executive Summary (3–5 sentences)  
        – Methodology  
        – Gaps & Conflicts  
        – Conclusion & Recommendations  

        Return your changed answer in about 5 paragraohs, clearly bullet pointed, explaining reasoning primed for a final user which is a JSON string:
        The format of the json output is as follows:

        jsonstart
        "output": "<full report>"
        jsonend

        sources: {old+new}
        generated answer: {final}
        generate 
        """
            final = model.invoke(tapppm).content.strip(r"`\\njso")
            print("FINAL==",final)
            final = json.loads(final,strict = False)
            deaf = 0
        except:
            pass
    
    return final["output"]





#Check the bias of the given articles
def getbias(prompt):
    article = prompt
    deaf=1
    while deaf == 1:
        try:
            pmptbiascheck = f"""**Objective:** Determine the dominant political bias (Left-Wing, Right-Wing, or Neutral) present in the provided news article.  This analysis should be detailed and incorporate multiple indicators.  Output should include a confidence level for each identified bias and a justification for the assessment.
        {article}
        **Instructions:**

        **Phase 1: Input & Initial Assessment (Model Instruction)**

        1.  **Input Text:** [Paste the full text of the news article here]
        2.  **Article Metadata (Important for Context):**
            *   Source: [Identify the news outlet - e.g., The New York Times, Fox News, Reuters, etc.]
            *   Date of Publication: [Insert date]
            *   Author (if known): [Insert name]
            *   Section/Category: [e.g., Politics, Business, World News]
        3.  **Initial Sentiment Scan:**  Perform a quick sentiment analysis on the entire article.  Is the overall tone predominantly positive, negative, or neutral? (Record: Overall Sentiment - Positive/Negative/Neutral - Confidence Level: High/Medium/Low)

        **Phase 2: Detailed Bias Analysis (Core Logic – Model Task)**

        This phase breaks down the analysis into several key categories, with specific questions to guide the model's examination.

        **1. Language & Tone:**
            *   **Loaded Language:**  Does the article use emotionally charged words, adjectives, or phrases that suggest a particular viewpoint? (Examples: "radical," "extremist," "threat," "champion," "hero," "crisis").  *Identify specific examples.* (Category: Loaded Language - Instances Identified: [List Instances] - Confidence: High/Medium/Low)
            *   **Euphemisms/Dysphemisms:** Are euphemisms used to soften negative concepts or dysphemisms used to heighten negative ones? (Identify Examples –  Confidence: High/Medium/Low)
            *   **Framing Words:** Identify words used to frame a situation. For example, is a protest framed as “disruptive” or “defending freedom”? (Identify Examples - Confidence: High/Medium/Low)
            *   **Personalization:** Does the article use personal anecdotes or stories to sway the reader? (Identify Examples - Confidence: High/Medium/Low)

        **2. Topic Selection & Emphasis:**
            *   **Story Angle:** What is the primary story angle presented? (e.g., Economic impact of a policy, individual character of a political figure, protest movement) (Identify Primary Angle - Confidence: High/Medium/Low)
            *   **Selective Reporting:** Does the article selectively highlight certain facts while omitting others that could offer a different perspective? (Evidence of Omission - Provide specific instances of facts left out - Confidence: High/Medium/Low)
            *   **Topic Prioritization:** Which topics are given the most attention and space? Are less critical issues downplayed? (Identify Areas of Emphasis - Confidence: High/Medium/Low)
            *   **Counter-Narratives:**  Does the article acknowledge and address opposing viewpoints, or does it primarily present a single narrative? (Evidence of Counter-Narrative - Provide specific instances - Confidence: High/Medium/Low)

        **3. Attribution & Sourcing:**
            *   **Source Selection:** Who is being quoted? Are the sources primarily from liberal, conservative, or neutral organizations?  (Analyze Source Representation - Confidence: High/Medium/Low)
            *   **Quote Usage:** How are quotes being used? Are they taken out of context, or presented accurately? (Evidence of Misrepresentation -  Provide specific examples - Confidence: High/Medium/Low)
            *   **Unnamed Sources:**  Does the article rely heavily on anonymous sources? (Analyze Reliance on Anonymous Sources -  Identify Instances - Confidence: High/Medium/Low) - *Note: Excessive use should raise a flag.*

        **4. Framing of Policy & Individuals:**
            *   **Policy Descriptions:** How are policies described? Are they presented as solutions or problems? (Analyze Policy Framing - Confidence: High/Medium/Low)
            *   **Individual Portrayals:** How are political figures portrayed? Are they depicted as competent leaders or as reckless actors? (Analyze Individual Portrayal - Confidence: High/Medium/Low)


        **Phase 5:  Bias Assessment & Output**

        1.  **Dominant Bias:** Based on the analysis above, determine the dominant political bias present in the article.  Choose *one* of the following:
            *   Left-Wing: return -1
            *   Right-Wing: return 1
            *   Neutral: return 0
        2.  **Confidence Level for Bias:** Provide a confidence level for your bias assessment
        3.  **Justification:** Provide a concise (100-200 words) justification for your overall bias assessment. Summarize the key evidence you considered and explain why you believe the dominant bias is present. Acknowledge any areas of ambiguity or where the bias is subtle.

        **Output Format (Model Deliverable):**
        jsonstart
        "Title":[Article Title]
        "Source":[News Outlet]
        "Bias":[Left-Wing (-1) /Right-Wing (1)/Neutral (0)]
        "Confidence":[High/Medium/Low]
        "Justification":[Detailed Justification - 100-200 words]
        jsonend"""
            model = lgg.ChatGoogleGenerativeAI(model = "gemini-1.5-flash", google_api_key = GOOGLE_API_KEY, model_kwargs={"response_format": "json"} )
            bias = model.invoke(pmptbiascheck).content.strip(r"`\\njso")
            deaf = 0
        except:
            pass
    bias = json.loads(bias,strict =False)
    return bias



#Gives the desired bias to the generated response
def biaser(prompt, User, bias):
    mbias = bias
    bias = ["Centrist", "Right", "Left"][mbias]
    getbia = getbias(prompt)
    if getbia != bias:
      biasprompt = f"""You are a political and cultural narrative adaptation assistant. You specialize in rephrasing and augmenting answers to reflect the tone, reasoning style, and value system of specific ideological perspectives within the Indian context.

  You will receive the following inputs:

  query: The user's original question.

  answer: A factual or neutral response to that question.

  bias_direction: One of left, right, or centrist, indicating the desired ideological framing.

  Your task is to:
  Modify the answer to reflect the values, language, and logical perspective of the specified bias_direction.

  Incorporate reasoning consistent with Indian sociopolitical or cultural frameworks that resonate with the chosen stance.

  Remain faithful to the user’s query — the modified answer should still directly respond to the original question.

  Do not hallucinate facts — base your reasoning on plausible, known ideological logic and cultural context, not invention.

  Preserve factual accuracy, but allow interpretive emphasis, rhetorical choices, and culturally informed reasoning aligned with the bias.

  Ideological Guidance (India-specific):

  Left (liberal/progressive): Focus on individual rights, social equity, secularism, inclusivity, minority protections, critical views on tradition, emphasis on institutional integrity, not very strong sense of nationalism.

  Right (conservative/nationalist): Emphasize cultural heritage, national pride, traditional values, civilizational continuity, majority voice, cultural nationalism, social order.

  Centrist (moderate/pragmatic): Balance tradition and modernity, support institutional stability, pragmatic development, cultural cohesion, and inclusive growth,.

  Input Format:

  python

  query = {prompt}

  answer = {User}

  bias_direction = {bias}
  Output Format:

  json

  jsonstart
    "modified_answer": "A reasoned, culturally and politically contextualized answer based on the specified bias, faithful to the original query, and without fabricated content."
  jsonend

  Ensure that the answer should be about 400 words, bulet pointed and should maintain the reasoning contained in the original answer
  """ 
      t = model.invoke(biasprompt).content.strip(r"`\\njso")
      print("t==",t)
      r = json.loads(t,strict = False)
      return r["modified_answer"]
    else:
        return prompt
    



#Summarizes the entire chat history
def historymaker(history):
    mess = 1
    while mess == 1:
        try:
            prompt = f"""You are a chat history summarization assistant. Your task is to condense a conversation history into a concise summary that captures the main points, questions, and responses without losing essential context. The summary should be clear and easy to understand.
            your output should be condensed to 200 words or less.
            return output in a json format with the following structure:
            jsonstart
                summary: "summary"
            jsonend
            history: {history}
            """
            mess = json.loads(model.invoke(prompt).content.strip(r"`\\njso"),strict = False)
        except:
            pass
    return mess["summary"]




#Send user query to the backend and get response
def query_backend(user_input: str, view_mode) -> Dict[str, Any]:
    
    Bias_User=view_mode
    # try:
        # Prepare the request payload
    payload = {
        "query": user_input,
        "timestamp": time.time(),
        "view_mode": view_mode 
    }
    User = user_input
    response = main_runner(User,Bias_User,history)
# Check if request was successful
    if response:
        return response
    else:
        # Fallback for error cases
        return {
            "type": "summary",
            "response": f"Error connecting to backend: Status code {response.status_code}",
        }
    


#Initialise the frontend website
def get_initial_suggestions() -> List[Dict[str, Any]]:
    return [
        {
            "headline": "Welcome!",
            "summary": "Welcome! I'm your AI news assistant—ask me about any topic and I'll fetch the latest headlines and summaries instantly.",
            "link": "Start yor chat"
        },
        # {
        #     "headline": "Tech Giants Announce Collaboration on AI Ethics",
        #     "summary": "Major technology companies have formed a coalition to establish ethical guidelines for artificial intelligence development and deployment.",
        #     "link": "https://example.com/ai-ethics"
        # },
        # {
        #     "headline": "Healthcare Breakthrough: New Treatment for Alzheimer's",
        #     "summary": "Researchers have announced promising results from clinical trials of a new drug that significantly slows the progression of Alzheimer's disease.",
        #     "link": "https://example.com/alzheimers-treatment"
        # }
    ]

# --- Display Functions ---
def display_suggestions(suggestions: List[Dict[str, Any]]):
    st.markdown("<h3 style='font-family:Poppins,sans-serif;margin-top:0.8em;'>📰 Trending News</h3>", unsafe_allow_html=True)
    cols = st.columns(len(suggestions), gap="medium")
    for i, suggestion in enumerate(suggestions):
        with cols[i]:
            st.markdown(f"""
            <div class="gradient-border">
                <div class="gradient-border-inner" style="height:100%;">
                    <h4 style='font-family:Poppins,sans-serif;margin-bottom:0.3em;font-size:1.1rem;'>{suggestion['headline']}</h4>
                    <p style='font-size:1.01rem;'>{suggestion['summary']}</p>
                    # <a href="{suggestion['link']}" target="_blank">Read more →</a>
                </div>
            </div>
            """, unsafe_allow_html=True)



def display_news_grid(news_articles: List[Dict[str, Any]]):
    if not news_articles:
        return
    st.markdown("<h3 style='font-family:Poppins,sans-serif;margin-top:1.2em;'>Search Results</h3>", unsafe_allow_html=True)
    for row in range(2):
        cols = st.columns(3, gap="medium")
        for col in range(3):
            idx = row * 3 + col
            if idx < len(news_articles):
                article = news_articles[idx]
                with cols[col]:
                    st.markdown(f"""
                    <div class="gradient-border">
                        <div class="gradient-border-inner" style="height:100%;">
                            <h4 style='font-family:Poppins,sans-serif;margin-bottom:0.4em;font-size:1.07rem;'>{article['headline']}</h4>
                            <p style='font-size:1rem;'>{article['summary']}</p>
                            <a href="{article['link']}" target="_blank">Read full article →</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


#POST the LLM response
def display_response(message):
    """Display either a text summary or news grid based on message type"""
    st.markdown(message["content"])

    if message.get("type") == "summary":
        # For text summary, we just display the content which is already shown above
        pass
    elif "news_articles" in message:
        # For news grid, display the articles
        display_news_grid(message["news_articles"])

#Main function
def main_runner(query,Bias_User,history):
    outputsold, outputsnew = callrag(optiraggen(query), "Base")
    summary = summariser(outputsold, outputsnew)
    final = callfinal(summary[0], summary[1], historymaker(history))
    final = biaser(final, query, Bias_User)
    json_string = {"type" : "summary", "response": final }
    json_string = json.dumps(json_string)
    python_dict = json.loads(json_string,strict = False)
    history=history+"\nuser:query\nLLM:final"
    return python_dict

# --- Main UI Components ---
st.markdown("<h1 style='font-family:Poppins,sans-serif;font-weight:700;margin-bottom:0.4em;'>📰 Intelligent News Assistant</h1>", unsafe_allow_html=True)

# --- Sidebar for Settings ---
# with st.sidebar:
#     st.title("Settings")
#     backend_url = st.text_input(
#         "Backend URL",
#         value=st.session_state.backend_url,
#         help="URL of your backend API (e.g., http://localhost:8000)"
#     )

#     if backend_url != st.session_state.backend_url:
#         st.session_state.backend_url = backend_url
#         st.success("Backend URL updated!")

# --- Display Initial Suggestions ---
if not st.session_state.suggestions:
    st.session_state.suggestions = get_initial_suggestions()
display_suggestions(st.session_state.suggestions)

# --- Chat Interface ---
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            display_response(message)
        else:
            st.markdown(message["content"])
st.markdown("</div>", unsafe_allow_html=True)

# --- User Input and Theme Toggle ---
col1, col2 = st.columns([6,1])
with col1:
    prompt = st.chat_input("Ask me about any news topic...")
with col2:
    theme_btn_label = "🌙" if st.session_state.theme == "light" else "☀️"
    if st.button(theme_btn_label, key="theme_toggle", help="Toggle light/dark mode"):
        toggle_theme()
        st.rerun()

# --- Handle User Query with Loading Spinner ---
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from backend
    with st.spinner("Fetching..."):
        response_data = query_backend(prompt, view_numeric)

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response_data["response"])

        # Handle different response types
        if response_data.get("type") == "summary":
            # Just display the summary text which is already shown
            pass
        elif "news_articles" in response_data:
            display_news_grid(response_data["news_articles"])

    # Store the complete response in session state
    assistant_message = {
        "role": "assistant",
        "content": response_data["response"],
    }

    # Add type if present
    if "type" in response_data:
        assistant_message["type"] = response_data.get("type")

    # Add news articles if present
    if "news_articles" in response_data:
        assistant_message["news_articles"] = response_data["news_articles"]

    st.session_state.messages.append(assistant_message)
