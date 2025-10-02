import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import streamlit as st
from google import genai
import altair as alt

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("Please set GEMINI_API_KEY in your .env file")
    st.stop()


client = genai.Client(api_key=API_KEY)


@st.cache_data
def load_data():
    df = pd.read_csv("data.csv", quotechar='"')
    df.columns = [col.strip('"') for col in df.columns]      
    df["Title"] = df["Title"].str.strip('"')                
    return df

df = load_data()
titles = df["Title"].tolist()
urls = df["URL"].tolist()


@st.cache_resource
def create_embeddings(texts):
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=texts
    )
    return np.array([r.values for r in response.embeddings])

embeddings = create_embeddings(titles)


def retrieve(query, k=3):
    query_emb = create_embeddings([query])[0]
    sims = cosine_similarity([query_emb], embeddings)[0]
    top_idx = sims.argsort()[-k:][::-1]
    results = [(titles[i], urls[i], sims[i]) for i in top_idx]
    return results


def rag_answer(query, k=3):
    results = retrieve(query, k)
    context = "\n".join([f"{t} ({u})" for t, u, _ in results])
    
    prompt = f"""
You are a helpful assistant with access to NASA resources.

User query: {query}

Here are the most relevant documents:
{context}

Answer the question using the context above. Include the URLs as references.
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text, results


st.set_page_config(page_title="BioGraph", page_icon="🌍")
st.title("BioGraph")
st.write("Ask a question about NASA missions, space, Earth science and more!")

query = st.text_input("Enter your question:")
if st.button("Ask") and query.strip():
    with st.spinner("Thinking..."):
        answer, refs = rag_answer(query)
    
    st.subheader("Answer")
    st.write(answer)

    st.subheader("References")
    for t, u, s in refs:
        st.markdown(f"- [{t}]({u}) (score={s:.3f})")


    scores_df = pd.DataFrame(refs, columns=["Title", "URL", "Score"])
    scores_df = scores_df.sort_values("Score", ascending=False)


    chart = alt.Chart(scores_df).mark_bar().encode(
    x=alt.X("Score:Q", title="Confidence Score", axis=alt.Axis(format='%')),
    y=alt.Y("Title:N", sort="-x", title="Reference"),
    tooltip=["Title", alt.Tooltip("Score", title="Score (%)", format=".0f")]  
    ).properties(height=300, width=700)


    st.subheader("Confidence Scores")
    st.altair_chart(chart)


st.markdown(
    """
    <style>
    /* Hide hamburger menu */
    #MainMenu {visibility: hidden;}
    /* Hide footer ("Made with Streamlit") */
    footer {visibility: hidden;}
    /* Hide header (optional) */
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

