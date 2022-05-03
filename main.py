import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from gensim.utils import simple_preprocess, tokenize
from rank_bm25 import BM25Okapi
from time import time
from gensim import corpora, models, similarities
import numpy as np

top_k = 100
fn = "jobs-2022-04-01.csv"


@st.cache(allow_output_mutation=True)
def read_data(fn: str):
    df = pd.read_csv(fn)
    df['text_description_tokenized'] = df['text_description'].apply(
        lambda x: simple_preprocess(x))
    bm25 = BM25Okapi(df['text_description_tokenized'])
    dic = corpora.Dictionary(df['text_description_tokenized'])
    df['corpus'] = df['text_description_tokenized'].apply(
        lambda x: dic.doc2bow(x))
    lsi = models.LsiModel(df['corpus'], id2word=dic, num_topics=350)
    index = similarities.Similarity(
        'index', lsi[df['corpus']], num_features=lsi.num_topics)
    top_k = 5
    df['similarity'] = None
    for l, degrees in enumerate(index):
        df.at[l, 'similarity'] = np.argpartition(degrees, -top_k)[-top_k-1:-1]
    return df, bm25


def aggrid_interactive_table(df: pd.DataFrame):
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )

    options.configure_side_bar()

    options.configure_selection("single")
    selection = AgGrid(
        df,
        # enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="light",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )

    return selection


def query_job(search_q: str, df: pd.DataFrame, bm25: BM25Okapi):
    q_totenized = simple_preprocess(search_q)
    search_result = bm25.get_top_n(q_totenized, df.index, n=1000)
    return df.iloc[search_result]


def show_job_details(item: pd.Series):
    st.markdown(f"""
                ## Title
                {item['title']}
                ## Job Description
                {item['text_description']}
    """)
    with st.expander("See Full Job Detail"):
        st.write(item)


df, bm25 = read_data(fn)

with st.form("my_form"):
    search_q = st.text_input(label="Search a job")
    job_type = st.selectbox(
        'Job Type',
        ('Both', 'Internship', 'Job'))

    submitted = st.form_submit_button("Submit")

# query
t = time()
filted_df = query_job(search_q, df, bm25)
st.write(f"Query Time: {time() - t}s")

# filter
if job_type != 'Both':
    filted_df = filted_df[filted_df['job_type_name'] == job_type]

# show list
st.header("Search Result")
selection = aggrid_interactive_table(
    filted_df[['id', 'job_type_name', 'title']].head(top_k))


if selection["selected_rows"] != []:
    job_id = selection["selected_rows"][0]["id"]
    st.header("Job Detail")
    item = df[df['id'] == job_id].to_dict('records')[0]
    show_job_details(item)
    related_jobs = df.iloc[item["similarity"]]
    st.header("Related Jobs")
    st.write(related_jobs[['id','title', 'text_description']])
