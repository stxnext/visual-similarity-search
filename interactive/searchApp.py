import sys
import pathlib
import streamlit as st

# Streamlit needs help with accessing external modules - syspath addition resolves this problem.
modules_path = str(pathlib.Path().absolute()).split("/interactive")[0]
sys.path.append(modules_path)

from metrics.core import MetricClient


st.write(""" ## Closest similarity images for anchor image """)
st.write(""" #### Select inputs """)

category_option = st.selectbox(
    'What king of category would you like to choose comparisons from?',
    ('dogs', 'shoes'))

similar_img_number = st.number_input(
    label='Insert a number of similar images to show.',
    min_value=1,
    format='%i'
)

uploaded_file = st.file_uploader("Choose a file.")

if category_option and similar_img_number and uploaded_file:
    client = MetricClient()

    def search_with_show(collection: str, k: int, file=uploaded_file):
        anchor, similars = client._get_best_choice_for_uploaded_image(uploaded_file, collection, k=k)

        st.write(""" Anchor Image """)
        st.image(anchor)
        st.write(""" Similar Images """)
        st.image(similars)

    search_with_show(category_option, similar_img_number)