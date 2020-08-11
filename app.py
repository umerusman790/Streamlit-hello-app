import streamlit as st

html_temp = """
    <div style="background-color:orange;padding:10px">
    <h1 style="color:white;text-align:center;">Titanic Crash APP </h1>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)

html_temp = """
    <div style="background-color:orange;padding:10px">
    <h1 style="color:white;text-align:center;">FA17-BSE-C-132</h1>
    </div>
    """
st.sidebar.markdown(html_temp,unsafe_allow_html=True)




data_set_name = st.sidebar.selectbox('Choose Dataset' ,
                                     ('Titanic Crash','Not Available'))

classifier_name = st.sidebar.selectbox('Choose Algorithm' ,
                                       ('Knn','Svm','Random Forest'))
data = pd.read_csv('dataconverted.csv' ,
                           index_col='Unnamed: 0')
st.write('# Dataset')
st.write(data.head())
