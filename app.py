import pandas as pd 
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

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

st.write('# Dataset Information')
info = st.radio('Metadata',('Shape','Ndim','Dtype','Index'))
if info == 'Shape':
    st.write(data.shape)
elif info == 'Ndim':
    st.write(data.ndim)
elif info =='Dtype':
    st.write(data.dtypes)
elif info == 'Index':
    st.write(data.index)


def get_dataset(name):
    data =None
    if name == 'Titanic Crash':
        data = pd.read_csv('dataconverted.csv' ,
                           index_col='Unnamed: 0')
    else:
        st.warning('dataset isn\'t avalilable ')
        
    X =data.drop('Survived' , axis=1)
    y =data['Survived']
    
    return  X,y

X , y = get_dataset(data_set_name)


    
 
    
def add_parameter_ui(c_name):
    parameter = dict()
    if c_name == 'Svm':
        kernel =st.sidebar.radio('kernel' ,
                                 ('linear', 'poly', 'rbf', 'sigmoid'))
        parameter['kernel'] =kernel
    elif c_name == 'Knn':
        K = st.sidebar.slider('K',1,100)
        parameter['K'] =K
    else:
        criterion= st.sidebar.radio('criterion',('gini','entropy') )
        parameter['criterion'] = criterion
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        parameter['n_estimators'] = n_estimators
    return parameter


parameter = add_parameter_ui(classifier_name)


def get_classifier(clf_name , parameter):
    clf =None
    
    if clf_name == 'Svm':
        clf = SVC(kernel=parameter['kernel'])
        
    elif clf_name == 'Knn':
        clf = KNeighborsClassifier(n_neighbors=parameter['K'])
        
    else:
        clf = RandomForestClassifier(n_estimators=parameter['n_estimators'],
                                     criterion=parameter['criterion'],random_state=11)
    return clf

classfi_name = get_classifier(classifier_name,parameter)




################### Training Model #############

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)
classfi_name.fit(X_train, y_train)
y_pred = classfi_name.predict(X_test)

################### Training End   ############


###################  measuring fucntions ##########

st.write('# Dataset Performance Measures')
acc    = accuracy_score(y_test, y_pred)
f1     = f1_score(y_test, y_pred, average='binary')
p      = precision_score(y_test, y_pred, average='binary')

st.write("Accuracy = " ,acc)
st.write("Precision= " ,p)
st.write("F1_score = " ,f1)

################### Functions End ##################


############################# user inputs  ###########    
st.title('Testing Unseen Instance')
    
pclass   = st.number_input('Enter Passenger Class',1,3)
sibling  = st.number_input('Enter Number Of Sibling',0,5,1)
gender   = st.number_input('Enter Gender',0,1,1)
embark_q = st.number_input('Enter Embarked_Q',0,1,1)
embark_s = st.number_input('Enter Embarked_S',0,1,1)

prediction = classfi_name.predict([[pclass,sibling,gender,embark_q,embark_s]])

if st.button('Submit'):
    if prediction == 1:
        st.success('Congrats, Passenger Survived')
    else:
        st.warning('Alas, Passenger Died')

#################  user  input ends ################
html_temp = """
    <div style="background-color:tomato;padding:2px">
    <h1 style="color:white;text-align:center;">To: Sir Rao Adeel Nawab</h1>
    </div>
    """
st.sidebar.markdown(html_temp,unsafe_allow_html=True)
      
from PIL import Image
img = Image.open('ss.jpg')
st.sidebar.image(img ,width =300 )
