#Streamlit dependencies
import streamlit as st
import joblib, os
import pandas as pd
import numpy as np
import pickle
import re
import string
from skimage import io
import cv2
from skimage.transform import resize
import os
#from skimage.io import imread
#from sklearn.feature_extraction.text import CountVectorizer
#from nltk.tokenize import word_tokenize
#from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt
import tensorflow as tf
#from wordcloud import WordCloud
import base64
from skimage import io, transform
from PIL import Image
from io import BytesIO
from zipfile import ZipFile
import seaborn as sns
import requests
from io import BytesIO

import pickle



                                                #LOAD THE MODEL FRO GOOGLE

#with open('C:/Users/KgotsoPhela(LubanziI/Downloads/Lubanzi prjs/Pics/Compliance/model.pkl', 'rb') as f:
#    model = pickle.load(f)

#from google.oauth2.credentials import Credentials
#from googleapiclient.discovery import build

#creds = Credentials.from_authorized_user_file('Downloads/iic11s37agofikrmsdkdjk01h8nfn7f3.apps.googleusercontent.com.json', ['https://www.googleapis.com/auth/drive'])
#drive_service = build('drive', 'v3', credentials=creds)

#file_id = 'https://drive.google.com/file/d/1EA7tUeN-DhRU0NBT33kgOJoAy4XSPkC4/view?usp=sharing'
#request = drive_service.files().get_media(fileId=file_id)
#file_content = request.execute()

#with open('path/to/model.pkl', 'wb') as f:
#    f.write(file_content)

import urllib.request
import pickle

# Define the URL of the .pkl model file on your VPS
#https://drive.google.com/file/d/1EA7tUeN-DhRU0NBT33kgOJoAy4XSPkC4/view?usp=share_link
url = "https://drive.google.com/uc?export=download&id=1EA7tUeN-DhRU0NBT33kgOJoAy4XSPkC4"

# Fetch the model using urllib.request
with urllib.request.urlopen(url) as f:
    # Deserialize the model using pickle
    model = pickle.load(f)


                                            #PREDICTIONS

                                      # CREATE CONTAINERS
#tittle_and_welcome = st.container()
#image_input = st.container()
#folder_input = st.container()
#image_visuals = st.container()
#folder_visuals = st.container()
#predictions = st.container()

                                        # OUR PAGE TITTLE


# Set the background color of the app
#st.set_page_config(page_title="My Streamlit App", page_icon=":smiley:", layout="wide", initial_sidebar_state="expanded", background_color="#87CEFA")
#name = st.sidebar.text_input('Compliance App')



# Create a text input widget in the sidebar
#name = st.sidebar.text_input('Compliance App')

#st.set_option('server.enableCORS', True)



st.markdown("<h1 style='text-align: center; background-color: #cd8e07;'>Non Compliance Detector</h1>", unsafe_allow_html=True)


p = st.sidebar.markdown("<h1 style='text-align: center;'>SELECT PAGES BELOW</h1>", unsafe_allow_html=True)


#slid = st.sidebar.slider(' ', 1, 0)
reactive = st.sidebar.checkbox('**Reactive**')
st.sidebar.write('<span style="font-size: 8px;">Reactive focuses on the history of events or insidents</span>', unsafe_allow_html=True)
pro = st.sidebar.checkbox('**Proactive**')
st.sidebar.write('<span style="font-size: 8px;">Proactive focuses on what is happening at the moment</span>', unsafe_allow_html=True)
home_page = st.sidebar.checkbox('Experiment')
about = st.sidebar.checkbox('**About Us Page**')
#st.che



                                   #REACTIVE
if reactive:


    # Create a header in the main section of the app
    #st.markdown("<h1 style='text-align: center; background-color: orange;'>Non Compliance Detector</h1>", unsafe_allow_html=True)

    st.write('  ')
    st.write('____**Note that this section will nolonger exist once we have automated image collection process**____')
    st.write('<span style="font-size: 8px;">For now we still upload pictures here manually.</span>', unsafe_allow_html=True)
    #st.write('<span style="font-size: 8px;">For now we still upload pictures here manually.</span>', unsafe_allow_html=True')
    # Load the model and run the prediction
    #with open('C:/Users/KgotsoPhela(LubanziI/Downloads/Lubanzi prjs/Pics/Compliance/model.pkl', 'rb') as f:
    #    model = pickle.load(f)



    #tt = st.markdown("<h2 style = 'text-align: center';>Click to upload images</h2>", unsafe_allow_html=True)
    #tt = st.markdown("<h1 style = 'text-align: center';>Non Compliance Detector</h1>", unsafe_allow_html=True)



    with st.expander("**Click to upload images**"):
        # Create a file uploader that is meant to take in zipped folders 
        multi_upload = st.file_uploader("Upload zipped files", accept_multiple_files=True)

    # Unzip or extract the contents of the folder
    if multi_upload is not None and len(multi_upload) > 0:


        with st.expander('**View Uploaded Images and Detected Classes**'):

            results = []
            folders = []

            n = st.number_input('**Select Grid Width**', 1, 5, 3)
            visualise_folders = st.button('Classify and View Uploads')
            if visualise_folders:
                for zipped_file in multi_upload:
                    with ZipFile(zipped_file) as zip:
                        zip.extractall()
                        folders.append(zipped_file.name.split('.')[0])
                        
                #Classify the images in each folder
                for folder in folders:
                    st.subheader(folder)

                    files = [f for f in os.listdir(folder) if f.endswith(".JPG") or f.endswith(".PNG") or f.endswith(".jpg") or f.endswith("png")]

                    # Display the images in columns according to the grid width
                    for i in range(0, len(files), n):

                        cols = st.columns(n)
                        for j in range(n):
                            if i + j < len(files):
                                file = files[i+j]
                                img_path = os.path.join(folder, file)
                                np_img = np.array(Image.open(img_path).convert('RGB').resize((256, 256)))
                                yhat = model.predict(tf.expand_dims(np_img, 0))
                                #st.dataframe(yhat)
                                pred_class = '**Seat Belt On**' if yhat > 0.5 else '**Seat Belt Off**'
                                color = 'green' if yhat > 0.5 else 'red'

                                with cols[j]:
                                    st.write(f'<span style="color:{color}">{pred_class}</span>', unsafe_allow_html=True)
                                    st.image(np_img, caption=None, use_column_width=True)




                                    # OUTCOME VISUALISATION, TABLE STATS



        with st.expander('**Outcomes Summary Table**'):

            # Create empty lists to store the results
            #st.write('**Stats Table**')
            reg_numbers = []
            seat_belt_on_counts = []
            seat_belt_off_counts = []

            # Loop through each folder
            for folder in folders:
                # Get the registration number from the folder name
                reg_number = folder.split('_')[0]
                reg_numbers.append(reg_number)

                # Count the number of images classified as seat belt on and seat belt off
                seat_belt_on_count = 0
                seat_belt_off_count = 0
                files = [f for f in os.listdir(folder) if f.endswith(".JPG") or f.endswith(".PNG") or f.endswith(".jpg") or f.endswith("png")]
                for file in files:
                    img_path = os.path.join(folder, file)
                    np_img = np.array(Image.open(img_path).convert('RGB').resize((256, 256)))
                    yhat = model.predict(tf.expand_dims(np_img, 0))
                    if yhat > 0.5:
                        seat_belt_on_count += 1
                    else:
                        seat_belt_off_count += 1
                seat_belt_on_counts.append(seat_belt_on_count)
                seat_belt_off_counts.append(seat_belt_off_count)

            # Create the dataframe

            total_counts = [seat_belt_on_counts[i] + seat_belt_off_counts[i] for i in range(len(seat_belt_on_counts))]



            # Create a function to determine the region based on the registration number
            def get_region(reg_number):
                if reg_number == 'KK 08 PZ GP':
                    return 'Midrand'
                elif reg_number == 'MY 85 QY GP':
                    return 'Centurion'
                
                elif reg_number == 'NN 76 ZB GP':
                    return 'Centurion'

                elif reg_number == 'ON 72 BX GP':
                    return 'Centurion'
                else:
                    return 'Unknown'
            #Create func to get company deployed to
            def get_company(reg_number):
                if reg_number == 'KK 08 PZ GP':
                    return 'Lubanzi'
                elif reg_number == 'NN 76 ZB GP':
                    return 'Imizizi'
                
                elif reg_number == 'ON 72 BX GP' :
                    return 'Imizizi'

                elif reg_number == 'MY 85 QY GP':
                    return 'Lubanzi'
                else:
                    return 'Unknown'
            #Create func to grt Responsible Manager
            def get_resp_manager(reg_number):
                if reg_number == 'KK 08 PZ GP':
                    return 'Sipho'
                
                elif reg_number == 'MY 85 QY GP':
                    return 'Kgotso'
                
                #elif reg_number == "NN 76 ZB GP":
                #    return 'Thotyelwa'
                
                elif reg_number == "NN 76 ZB GP":
                    return 'Hope'
                
                elif reg_number == 'ON 72 BX GP':
                    return 'Thotyelwa'


                else:
                    return 'Unknown'

            # Create the dataframe
            data = {'Incident Date & Time': 'coming soon...',
                    'Speed': '...',
                    'Registration No': reg_numbers,
                    'SB on': seat_belt_on_counts,
                    'SB off': seat_belt_off_counts,
                    'Total number of pictures': total_counts}

            df = pd.DataFrame(data)

            # Add the Region column to the dataframe
            df['Region'] = df['Registration No'].apply(get_region)
            df['Company Deployed To'] = df['Registration No'].apply(get_company)
            df['Responsible Manager'] = df['Registration No'].apply(get_resp_manager)

            # Display the dataframe in your Streamlit app
            st.experimental_data_editor(df)



            #df2 = pd.DataFrame(data2)
            #df2['Predictions'] = yhat
            #st.dataframe(df2)
            
        with st.expander('**Outcomes Complete Table**'):
            # Loop through each folder
            #st.write('**------------------------------------------------FULL DETAILS TABLE-----------------------------------------------------**')
            predictions = []
            reg_numbers = []
            for folder in folders:
                # Get the registration number from the folder name
                reg_number = folder.split('_')[0]
                files = [f for f in os.listdir(folder) if f.endswith(".JPG") or f.endswith(".PNG") or f.endswith(".jpg") or f.endswith("png")]
                for file in files:
                    img_path = os.path.join(folder, file)
                    np_img = np.array(Image.open(img_path).convert('RGB').resize((256, 256)))
                    yhat = model.predict(tf.expand_dims(np_img, 0))
                    predictions.append(yhat[0][0])
                    reg_numbers.append(reg_number)

            def get_class(predictions):
                if predictions > 0.5:
                    return 'on'
                else:
                    return 'off'
                
            

            

            data2 = {'Incident Date & Time': 'coming soon...',
                    'Speed': '...',
                    'Registration No': reg_numbers,
                    'Predictions': predictions}

            df2 = pd.DataFrame(data2)

            df2['SB_satus'] = df2['Predictions'].apply(get_class)

            def get_compl(SB_status):
                if SB_status == 'on':
                    return 'Comp'
                else:
                    return 'Non-Comp'
                

            
            df2['Comp_stat'] = df2['SB_satus'].apply(get_compl)
            df2['Vehicle Region'] = df2['Registration No'].apply(get_region)
            df2['Company Deployed To'] = df2['Registration No'].apply(get_company)
            df2['Responsible Manager'] = df2['Registration No'].apply(get_resp_manager)
            


            st.dataframe(df2)
            st.write('Note that the Comp_status will be determined by speed once we have the values. If the speed is above the threshold then we are going to have non compliant and if the speed is below the threshold we will have compliant')
            









                                        # GRAPHICAL REPRESENTATION OF OUR DATA


        with st.expander('**Data**'):

            df_melted = pd.melt(df, id_vars=['Registration No'], value_vars=['SB on', 'SB off'], var_name='Seat belt status')

            df_melted = pd.melt(df, id_vars=['Responsible Manager'], value_vars=['SB on', 'SB off'], var_name='Seat belt status')
                
            import plotly.express as px

                # Create example data
                #data = {'Responsible Manager': ['John', 'Sarah', 'Mike'],
                #        'SB on': [20, 10, 5],
                #        'SB off': [5, 8, 12]}
                #df = pd.DataFrame(data)

                # Melt the data for plotting
            df_melted = pd.melt(df, id_vars=['Responsible Manager', 'Region'], value_vars=['SB on', 'SB off'], var_name='Seat belt status')


            sm = df_melted['value'].sum()
            st.write('Our sum of Seat belt status **(SB on + SB of)** values is ' + str(sm))
            #st.write(df_melted)
            #st.write(df_melted['value'].sum() / df_melted['value']df_melted['Seat belt status'] == 'on'.sum())
            m_data = df_melted.loc[df_melted['Seat belt status'] == 'SB off']

                                #TRY FILTERS

                # filtering the dataframe by region
            selected_region = st.multiselect('Select Regions:', options=m_data['Region'].unique())

                # Filter the dataframe based on the selected departments
            if selected_region:
                filtered_df = m_data[m_data['Department'].isin(selected_region)]
            else:
                filtered_df = m_data

                # Display the filtered dataframe
            st.table(filtered_df)



        try:
            with st.expander("**Graphical Representation of outcomes**"):
                # Create a figure with a specific size
                # Melt the dataframe to a long format



                df_melted = pd.melt(df, id_vars=['Registration No'], value_vars=['SB on', 'SB off'], var_name='Seat belt status')

                # Create the barplot using seaborn
                fig = plt.figure(figsize=(10,4))
                sns.barplot(data=df_melted, x='Registration No', y='value', hue='Seat belt status', estimator=np.sum)

                # Show the plot
                sns.set(rc={"figure.figsize": (2, 20)})
                st.pyplot(fig)



                df_melted = pd.melt(df, id_vars=['Responsible Manager'], value_vars=['SB on', 'SB off'], var_name='Seat belt status')

                # Create the barplot using seaborn
                fig = plt.figure(figsize=(10,4))
                sns.barplot(data=df_melted, x='Responsible Manager', y='value', hue='Seat belt status', estimator=np.sum)

                # Show the plot
                sns.set(rc={"figure.figsize": (2, 20)})
                st.pyplot(fig)
                


                import plotly.express as px

                # Create example data
                #data = {'Responsible Manager': ['John', 'Sarah', 'Mike'],
                #        'SB on': [20, 10, 5],
                #        'SB off': [5, 8, 12]}
                #df = pd.DataFrame(data)

                # Melt the data for plotting
                df_melted = pd.melt(df, id_vars=['Responsible Manager', 'Region'], value_vars=['SB on', 'SB off'], var_name='Seat belt status')

                # Create pie chart using plotly
                fig = px.pie(df_melted, names='Seat belt status', values='value', 
                            hole=0.5, color_discrete_sequence=['#1f77b4', '#ff7f0e'], title = 'What percentage of our images were classified as SB on and SB off?',
                            labels={'value': 'Count'})

                # Display pie chart in Streamlit app
                st.plotly_chart(fig)
                sm = df_melted['value'].sum()
                st.write('Our sum of Seat belt status **(SB on + SB of)** values is ' + str(sm))
                st.write(df_melted)
                #st.write(df_melted['value'].sum() / df_melted['value']df_melted['Seat belt status'] == 'on'.sum())
                m_data = df_melted.loc[df_melted['Seat belt status'] == 'SB off']


                fig = px.pie(m_data, names='Responsible Manager', values='value', 
                            hole=0.5, color_discrete_sequence=['#1f77b4', '#ff7f0e'], title = 'SB Non Compliance per Manager',
                            labels={'value': 'Count'})
                st.plotly_chart(fig)

                # Create the barplot using seaborn
                fig = plt.figure(figsize=(10,4))
                sns.barplot(data=m_data, x='Responsible Manager', y='value', hue='Region', estimator=np.sum)

                # Show the plot
                sns.set(rc={"figure.figsize": (2, 20)})
                st.pyplot(fig)

                st.write('SB Non Compliance per Manager Table')
                #st.write(m_data)
                                #TRY FILTERS

                # filtering the dataframe by region
                selected_region = st.multiselect('Select Regions:', options=m_data['Region'].unique())

                # Filter the dataframe based on the selected departments
                if selected_region:
                    filtered_df = m_data[m_data['Department'].isin(selected_region)]
                else:
                    filtered_df = m_data

                # Display the filtered dataframe
                st.table(filtered_df)

                #st.write(df_melted)



                #filtered_df = df.loc[df['Age'] >= 35]






                #st.snow()
                #st.balloons()


                #df_melted2 = pd.melt(df2, id_vars = ['Registration No'], value_vars = ['SB_status', ])

        except Exception as e:
            st.write('**Classify in order to see this section.**')#: ' + str(e))


elif pro:
    #st.markdown("<h1 style='text-align: center; background-color: orange;'>Non Compliance Detector</h1>", unsafe_allow_html=True)
    st.write(' ')
    with st.expander("**images**"):
        # Create a file uploader that is meant to take in zipped folders 
        multi_upload = st.file_uploader("Upload zipped files", accept_multiple_files=True)

        # Unzip or extract the contents of the folder
        if multi_upload is not None and len(multi_upload) > 0:

            st.write(' ')
            st.write('**This section is still under development**')
            st.write('Soon as the snapshots are uploaded automatically, our model will go through each image and look for non-compliances like the officer would. If the model detects one, it will then provide or display the registration number of the vehicle in which the non-compliance was detected in the Non-Compliance Board which is still under development.')


    with st.expander('**Open Non-compliance Board**'):
        response = requests.get('https://raw.githubusercontent.com/KgotsoPhela/Compliance/main/Black_Technology_LinkedIn_Banner_2.png')
        im = Image.open(BytesIO(response.content))
        st.image(im)
        non_compliant_folders = []

        n = 5
        #visualise_folders = st.button('Open Non Compliance Board')
        #if visualise_folders:
        for zipped_file in multi_upload:
            with ZipFile(zipped_file) as zip:
                zip.extractall()
                folder_name = zipped_file.name.split('.')[0]
                files = [f for f in os.listdir(folder_name) if f.endswith(".JPG") or f.endswith(".PNG") or f.endswith(".jpg") or f.endswith("png")]

                    # Classify the images in each folder
                has_off_prediction = False
                for i in range(0, len(files), n):
                    cols = st.columns(n)
                    for j in range(n):
                        if i + j < len(files):
                            file = files[i+j]
                            img_path = os.path.join(folder_name, file)
                            np_img = np.array(Image.open(img_path).convert('RGB').resize((256, 256)))
                            yhat = model.predict(tf.expand_dims(np_img, 0))
                            pred_class = '**Seat Belt On**' if yhat > 0.5 else '**Seat Belt Off**'

                            color = 'green' if yhat > 0.5 else 'red'

                            with cols[j]:
                                #st.write(f'<span style="color:{color}">{pred_class}</span>', unsafe_allow_html=True)
                                if pred_class == '**Seat Belt Off**':
                                    has_off_prediction = True

                if has_off_prediction:
                    non_compliant_folders.append(folder_name)

        # Display the non-compliant folders
        if non_compliant_folders:
            st.write('**See Registrations Our model Suspects have Non-compliant Behaviors Below:**')
            for folder_name in non_compliant_folders:
                nc = st.checkbox(folder_name)

                if nc:
                    
                    st.write('This should take you to the live streaming of:    ' + '  '+ folder_name)
                    st.write('And this functionality is still under development...')
        else: 
            st.write('<span style="color: green;">**No Violations Observed**</span>', unsafe_allow_html=True)

elif home_page:
    
    css_styles = '''
    <style>
    body {
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 0;
    }

    header {
        background-color: #aca5912d;
        color: black;
        padding: 20px;
    }

    h1 {
        margin: 0;
    }

    nav ul {
        list-style-type: none;
        margin: 0;
        padding: 0;
    }

    nav li {
        display: inline-block;
        margin-right: 20px;
    }

    nav li:last-child {
        margin-right: 0;
    }

    nav a {
        color: black;
        text-decoration: none;
    }

    main {
        padding: 20px;
    }

    section {
        margin-bottom: 20px;
    }

    footer {
        background-color: #cd8e07;
        color: rgb(81, 76, 76);
        padding: 20px;
        text-align: center;
    }

    nav ul li:nth-child(2) a {
    color: black; /* or any other color you prefer */
    font-weight: bold;
    }

    nav ul li:nth-child(1) a {
    color: black; /* or any other color you prefer */
    font-weight: bold;
    }

    nav ul li:nth-child(3) a {
    color: black; /* or any other color you prefer */
    font-weight: bold;
    }

    nav ul li:nth-child(4) a {
    color: black; /* or any other color you prefer */
    font-weight: bold;
    }
    </style>
    '''

    html_code = '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Compliance Solution</title>
        <link rel="stylesheet" href="style.css">
    </head>
    <body>
        <header>
        <nav>
            <ul>
            <li><a href="#">HOME</a></li>
            <li><a href ="http://localhost:8501/">REACTIVE</a></li>
            <li><a href="#">PROACTIVE</a></li>
            <li><a href="#">CONTACTS</a></li>
            </ul>
        </nav>
        </header>
    </body>
    </html>
    '''  
    st.markdown(html_code, unsafe_allow_html=True)
    st.markdown(css_styles, unsafe_allow_html=True)
    #st.markdown(html2, unsafe_allow_html=True)


        
        


    response = requests.get('https://raw.githubusercontent.com/KgotsoPhela/Compliance/main/s.png')
    imgg = Image.open(BytesIO(response.content))
    st.image(imgg)
    # Create a dataframe
    data = {'Name': ['John', 'Emily', 'Sarah', 'David'],
            'Age': [25, 28, 32, 19],
            'City': ['New York', 'Paris', 'London', 'Tokyo']}
    df = pd.DataFrame(data)

    # Display the dataframe as a table using st.table()
    st.table(df)


elif about:
    st.write(' ')
    st.write('**About Us Coming Soon...**')

    # Create a sample dataframe
    df = pd.DataFrame({
        'Name': ['Hope', 'Joreto', 'Sandile', 'Dineo', 'Kgotso'],
        'Age': [28, 31, 45, 22, 39],
        'Salary': [50000, 75000, 60000, 40000, 50000],
        'Department': ['Marketing', 'Sales', 'IT', 'HR', 'Finance']
    })

    # Create a sidebar for filtering the dataframe by department
    selected_dept = st.multiselect('Select departments:', options=df['Department'].unique())

    # Filter the dataframe based on the selected departments
    if selected_dept:
        filtered_df = df[df['Department'].isin(selected_dept)]
    else:
        filtered_df = df

    # Display the filtered dataframe
    st.table(filtered_df)

    options = ["Option 1", "Option 2", "Option 3"]
    selected_option = st.radio("Select an option", options)

    if selected_option == "Option 1":
        st.write("You selected Option 1")
    elif selected_option == "Option 2":
        st.write("You selected Option 2")
    else:
        st.write("You selected Option 3")

else:
    st.write('')



import webbrowser

# Define a function to open the HTML file in the same tab/window



# Define a function to open the HTML file in the same tab
#def open_html_file():
#    html_file = open("file:///C:/Users/KgotsoPhela(LubanziI/Downloads/Lubanzi%20prjs/Pics/Compliance/Compliance%20Streamlit%20App/Web.html", "r")
#    source_code = html_file.read()
#    print(source_code)
#    components.html(source_code, height=1000)

# Display a button that, when clicked, will call the open_html_file function
#if st.sidebar.button('Back to Home Page'):
#    open_html_file()





#else:
    #st.markdown("<h1 style='text-align: center; background-color: orange;'>Non Compliance Detector</h1>", unsafe_allow_html=True)
    #image = Image.open('C:/Users/KgotsoPhela(LubanziI/Desktop/snp.png')
    #st.image(image, caption='Think automation')


           #st.write('coming soon...')

    # Load the image
    #image = Image.open('C:/Users/KgotsoPhela(LubanziI/Downloads/Lubanzi prjs/Pics/Slides Pics/Lubanzi_Profile and banner pics/1667928151730.jpg')

    # Display the image
    #st.image(image, caption='Think automation')






























        # Load the image from the specified directory
    #img = io.imread(directory)

    # Resize the image using TensorFlow
    #resize = tf.image.resize(img, (256, 256))

    # Display the resized image using Matplotlib
    #fig, ax = plt.subplots()
    #ax.imshow(resize.numpy().astype(int))
    #ax.set_title('Resized Image')
    #ax.axis('off')
    #st.pyplot(fig)


st.markdown(
    """
    <style>
    body {
        background-image: url('Merchant-Fleet-Management.jpg');
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)