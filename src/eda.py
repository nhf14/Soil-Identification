import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

def run():
    st.header("🦣Exploratory Data Analysis (EDA)")

    def load_data():
        path = os.path.join('src')
        df = pd.read_csv(os.path.join(path, 'Original-Dataset.csv'))
        return df
    
    # 1. Dataset Overview
    st.subheader('🎶Dataset Overview')
    st.dataframe(load_data())


    # Membuat visualisasi jumlah gambar pada setiap kelas
    dis = load_data().groupby('label').agg(
        total = ('label','value_counts'),
        ).reset_index()

    figure = plt.figure(figsize=(8, 5))

    plt.bar(
        dis.label,
        dis.total
    )

    plt.title("Distribusi Gambar tiap Kelas")
    plt.xlabel("Nama Kelas")
    plt.ylabel("Jumlah Gambar")
    plt.xticks(rotation=15)

    # 2. Distribusi Gambar Tiap Kelas
    st.subheader('❤️Distribusi Gambar Tiap Kelas')
    st.pyplot(figure)

    st.markdown('''
    ### **Insight:**  
    - Kelas **Arid Soil** memiliki gambar berjumlah 284 
    - Kelas **Black Soil** memiliki gambar berjumlah 255
    - Kelas **Laterite Soil** memiliki gambar berjumlah 219 
    - Kelas **Mountain Soil** memiliki gambar berjumlah 201 
    - Kelas **Red Soil** memiliki gambar berjumlah 109 
    - Kelas **Yellow Soil** memiliki gambar berjumlah 69   
    - Kelas **Alluvial Soil** memiliki gambar berjumlah 51 
    ''')

    # # 2. Distribusi heart disease
    # st.subheader("Distribusi Dataset")
    # heart_counts = load_data()['HeartDisease'].value_counts().sort_index()

    # # Label
    # labels = ['Normal', 'Heart Disease']

    # # Membuat pie chart
    # pie_chart = plt.figure(figsize=(6,6))

    # plt.pie(
    #     heart_counts,
    #     labels=labels,
    #     autopct='%1.1f%%',
    #     startangle=90,
    #     colors=['steelblue', 'lightcoral'],
    #     explode=(0, 0.05),
    #     shadow=True
    # )

    # plt.title('Distribusi Heart Disease')
    # plt.axis('equal') 

    # st.pyplot(pie_chart)

    
if __name__=="__main__" :
    run()