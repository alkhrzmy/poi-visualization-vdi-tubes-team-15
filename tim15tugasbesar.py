import os
import re
import math
from urllib.parse import unquote
import pandas as pd
import seaborn as sns
from plotly.subplots import make_subplots
import streamlit as st
import zipfile
import networkx as nx

# Standard library imports

# Data manipulation and analysis

# Data visualization
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Web framework

# File handling


@st.cache_data
def unzip_and_load():
    # -- Pastikan path untuk zip & extract sudah sesuai di environment Anda --
    zip_path = 'data-ijcai15.zip'
    extract_path = 'data-ijcai15/'
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    # --------------------------------------------------------------------------
    # Memuat costProfCat
    # --------------------------------------------------------------------------
    directory_cost = os.path.join(extract_path, 'costProf-ijcai15')
    dfs_cost = []
    for filename in os.listdir(directory_cost):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory_cost, filename)
            try:
                df_temp = pd.read_csv(filepath, delimiter=';')
                match = re.search(r'costProfCat-(.+?)POI-all', filename)
                if match:
                    city_name = match.group(1)
                else:
                    city_name = "Unknown"
                df_temp['cities'] = city_name
                dfs_cost.append(df_temp)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    costProfCat = pd.concat(dfs_cost, ignore_index=True)

    # --------------------------------------------------------------------------
    # Memuat poiList
    # --------------------------------------------------------------------------
    directory_poi = os.path.join(extract_path, 'poiList-ijcai15')
    dfs_poi = []
    for filename in os.listdir(directory_poi):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory_poi, filename)
            try:
                df_temp = pd.read_csv(filepath, delimiter=';')
                match = re.search(r'POI-(.+?)\.csv', filename)
                if match:
                    city_name = match.group(1)
                else:
                    city_name = "Unknown"
                df_temp['cities'] = city_name
                dfs_poi.append(df_temp)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    poiList = pd.concat(dfs_poi, ignore_index=True)

    # --------------------------------------------------------------------------
    # Memuat userVisits
    # --------------------------------------------------------------------------
    directory_visits = os.path.join(extract_path, 'userVisits-ijcai15')
    dfs_visits = []
    for filename in os.listdir(directory_visits):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory_visits, filename)
            try:
                df_temp = pd.read_csv(filepath, delimiter=';')
                name_parts = filename[:-4].split('-') 
                if len(name_parts) > 2:
                    city_name = name_parts[1]
                else:
                    city_name = name_parts[-1]
                df_temp['cities'] = city_name
                dfs_visits.append(df_temp)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    userVisits = pd.concat(dfs_visits, ignore_index=True)

    # --------------------------------------------------------------------------
    # Membersihkan poiName
    # --------------------------------------------------------------------------
    poiList['poiName'] = poiList['poiName'].apply(lambda x: unquote(x))
    poiList['poiName'] = poiList['poiName'].apply(lambda x: unquote(x))
    poiList['poiName'] = poiList['poiName'].str.replace('_', ' ')

    # --------------------------------------------------------------------------
    # Pastikan dateTaken jadi datetime (jika ada)
    # --------------------------------------------------------------------------
    if 'dateTaken' in userVisits.columns:
        # Misalnya type-nya second-based
        try:
            userVisits['dateTaken'] = pd.to_datetime(userVisits['dateTaken'], unit='s', errors='coerce')
        except:
            # Jika ada format lain, dapat disesuaikan
            userVisits['dateTaken'] = pd.to_datetime(userVisits['dateTaken'], errors='coerce')

    return costProfCat, poiList, userVisits

def top_10_popular(userVisits, poiList):
    merged_df = pd.merge(userVisits, poiList, on=['poiID', 'cities'], how='left')
    visit_counts = merged_df.groupby(['cities', 'poiName']).size().reset_index(name='visit_count')
    top_pois = visit_counts.sort_values(['cities','visit_count'], ascending=[True,False]).groupby('cities').head(10)
    fig = px.bar(top_pois, x='poiName', y='visit_count', color='cities', title='Top 10 POI Terpopuler di Setiap Kota')
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def barplot_top_10(userVisits, poiList):
    # Menggabungkan userVisits + poiList
    userVisits_poi = pd.merge(userVisits, poiList, on=['poiID','cities'], how='left')
    # Ambil poiFreq
    poi_freq = userVisits_poi[['poiID','cities','poiName','poiFreq']].drop_duplicates()
    # Plot per kota
    cities = poi_freq['cities'].unique()
    n_cols = 2
    n_rows = math.ceil(len(cities)/n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15,5*n_rows))
    axes = axes.flatten()
    
    for i, city in enumerate(cities):
        city_data = poi_freq[poi_freq['cities']==city].sort_values('poiFreq', ascending=False).head(10)
        sns.barplot(data=city_data, x='poiFreq', y='poiName', palette='viridis', ax=axes[i])
        axes[i].set_title(f'Top 10 POI Terpopuler di Kota {city}')
        axes[i].set_xlabel('Frekuensi Kunjungan')
        axes[i].set_ylabel('Nama POI')
    
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.tight_layout()
    return fig

def stacked_theme(userVisits, poiList):
    userVisits_poi = pd.merge(userVisits, poiList, on=['poiID','cities'], how='left')
    poi_freq_theme = userVisits_poi[['cities','poiTheme','poiFreq','poiID']].drop_duplicates()
    theme_city_freq = poi_freq_theme.groupby(['cities','poiTheme'])['poiFreq'].sum().reset_index()
    pivot_table = theme_city_freq.pivot(index='cities', columns='poiTheme', values='poiFreq').fillna(0).reset_index()

    fig = px.bar(
        pivot_table,
        x='cities',
        y=pivot_table.columns[1:],
        title='Distribusi Tema POI di Setiap Kota',
        labels={'value': 'Total Frekuensi Kunjungan', 'cities': 'Kota'},
        barmode='stack',
        width=1000  # Adjust the width of the plot
    )
    fig.update_layout(xaxis_title='Kota', yaxis_title='Total Frekuensi Kunjungan', legend_title='Tema POI')
    return fig

def heatmap_cost(costProfCat):
    avg_cost = costProfCat.groupby(['cities', 'category'])['cost'].mean().reset_index()
    pivot_table = avg_cost.pivot(index='category', columns='cities', values='cost')

    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale=[
            [1, 'rgb(165,0,38)'],   # Lowest
            [0.5, 'rgb(255,255,190)'], # Middle
            [0, 'rgb(0,104,55)']    # Highest
        ],
        hoverongaps=False,
        text=pivot_table.values,
        texttemplate="%{text:.2f}",
        textfont={"size":12}
    ))

    fig.update_layout(
        title='Biaya Rata-rata per Kategori POI di Setiap Kota',
        xaxis_title='Kota',
        yaxis_title='Kategori POI',
        width=800,
        height=600
    )

    return fig


def scatter_cost_profit(costProfCat):
    Q1 = costProfCat['cost'].quantile(0.25)
    Q3 = costProfCat['cost'].quantile(0.75)
    IQR = Q3 - Q1
    lb = Q1 - 1.5*IQR
    ub = Q3 + 1.5*IQR
    costProfCat_no_outliers = costProfCat[(costProfCat['cost']>=lb)&(costProfCat['cost']<=ub)]
    fig, ax = plt.subplots(figsize=(10,6))
    sns.scatterplot(data=costProfCat_no_outliers, x='cost', y='profit', hue='category', palette='tab20', alpha=0.7, ax=ax)
    ax.set_title('Hubungan Biaya dan Keuntungan (Tanpa Outliers)')
    ax.set_xlabel('Biaya')
    ax.set_ylabel('Keuntungan')
    ax.legend(title='Kategori POI', bbox_to_anchor=(1.05,1), loc='upper left')
    fig.tight_layout()
    return fig

def line_trend_visits(userVisits):
    df_filtered = userVisits[userVisits['dateTaken'].dt.year >= 2005].copy()
    df_filtered['Year_Month'] = df_filtered['dateTaken'].dt.to_period('M')
    monthly_visits = df_filtered.groupby(['cities','Year_Month']).size().reset_index(name='Visit_Counts')
    monthly_visits['Year_Month'] = monthly_visits['Year_Month'].astype(str)
    monthly_visits['Year_Month'] = pd.to_datetime(monthly_visits['Year_Month'])
    
    fig, ax = plt.subplots(figsize=(12,6))
    sns.lineplot(data=monthly_visits, x='Year_Month', y='Visit_Counts', hue='cities', marker='o', ax=ax)
    ax.set_title('Jumlah Kunjungan per Bulan di Setiap Kota (2005 ke atas)')
    ax.set_xlabel('Waktu')
    ax.set_ylabel('Jumlah Kunjungan')
    ax.legend(title='Kota')
    plt.xticks(rotation=45)
    fig.tight_layout()
    return fig

def bar_profit_category(costProfCat):
    avg_profit = costProfCat.groupby('category')['profit'].mean().reset_index()
    avg_profit = avg_profit.sort_values('profit', ascending=False)
    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(data=avg_profit, x='profit', y='category', palette='Spectral', ax=ax)
    ax.set_title('Keuntungan Rata-rata per Kategori POI')
    ax.set_xlabel('Keuntungan Rata-rata')
    ax.set_ylabel('Kategori POI')
    fig.tight_layout()
    return fig

def map_city(poiList, city='Toro'):
    city_poi = poiList[poiList['cities'] == city]
    fig = px.scatter_mapbox(
        city_poi,
        lat='lat',
        lon='long',
        hover_name='poiName',
        hover_data=['theme'],
        zoom=12,
        mapbox_style='carto-positron',
        title=f'Lokasi POI di Kota {city}'
    )
    fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
    return fig

def pie_theme_city(userVisits, city='Toro'):
    city_data = userVisits[userVisits['cities'] == city]
    city_poi_freq = city_data[['poiID', 'poiTheme', 'poiFreq']].drop_duplicates()
    theme_freq = city_poi_freq.groupby('poiTheme')['poiFreq'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(8,8))
    ax.pie(theme_freq['poiFreq'], 
            labels=theme_freq['poiTheme'], 
            autopct='%1.1f%%',
            startangle=140, 
            colors=sns.color_palette('pastel'))
    ax.set_title(f'Distribusi Tema POI di Kota {city}')
    ax.axis('equal')
    fig.tight_layout()
    return fig

def create_poi_network(costProfCat, poiList, selected_city):
    # Filter data untuk kota yang dipilih
    city_costs = costProfCat[costProfCat['cities'] == selected_city]
    city_pois = poiList[poiList['cities'] == selected_city]
    
    # Buat mapping POI ID ke nama
    poi_names = dict(zip(city_pois['poiID'], city_pois['poiName']))
    
    # Buat graph
    G = nx.DiGraph()
    
    # Tambahkan edges dengan informasi cost
    for _, row in city_costs.iterrows():
        from_poi = poi_names.get(row['from'], f'POI {row["from"]}')
        to_poi = poi_names.get(row['to'], f'POI {row["to"]}')
        G.add_edge(from_poi, to_poi, weight=row['cost'])
    
    return G

# --------------------------------------------------------------------------
# Main Page (Streamlit)
# --------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="VDI - Tugas Besar",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            'Get Help': 'https://wa.me/6287860811076',
            'About': '''Copyright©️ ITERA 2024
            by Team 15 VDI
            Gymnastiar Al Khoarizmy
            Dea Mutia Risani
            📊Analisis Visualisasi Kunjungan Wisatawan ke Point-of-Interest di 8 Kota Pada Dataset Flickr'''
        }
    )

    t1, t2 = st.columns((0.07,1)) 

    t1.image('images/Logo_ITERA.png', width = 120)
    t2.title("Analisis Visualisasi Kunjungan Wisatawan ke Point-of-Interest di 8 Kota Pada Dataset Flickr")
    t2.markdown("by Team 15 VDI | Gymnastiar Al Khoarizmy (122450096) & Dea Mutia Risani (122450099)")

    with st.spinner('memuat data...'):
        # 1. Muat data
        costProfCat, poiList, userVisits = unzip_and_load()

        st.sidebar.header("POI Visualization Dashboard")
        menu = [
            "Dashboard POI Visualization",
            "About Us"
        ]
        choice = st.sidebar.selectbox("Pilih Menu", menu)

        if choice == "Dashboard POI Visualization":
            st.subheader("Informasi Dataset FLickr")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Jumlah Kota", value=poiList['cities'].nunique(), help="Jumlah kota dalam dataset, yaitu Edinburgh, Budapest, Delhi, Glasgow, Osaka, Perth, Toronto, dan Vienna")
            with col2:
                poi_counts = poiList.groupby('cities')['poiID'].count()
                st.metric("Jumlah POI", value=poi_counts.sum(), help="Jumlah Point of Interest (tempat kunjungan) dalam dataset")
            with col3:
                st.metric("Jumlah Kategori", value=costProfCat['category'].nunique(), help="Jumlah kategori POI dalam dataset")
            with col4:
                # Calculate total visits per POI per city
                poi_visits = userVisits.groupby(['cities', 'poiID'])['seqID'].count().reset_index(name='total_visits')
                # Show total visits across all POIs and cities
                total_visits = poi_visits['total_visits'].sum()
                st.metric("Jumlah Kunjungan", value=total_visits, help="Jumlah kunjungan user ke POI dalam dataset")
            with col5:
                st.metric("Jumlah User", value=userVisits['userID'].nunique(), help="Jumlah user yang melakukan kunjungan ke POI")

            #first row
            st.title("Visualisasi Data Point of Interest (POI)")
            a, b, c, d, e  = st.columns([0.42,0.01,0.12,0.01,0.42])
            with a:
                st.subheader("Top 10 POI Terpopuler (Plotly)")
                fig = top_10_popular(userVisits, poiList)
                fig.update_layout(xaxis_title='Nama POI', yaxis_title='Total Kunjungan', xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("ℹ️ Informasi Grafik", expanded=False):
                        st.markdown(
                            """
                            ####  Pertanyaan:
                            POI (Point of Interest) mana yang paling populer di setiap kota berdasarkan frekuensi kunjungan?

                            #### Visualisasi: 
                            Bar chart horizontal untuk setiap kota yang menunjukkan poiName berdasarkan poiFreq.
                            """)
            with b:
                st.write("")
            with c:
                st.subheader("Informasi Tambahan")
                avg_visits_per_poi = userVisits.groupby('poiID').size().mean()
                st.metric("Rata-rata Kunjungan per POI", value=f"{avg_visits_per_poi:.0f}", 
                            help="Rata-rata jumlah kunjungan untuk setiap POI")
                unique_sequences = userVisits['seqID'].nunique()
                st.metric("Jumlah Sequence Kunjungan", value=unique_sequences, 
                            help="Jumlah total urutan kunjungan yang berbeda")
                avg_visits_per_user = userVisits.groupby('userID').size().mean()
                st.metric("Rata-rata Kunjungan per User", value=f"{avg_visits_per_user:.1f}", 
                            help="Rata-rata jumlah kunjungan yang dilakukan oleh setiap user")
            with d:
                st.write("")
            with e:
                city_select = st.selectbox("Pilih Kota", ['Edin', 'Buda', 'Delh', 'Glas', 'Osak', 'Perth', 'Toro', 'Vien'], help='Pilih kota dengan kode Edin (Edinburgh), Buda (Budapest), Delh (Delhi), Glas (Glasgow), Osak (Osaka), Perth (Perth), Toro (Toronto), Vien (Vienna)')
                if city_select:
                    userVisits_poi = pd.merge(userVisits, poiList, on=['poiID','cities'], how='left')
                    # Ambil poiFreq
                    poi_freq = userVisits_poi[['poiID','cities','poiName','poiFreq']].drop_duplicates()
                    city_data = poi_freq[poi_freq['cities'] == city_select].sort_values('poiFreq', ascending=False).head(10)
                    fig = px.bar(city_data, x='poiFreq', y='poiName', color='poiName', title=f'Top 10 POI Terpopuler di Kota {city_select}')
                    fig.update_layout(xaxis_title='Frekuensi Kunjungan', yaxis_title='Nama POI', xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("ℹ️ Informasi Grafik", expanded=False):
                        st.markdown(
                            """
                            ####  Pertanyaan:
                            POI (Point of Interest) mana yang paling banyak dikunjungi di kota tertentu berdasarkan frekuensi kunjungan?

                            #### Visualisasi: 
                            Bar chart horizontal yang menunjukkan poiName berdasarkan poiFreq di kota tertentu.
                            """)
            st.write("")
            st.progress(100, text="")
            st.write("")

            #second row
            col0, col1, col2 = st.columns((0.15,1,0.02))
            with col0:
                st.write("")
            with col1:
                st.subheader("Distribusi Tema POI (Stacked Bar)")
                fig = stacked_theme(userVisits, poiList)
                st.plotly_chart(fig)
                with st.expander("ℹ️ Informasi Grafik", expanded=False):
                    st.markdown(
                        """
                        ####  Pertanyaan:
                        Bagaimana distribusi tema POI di setiap kota?

                        #### Visualisasi: 
                        Stacked bar chart yang menunjukkan distribusi tema POI di setiap kota.
                        """)
            with col2:
                st.write("")

            st.write("")
            st.progress(100, text="")
            st.write("")

            #third row
            c1, c2, c3 = st.columns((0.02,1,0.02))
            with c1:
                st.write("")
            with c2:
                st.subheader("Heatmap Biaya Rata-rata per Kategori")
                col1, col2 = st.columns([0.6, 0.4])
                
                with col1:
                    fig = heatmap_cost(costProfCat)
                    st.plotly_chart(fig)
                    with st.expander("ℹ️ Informasi Grafik", expanded=False):
                        st.markdown(
                            """
                            ####  Pertanyaan:
                            Bagaimana distribusi biaya rata-rata per kategori POI di setiap kota?

                            #### Visualisasi: 
                            Heatmap yang menunjukkan biaya rata-rata per kategori POI di setiap kota.
                            """)
                
                with col2:
                    selected_city = st.selectbox(
                        "Pilih Kota untuk Melihat Detail Cost antar POI",
                        costProfCat['cities'].unique()
                    )
                    
                    cost_option = st.selectbox(
                        "Pilih Opsi Biaya",
                        ["Top 10 Biaya Tertinggi", "Top 10 Biaya Terendah"]
                    )
                    
                    if selected_city:
                        city_costs = costProfCat[costProfCat['cities'] == selected_city]
                        city_pois = poiList[poiList['cities'] == selected_city]
                        
                        # Tambahkan nama POI ke dataframe
                        from_names = city_pois.set_index('poiID')['poiName'].to_dict()
                        to_names = from_names.copy()
                        
                        city_costs['From POI'] = city_costs['from'].map(from_names)
                        city_costs['To POI'] = city_costs['to'].map(to_names)
                        
                        if cost_option == "Top 10 Biaya Tertinggi":
                            sorted_costs = city_costs.sort_values('cost', ascending=False).head(10)
                        else:
                            sorted_costs = city_costs.sort_values('cost', ascending=True).head(10)
                        
                        # Reset index untuk tabel yang ditampilkan
                        sorted_costs = sorted_costs.reset_index(drop=True)
                        sorted_costs.index += 1
                        
                        # Tampilkan dataframe dengan informasi yang lebih readable
                        st.dataframe(
                            sorted_costs[['From POI', 'To POI', 'cost', 'category']]
                            .style.format({'cost': '{:.2f}'})
                        )
                        
                        st.caption(f"Top 10 rute dengan {cost_option.lower()} di {selected_city}")
            with c3:
                st.write("")

            st.write("")
            st.progress(100, text="")
            st.write("")
            
            #fourth row
            d1, d2, d3, d4, d5 = st.columns((0.42,0.01,0.12,0.01,0.42))
            with d1:
                st.subheader("Analisis Kunjungan Weekday vs Weekend")

                # Mengubah kolom 'dateTaken' menjadi format datetime dengan error handling
                userVisits['datetime'] = pd.to_datetime(userVisits['dateTaken'], unit='s', errors='coerce')
                userVisits['is_weekend'] = userVisits['datetime'].dt.dayofweek.isin([5, 6])  # 5=Saturday, 6=Sunday
                userVisits['day_of_week'] = userVisits['datetime'].dt.day_name()

                # Filter out invalid dates
                userVisits = userVisits[userVisits['datetime'].notna()]

                # Visualisasi interaktif menggunakan Plotly
                visits_by_day = userVisits['day_of_week'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
                
                # Create color list - blue for weekdays, red for weekends
                colors = ['#1f77b4']*5 + ['#d62728']*2  # First 5 blue (weekdays), last 2 red (weekend)
                
                fig = px.bar(visits_by_day, 
                            x=visits_by_day.index, 
                            y=visits_by_day.values, 
                            title='Distribusi Kunjungan POI Berdasarkan Hari dalam Seminggu',
                            color=visits_by_day.index,
                            color_discrete_sequence=colors)
                            
                fig.update_layout(xaxis_title='Hari dalam Seminggu', 
                                yaxis_title='Jumlah Kunjungan', 
                                xaxis_tickangle=-45,
                                showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("ℹ️ Informasi Grafik", expanded=False):
                    st.markdown(
                        """
                        ####  Pertanyaan:
                        Bagaimana pola kunjungan wisatawan ke POI berdasarkan hari dalam seminggu (weekday vs weekend)?

                        #### Visualisasi: 
                        Bar chart yang menunjukkan distribusi jumlah kunjungan untuk setiap hari dalam seminggu, dengan warna berbeda untuk weekday (biru) dan weekend (merah).
                        """)
            with d2:
                st.write("")
            with d3:
                # Calculate weekend vs weekday stats
                total_visits = len(userVisits)
                weekend_visits = userVisits[userVisits['is_weekend']]['poiID'].count()
                weekday_visits = total_visits - weekend_visits

                # Display metrics
                weekday_percent = weekday_visits/total_visits*100
                weekend_percent = weekend_visits/total_visits*100
                
                st.metric("Total Kunjungan", f"{total_visits:,}", 
                         help="Jumlah total kunjungan ke POI")
                st.metric("Kunjungan Weekday", f"{weekday_visits:,}", 
                         delta=f"{weekday_percent:.1f}%",
                         delta_color="off",
                         help="Jumlah dan persentase kunjungan di hari kerja")
                st.metric("Kunjungan Weekend", f"{weekend_visits:,}",
                         delta=f"{weekend_percent:.1f}%", 
                         delta_color="off",
                         help="Jumlah dan persentase kunjungan di akhir pekan")

                # Calculate and display average visits
                visits_per_day = userVisits.groupby(['is_weekend', userVisits['datetime'].dt.date])['poiID'].count().reset_index()
                avg_visits = visits_per_day.groupby('is_weekend')['poiID'].mean()
                            
                st.metric("Rata-rata Kunjungan per Hari Weekday", f"{avg_visits[False]:.1f}",
                          help="Rata-rata jumlah kunjungan per hari kerja")
                st.metric("Rata-rata Kunjungan per Hari Weekend", f"{avg_visits[True]:.1f}", 
                          delta=f"+{(avg_visits[True]-avg_visits[False]):.1f}",
                          help="Rata-rata jumlah kunjungan per hari di akhir pekan") 
            with d4:
                st.write("")
            with d5:
                # Visualisasi tema POI berdasarkan weekend/weekday
                theme_weekend_analysis = pd.crosstab(userVisits['poiTheme'], userVisits['is_weekend'], normalize='index') * 100
                fig = px.bar(theme_weekend_analysis, x=theme_weekend_analysis.index, y=[False, True], title='Distribusi Kunjungan Berdasarkan Tema POI (Weekday vs Weekend)', labels={'value': 'Persentase Kunjungan', 'poiTheme': 'Tema POI'}, barmode='stack')
                fig.update_layout(xaxis_title='Tema POI', yaxis_title='Persentase Kunjungan', legend_title='Is Weekend')
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("ℹ️ Informasi Grafik", expanded=False):
                    st.markdown(
                        """
                        ####  Pertanyaan:
                        Bagaimana pola distribusi kunjungan ke berbagai tema POI antara weekday dan weekend?

                        #### Visualisasi: 
                        Stacked bar chart yang menunjukkan persentase kunjungan untuk setiap tema POI, dibedakan antara weekday dan weekend.
                        """)

        elif choice == "About Us":
            st.subheader("Info Data userVisits")
            st.write(userVisits.info(verbose=True))
            st.subheader("Info Data poiList") 
            st.write(poiList.info(verbose=True))
            st.subheader("Info Data costProfCat")
            st.write(costProfCat.info(verbose=True))

            st.subheader("Cek Missing Values & Duplicates")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("userVisits Duplicates:", userVisits.duplicated().sum())
            with col2:
                st.write("poiList Duplicates:", poiList.duplicated().sum())
            with col3:
                st.write("costProfCat Duplicates:", costProfCat.duplicated().sum())
            st.write("Missing userVisits:", userVisits.isnull().sum())
            st.write("Missing poiList:", poiList.isnull().sum())
            st.write("Missing costProfCat:", costProfCat.isnull().sum())

            st.write("Preview userVisits:")
            st.dataframe(userVisits.head())
            st.write("Preview poiList:")
            st.dataframe(poiList.head())
            st.write("Preview costProfCat:")
            st.dataframe(costProfCat.head())

        elif choice == "Top 10 POI (Plotly)":
            st.subheader("Plot: Top 10 POI Terpopuler (Plotly)")
            fig = top_10_popular(userVisits, poiList)
            st.plotly_chart(fig, use_container_width=True)

        elif choice == "Top 10 POI (Matplotlib)":
            st.subheader("Plot: Top 10 POI Terpopuler (Matplotlib)")
            fig = barplot_top_10(userVisits, poiList)
            st.pyplot(fig)

        elif choice == "Distribusi Tema (Stacked Bar)":
            st.subheader("Stacked Bar Tema POI")
            fig = stacked_theme(userVisits, poiList)
            st.pyplot(fig)

        elif choice == "Heatmap Biaya":
            st.subheader("Heatmap Biaya Rata-rata per Kategori")
            fig = heatmap_cost(costProfCat)
            st.pyplot(fig)

        elif choice == "Scatter Cost-Profit":
            st.subheader("Scatter Plot Cost vs Profit")
            fig = scatter_cost_profit(costProfCat)
            st.pyplot(fig)

        elif choice == "Tren Kunjungan Waktu":
            st.subheader("Line Chart Tren Kunjungan")
            fig = line_trend_visits(userVisits)
            st.pyplot(fig)

        elif choice == "Bar Profit Kategori":
            st.subheader("Bar Chart Profit Rata-rata per Kategori")
            fig = bar_profit_category(costProfCat)
            st.pyplot(fig)

        elif choice == "Map Suatu Kota":
            st.subheader("Map POI di Suatu Kota")
            city_list = poiList['cities'].unique()
            select_city = st.selectbox("Pilih Kota", city_list)
            fig = map_city(poiList, city=select_city)
            st.plotly_chart(fig, use_container_width=True)

        elif choice == "Pie Tema di Kota":
            st.subheader("Pie Chart Tema di Suatu Kota")
            city_list = userVisits['cities'].unique()
            select_city = st.selectbox("Pilih Kota", city_list)
            fig = pie_theme_city(userVisits, city=select_city)
            st.pyplot(fig)

if __name__ == "__main__":
    main()