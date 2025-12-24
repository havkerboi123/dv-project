import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Global YouTube Analytics", layout="wide", page_icon="▶️")

# Data Cleaning
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Global YouTube Statistics.csv', encoding='latin-1')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

    df['Country'] = df['Country'].fillna('Unknown')
    df['category'] = df['category'].fillna('Misc')
    
    df = df.dropna(subset=['created_year'])
    df['created_year'] = df['created_year'].astype(int)
    
    # Create 'Earnings' column
    df['Avg_Yearly_Earnings'] = (df['lowest_yearly_earnings'] + df['highest_yearly_earnings']) / 2
    
    return df

df = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Interactive Dashboard", "EDA Analysis"])

if df is not None:

    # Dashboard Page

    if page == "Interactive Dashboard":
        # Initialize session state for map selection (can be a list for multiple countries)
        if 'map_selected_countries' not in st.session_state:
            st.session_state.map_selected_countries = []
        
        # Initialize session state for scatter plot selection (list of selected youtubers)
        if 'scatter_selected_youtubers' not in st.session_state:
            st.session_state.scatter_selected_youtubers = []
        
        # Sidebar Filters
        st.sidebar.header("Dashboard Filters")
        
        min_year = int(df['created_year'].min())
        max_year = int(df['created_year'].max())
        year_range = st.sidebar.slider("Channel Creation Year", min_year, max_year, (2010, 2022))
        
        categories = ['All'] + sorted(list(df['category'].unique()))
        selected_category = st.sidebar.selectbox("Category", categories)
        
        countries = ['All'] + sorted(list(df['Country'].unique()))
        selected_country = st.sidebar.selectbox("Country", countries)
        
        # Clear selection buttons
        col_clear1, col_clear2 = st.sidebar.columns(2)
        with col_clear1:
            if st.button("🔄 Clear Map"):
                st.session_state.map_selected_countries = []
                st.rerun()
        with col_clear2:
            if st.button("🔄 Clear Scatter"):
                st.session_state.scatter_selected_youtubers = []
                st.rerun()

        # Apply base filters (year, category, country) - this is what scatter plot will show
        df_for_scatter = df[(df['created_year'] >= year_range[0]) & (df['created_year'] <= year_range[1])]
        if selected_category != 'All':
            df_for_scatter = df_for_scatter[df_for_scatter['category'] == selected_category]
        
        # Map selection takes priority over sidebar country filter
        if st.session_state.map_selected_countries:
            df_for_scatter = df_for_scatter[df_for_scatter['Country'].isin(st.session_state.map_selected_countries)]
        elif selected_country != 'All':
            df_for_scatter = df_for_scatter[df_for_scatter['Country'] == selected_country]
        
        # Apply scatter plot selection to get final filtered data for other visualizations
        df_filtered = df_for_scatter.copy()
        if st.session_state.scatter_selected_youtubers:
            df_filtered = df_filtered[df_filtered['Youtuber'].isin(st.session_state.scatter_selected_youtubers)]

        # Main Layout
        st.title("📊 Global YouTube Statistics Dashboard")
        st.markdown("Dynamic overview of top creators, geography, and growth trends.")
        
        # Show active filters
        active_filters = []
        if st.session_state.map_selected_countries:
            countries_str = ", ".join(st.session_state.map_selected_countries[:3])
            if len(st.session_state.map_selected_countries) > 3:
                countries_str += f" (+{len(st.session_state.map_selected_countries) - 3} more)"
            active_filters.append(f"🗺️ **Map:** {countries_str} ({len(st.session_state.map_selected_countries)} countries)")
        
        if st.session_state.scatter_selected_youtubers:
            youtubers_str = ", ".join(st.session_state.scatter_selected_youtubers[:3])
            if len(st.session_state.scatter_selected_youtubers) > 3:
                youtubers_str += f" (+{len(st.session_state.scatter_selected_youtubers) - 3} more)"
            active_filters.append(f"📊 **Scatter Plot:** {youtubers_str} ({len(st.session_state.scatter_selected_youtubers)} channels)")
        
        if active_filters:
            st.info(" | ".join(active_filters))

        # KPI - handle empty dataframe
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        if len(df_filtered) > 0:
            kpi1.metric("Total Channels", f"{len(df_filtered):,}")
            kpi2.metric("Total Subscribers", f"{df_filtered['subscribers'].sum()/1e9:.2f}B")
            kpi3.metric("Avg Views", f"{df_filtered['video views'].mean()/1e6:.1f}M")
            kpi4.metric("Avg Earnings (Est.)", f"${df_filtered['Avg_Yearly_Earnings'].mean()/1e6:.2f}M")
        else:
            kpi1.metric("Total Channels", "0")
            kpi2.metric("Total Subscribers", "0B")
            kpi3.metric("Avg Views", "0M")
            kpi4.metric("Avg Earnings (Est.)", "$0M")

        st.markdown("---")

        # Row 1: Map and Time Series
        c1, c2 = st.columns([3, 2])
        
        with c1:
            st.subheader("🌍 Creator Geography")
            st.markdown("*💡 Click on a point to filter all visualizations by that country*")
            
            # Prepare map data (use full dataset for map, but filter for display)
            country_stats_full = df.groupby(['Country', 'Latitude', 'Longitude']).size().reset_index(name='Count')
            country_stats_full = country_stats_full.dropna(subset=['Latitude', 'Longitude']).reset_index(drop=True)
            
            # Create map with selection enabled - add index to customdata for easier lookup
            # Store country names in a way that's easily accessible
            # custom_data needs to be a list of lists for plotly express
            country_stats_full['Country_Data'] = country_stats_full['Country']
            fig_map = px.scatter_geo(country_stats_full, lat='Latitude', lon='Longitude', size='Count',
                                     hover_name='Country', 
                                     hover_data={'Country': True, 'Count': True},
                                     projection="natural earth",
                                     title="Concentration of YouTube Channels (Click or Lasso to Filter)", 
                                     template="plotly_dark",
                                     custom_data=['Country_Data'])
            
            fig_map.update_geos(showcountries=True, countrycolor="Black")
            fig_map.update_layout(clickmode='event+select')
            
            # Handle map selection
            selected_map = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun", key="map_chart")
            
            # Process selection event - handle both single clicks and lasso selections
            if selected_map:
                try:
                    # Debug: Check the structure (can be removed later)
                    # st.write("Selection data:", selected_map)
                    
                    # Extract points from selection - try multiple possible structures
                    points = []
                    
                    # Try different possible structures
                    if isinstance(selected_map, dict):
                        # Structure 1: {'selection': {'points': [...]}}
                        if 'selection' in selected_map:
                            sel = selected_map['selection']
                            if isinstance(sel, dict) and 'points' in sel:
                                points = sel['points']
                            elif isinstance(sel, list):
                                points = sel
                        # Structure 2: {'points': [...]}
                        elif 'points' in selected_map:
                            points = selected_map['points']
                        # Structure 3: Direct list of points
                        elif isinstance(selected_map, list):
                            points = selected_map
                    elif isinstance(selected_map, list):
                        points = selected_map
                    
                    if points and len(points) > 0:
                        selected_countries = []
                        
                        # Extract all countries from selected points
                        for point in points:
                            if isinstance(point, dict):
                                country_name = None
                                
                                # Method 1: Try customdata (most reliable)
                                if 'customdata' in point:
                                    customdata = point['customdata']
                                    if customdata and len(customdata) > 0:
                                        country_name = customdata[0]
                                
                                # Method 2: Try hovertext
                                if not country_name and 'hovertext' in point:
                                    country_name = point['hovertext']
                                
                                # Method 3: Try text
                                if not country_name and 'text' in point:
                                    country_name = point['text']
                                
                                # Method 4: Use point index as fallback
                                if not country_name:
                                    idx = None
                                    if 'pointIndex' in point:
                                        idx = point['pointIndex']
                                    elif 'pointNumber' in point:
                                        idx = point['pointNumber']
                                    elif 'point_index' in point:
                                        idx = point['point_index']
                                    
                                    if idx is not None and 0 <= idx < len(country_stats_full):
                                        country_name = country_stats_full.iloc[idx]['Country']
                                
                                # Add country if found and not already in list
                                if country_name and country_name not in selected_countries:
                                    selected_countries.append(country_name)
                        
                        # Update session state if countries changed
                        if selected_countries:
                            # Sort to maintain consistent order
                            selected_countries = sorted(selected_countries)
                            if selected_countries != st.session_state.map_selected_countries:
                                st.session_state.map_selected_countries = selected_countries
                                st.rerun()
                        # If no countries found but points exist, clear selection
                        elif len(points) > 0:
                            # Selection was made but couldn't extract countries - clear it
                            if st.session_state.map_selected_countries:
                                st.session_state.map_selected_countries = []
                                st.rerun()
                except Exception as e:
                    # Silently handle errors - selection might not be in expected format
                    # Uncomment below for debugging if needed:
                    # st.error(f"Error processing selection: {str(e)}")
                    # st.write("Selection data structure:", selected_map)
                    pass

        with c2:
            st.subheader("📈 Creation Trends")
            if len(df_filtered) > 0:
                year_counts = df_filtered.groupby('created_year').size().reset_index(name='Count')
                fig_line = px.area(year_counts, x='created_year', y='Count', markers=True,
                                   title="Channels Created Over Time", template="plotly_dark")
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("No data available for selected filters.")

        # Row 2: Categories and Scatter
        c3, c4 = st.columns(2)
        
        with c3:
            st.subheader("🏆 Top Categories")
            if len(df_filtered) > 0:
                cat_counts = df_filtered['category'].value_counts().nlargest(10).reset_index()
                cat_counts.columns = ['Category', 'Count']
                if len(cat_counts) > 0:
                    fig_bar = px.bar(cat_counts, x='Count', y='Category', orientation='h', color='Count',
                                     title="Top 10 Categories", template="plotly_dark")
                    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("No category data available.")
            else:
                st.info("No data available for selected filters.")
            
        with c4:
            st.subheader("💰 Subs vs Earnings")
            st.markdown("*💡 Click or lasso select points to filter all visualizations*")
            if len(df_for_scatter) > 0:
                # Filter out rows with invalid data - use df_for_scatter so scatter shows all available data
                scatter_data = df_for_scatter[(df_for_scatter['subscribers'] > 0) & (df_for_scatter['Avg_Yearly_Earnings'] > 0)].copy()
                if len(scatter_data) > 0:
                    # Add Youtuber name to custom_data for selection
                    scatter_data['Youtuber_Data'] = scatter_data['Youtuber']
                    fig_scat = px.scatter(scatter_data, x='subscribers', y='Avg_Yearly_Earnings', 
                                          color='category', size='video views', hover_name='Youtuber',
                                          log_x=True, log_y=True, title="Subscribers vs Yearly Earnings (Click or Lasso to Filter)",
                                          template="plotly_dark",
                                          custom_data=['Youtuber_Data'])
                    fig_scat.update_layout(clickmode='event+select')
                    
                    # Handle scatter plot selection
                    selected_scatter = st.plotly_chart(fig_scat, use_container_width=True, on_select="rerun", key="scatter_chart")
                    
                    # Process scatter plot selection event
                    if selected_scatter:
                        try:
                            # Extract points from selection
                            points = []
                            if isinstance(selected_scatter, dict):
                                if 'selection' in selected_scatter:
                                    sel = selected_scatter['selection']
                                    if isinstance(sel, dict) and 'points' in sel:
                                        points = sel['points']
                                    elif isinstance(sel, list):
                                        points = sel
                                elif 'points' in selected_scatter:
                                    points = selected_scatter['points']
                            elif isinstance(selected_scatter, list):
                                points = selected_scatter
                            
                            if points and len(points) > 0:
                                selected_youtubers = []
                                
                                # Extract all youtubers from selected points
                                for point in points:
                                    if isinstance(point, dict):
                                        youtuber_name = None
                                        
                                        # Method 1: Try customdata
                                        if 'customdata' in point:
                                            customdata = point['customdata']
                                            if customdata and len(customdata) > 0:
                                                youtuber_name = customdata[0]
                                        
                                        # Method 2: Try hovertext
                                        if not youtuber_name and 'hovertext' in point:
                                            youtuber_name = point['hovertext']
                                        
                                        # Method 3: Try text
                                        if not youtuber_name and 'text' in point:
                                            youtuber_name = point['text']
                                        
                                        # Method 4: Use point index as fallback
                                        if not youtuber_name:
                                            idx = None
                                            if 'pointIndex' in point:
                                                idx = point['pointIndex']
                                            elif 'pointNumber' in point:
                                                idx = point['pointNumber']
                                            elif 'point_index' in point:
                                                idx = point['point_index']
                                            
                                            if idx is not None and 0 <= idx < len(scatter_data):
                                                youtuber_name = scatter_data.iloc[idx]['Youtuber']
                                        
                                        # Add youtuber if found and not already in list
                                        if youtuber_name and youtuber_name not in selected_youtubers:
                                            selected_youtubers.append(youtuber_name)
                                
                                # Update session state if youtubers changed
                                if selected_youtubers:
                                    # Sort to maintain consistent order
                                    selected_youtubers = sorted(selected_youtubers)
                                    if selected_youtubers != st.session_state.scatter_selected_youtubers:
                                        st.session_state.scatter_selected_youtubers = selected_youtubers
                                        st.rerun()
                                # If no youtubers found but points exist, clear selection
                                elif len(points) > 0:
                                    if st.session_state.scatter_selected_youtubers:
                                        st.session_state.scatter_selected_youtubers = []
                                        st.rerun()
                        except Exception as e:
                            # Silently handle errors
                            pass
                else:
                    st.info("No valid data for scatter plot.")
            else:
                st.info("No data available for selected filters.")

    # EDA Page
    elif page == "EDA Analysis":
        st.title("🔎 Exploratory Data Analysis")
        st.markdown("Deep dive into the statistical properties and distributions of the dataset.")

        # Overview
        with st.expander("📄 Dataset Overview & Summary Statistics", expanded=True):
            st.write("First 5 rows of the dataset:")
            st.dataframe(df.head())
            st.write("Statistical Summary:")
            st.dataframe(df.describe())

        # Univariate Analysis
        st.subheader("1. Distribution Analysis")
        col_u1, col_u2 = st.columns(2)
        
        with col_u1:
            st.markdown("**Distribution of Subscribers (Log Scale)**")
            fig_hist = px.histogram(df, x='subscribers', nbins=50, log_y=True, 
                                    title="Histogram of Subscribers", template="plotly_white",
                                    color_discrete_sequence=['teal'])
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col_u2:
            st.markdown("**Distribution of Video Views (Log Scale)**")
            fig_hist_v = px.histogram(df, x='video views', nbins=50, log_y=True, 
                                      title="Histogram of Video Views", template="plotly_white",
                                      color_discrete_sequence=['purple'])
            st.plotly_chart(fig_hist_v, use_container_width=True)

        # Correlation Analysis
        st.subheader("2. Correlation Heatmap")
        st.markdown("Analyzing the relationship between numerical variables.")
        
        numeric_df = df[['subscribers', 'video views', 'uploads', 'video_views_for_the_last_30_days', 
                         'lowest_yearly_earnings', 'highest_yearly_earnings', 'Avg_Yearly_Earnings']]
        corr = numeric_df.corr()
        
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r',
                             title="Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

        # Categorical
        st.subheader("3. Earnings by Category")
        st.markdown("Which categories have the highest median earnings?")
        
        top_cats = df['category'].value_counts().nlargest(10).index
        df_top_cat = df[df['category'].isin(top_cats)]
        
        fig_box = px.box(df_top_cat, x='category', y='Avg_Yearly_Earnings', 
                         log_y=True, color='category',
                         title="Yearly Earnings Distribution by Top Categories (Log Scale)",
                         template="plotly_white")
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Missing Data
        st.subheader("4. Missing Data Check")
        missing_data = df.isnull().sum().reset_index()
        missing_data.columns = ['Column', 'Missing Values']
        missing_data = missing_data[missing_data['Missing Values'] > 0]
        
        if not missing_data.empty:
            fig_missing = px.bar(missing_data, x='Column', y='Missing Values', 
                                 title="Count of Missing Values per Column", color='Missing Values')
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("No significant missing data in key columns used for analysis.")