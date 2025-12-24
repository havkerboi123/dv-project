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

        st.markdown("---")
        
        # Row 3: Uploads and Channel Type
        c5, c6 = st.columns(2)
        
        with c5:
            st.subheader("📤 Top Channels by Uploads")
            if len(df_filtered) > 0:
                uploads_data = df_filtered[df_filtered['uploads'] > 0].nlargest(10, 'uploads')[['Youtuber', 'uploads', 'subscribers']]
                if len(uploads_data) > 0:
                    fig_uploads = px.bar(uploads_data, x='uploads', y='Youtuber', orientation='h',
                                         color='subscribers', color_continuous_scale='viridis',
                                         title="Top 10 Channels by Total Uploads", template="plotly_dark",
                                         labels={'uploads': 'Total Uploads', 'subscribers': 'Subscribers'})
                    fig_uploads.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_uploads, use_container_width=True)
                else:
                    st.info("No upload data available.")
            else:
                st.info("No data available for selected filters.")
        
        with c6:
            st.subheader("📺 Channel Type Distribution")
            if len(df_filtered) > 0:
                channel_type_data = df_filtered['channel_type'].value_counts().head(10).reset_index()
                channel_type_data.columns = ['Channel Type', 'Count']
                if len(channel_type_data) > 0:
                    fig_channel_type = px.pie(channel_type_data, values='Count', names='Channel Type',
                                             title="Distribution by Channel Type", template="plotly_dark",
                                             hole=0.4)
                    st.plotly_chart(fig_channel_type, use_container_width=True)
                else:
                    st.info("No channel type data available.")
            else:
                st.info("No data available for selected filters.")

        # Row 4: Recent Engagement and Earnings Range
        c7, c8 = st.columns(2)
        
        with c7:
            st.subheader("🔥 Recent Engagement (Last 30 Days)")
            if len(df_filtered) > 0:
                # Filter valid data
                recent_data = df_filtered[
                    (df_filtered['video_views_for_the_last_30_days'].notna()) & 
                    (df_filtered['video_views_for_the_last_30_days'] > 0)
                ].nlargest(10, 'video_views_for_the_last_30_days')
                
                if len(recent_data) > 0:
                    fig_recent = px.bar(recent_data, 
                                       x='video_views_for_the_last_30_days', 
                                       y='Youtuber',
                                       orientation='h',
                                       color='subscribers_for_last_30_days',
                                       color_continuous_scale='reds',
                                       title="Top 10 by Views (Last 30 Days)", 
                                       template="plotly_dark",
                                       labels={'video_views_for_the_last_30_days': 'Views (30 Days)', 
                                              'subscribers_for_last_30_days': 'New Subs (30 Days)'})
                    fig_recent.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_recent, use_container_width=True)
                else:
                    st.info("No recent engagement data available.")
            else:
                st.info("No data available for selected filters.")
        
        with c8:
            st.subheader("💵 Earnings Range Analysis")
            if len(df_filtered) > 0:
                earnings_data = df_filtered[
                    (df_filtered['lowest_yearly_earnings'] > 0) & 
                    (df_filtered['highest_yearly_earnings'] > 0)
                ].nlargest(15, 'highest_yearly_earnings')
                
                if len(earnings_data) > 0:
                    # Create a range visualization
                    fig_earnings = go.Figure()
                    
                    fig_earnings.add_trace(go.Scatter(
                        x=earnings_data['Youtuber'],
                        y=earnings_data['lowest_yearly_earnings'],
                        mode='markers',
                        name='Lowest Earnings',
                        marker=dict(color='lightblue', size=8)
                    ))
                    
                    fig_earnings.add_trace(go.Scatter(
                        x=earnings_data['Youtuber'],
                        y=earnings_data['highest_yearly_earnings'],
                        mode='markers',
                        name='Highest Earnings',
                        marker=dict(color='orange', size=8)
                    ))
                    
                    # Add lines to show range
                    for idx, row in earnings_data.iterrows():
                        fig_earnings.add_trace(go.Scatter(
                            x=[row['Youtuber'], row['Youtuber']],
                            y=[row['lowest_yearly_earnings'], row['highest_yearly_earnings']],
                            mode='lines',
                            line=dict(color='gray', width=1, dash='dash'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    
                    fig_earnings.update_layout(
                        title="Earnings Range: Lowest vs Highest (Top 15)",
                        xaxis_title="YouTuber",
                        yaxis_title="Yearly Earnings ($)",
                        template="plotly_dark",
                        xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
                        yaxis=dict(type='log'),
                        height=400,
                        hovermode='closest'
                    )
                    st.plotly_chart(fig_earnings, use_container_width=True)
                else:
                    st.info("No earnings data available.")
            else:
                st.info("No data available for selected filters.")

        # Row 5: Monthly Creation Patterns
        st.markdown("---")
        c9 = st.columns(1)[0]
        
        with c9:
            st.subheader("📅 Channel Creation by Month")
            if len(df_filtered) > 0:
                # Count channels by month (filter out NaN)
                monthly_counts = df_filtered['created_month'].dropna().value_counts().reset_index()
                monthly_counts.columns = ['Month', 'Count']
                
                if len(monthly_counts) > 0:
                    # Order months properly
                    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    # Only keep months that exist in data
                    available_months = [m for m in month_order if m in monthly_counts['Month'].values]
                    monthly_counts['Month'] = pd.Categorical(monthly_counts['Month'], 
                                                           categories=available_months, 
                                                           ordered=True)
                    monthly_counts = monthly_counts.sort_values('Month')
                    
                    fig_monthly = px.bar(monthly_counts, x='Month', y='Count',
                                        title="Channels Created by Month",
                                        color='Count',
                                        color_continuous_scale='blues',
                                  template="plotly_dark")
                    st.plotly_chart(fig_monthly, use_container_width=True)
                else:
                    st.info("No monthly creation data available.")
            else:
                st.info("No data available for selected filters.")

        # Row 6: Ranking Analysis
        st.markdown("---")
        c10 = st.columns(1)[0]
        
        with c10:
            st.subheader("🏅 Global Rank vs Country Rank")
            if len(df_filtered) > 0:
                ranking_data = df_filtered[
                    (df_filtered['rank'].notna()) & 
                    (df_filtered['country_rank'].notna()) &
                    (df_filtered['rank'] <= 100)  # Top 100 for readability
                ].nlargest(20, 'subscribers')
                
                if len(ranking_data) > 0:
                    fig_rank = px.scatter(ranking_data, 
                                         x='rank', 
                                         y='country_rank',
                                         size='subscribers',
                                         color='Country',
                                         hover_name='Youtuber',
                                         title="Global Rank vs Country Rank (Top 20 by Subscribers)",
                                         template="plotly_dark",
                                         labels={'rank': 'Global Rank', 'country_rank': 'Country Rank'})
                    fig_rank.update_xaxes(autorange="reversed")
                    fig_rank.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig_rank, use_container_width=True)
                else:
                    st.info("No ranking data available.")
            else:
                st.info("No data available for selected filters.")

        # Row 7: Country Demographics Analysis
        st.markdown("---")
        c12 = st.columns(1)[0]
        
        with c12:
            st.subheader("🌍 Country Demographics vs YouTube Success")
            if len(df_filtered) > 0:
                # Aggregate by country
                country_analysis = df_filtered.groupby('Country').agg({
                    'subscribers': 'sum',
                    'Population': 'first',
                    'Unemployment rate': 'first',
                    'Urban_population': 'first',
                    'Gross tertiary education enrollment (%)': 'first'
                }).reset_index()
                
                # Filter valid data
                country_analysis = country_analysis[
                    (country_analysis['Population'].notna()) & 
                    (country_analysis['subscribers'] > 0) &
                    (country_analysis['Population'] > 0)
                ]
                
                if len(country_analysis) > 0:
                    # Create bubble chart: Population vs Subscribers
                    fig_demo = px.scatter(country_analysis,
                                         x='Population',
                                         y='subscribers',
                                         size='Urban_population',
                                         color='Unemployment rate',
                                         hover_name='Country',
                                         hover_data={'Gross tertiary education enrollment (%)': True},
                                         title="Country Demographics: Population vs Total Subscribers",
                                         template="plotly_dark",
                                         labels={'Population': 'Country Population',
                                                'subscribers': 'Total Subscribers',
                                                'Urban_population': 'Urban Population',
                                                'Unemployment rate': 'Unemployment Rate (%)'},
                                         color_continuous_scale='RdYlGn_r',
                                         log_x=True,
                                         log_y=True)
                    st.plotly_chart(fig_demo, use_container_width=True)
                else:
                    st.info("No demographic data available.")
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
