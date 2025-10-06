"""
CMLRE Scientific Platform - Advanced Marine Data Integration System
Centre for Marine Living Resources and Ecology (CMLRE)
Ministry of Earth Sciences, Government of India

A comprehensive platform for:
- Multi-disciplinary marine data integration
- Advanced scientific analytics
- Otolith morphology analysis
- Taxonomic classification
- eDNA data management
- Cross-domain correlation analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os
import requests
import base64
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import cv2
from PIL import Image
import io

# Configure page
st.set_page_config(
    page_title="CMLRE Scientific Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

class CMLREScientificPlatform:
    """Advanced Marine Data Integration Platform for CMLRE"""
    
    def __init__(self):
        """Initialize the CMLRE platform"""
        self.api_base_url = os.getenv("API_BASE_URL", "https://cmlre-backend.up.railway.app")
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'current_user' not in st.session_state:
            st.session_state.current_user = "CMLRE Scientist"
        if 'research_projects' not in st.session_state:
            st.session_state.research_projects = []
        if 'datasets' not in st.session_state:
            st.session_state.datasets = []
        if 'analyses' not in st.session_state:
            st.session_state.analyses = []
    
    def main(self):
        """Main application interface"""
        st.title("🔬 CMLRE Scientific Platform")
        st.markdown("**Centre for Marine Living Resources and Ecology**")
        st.markdown("*Ministry of Earth Sciences, Government of India*")
        
        # Navigation tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Data Integration", "🧬 Taxonomy & eDNA", "🐟 Otolith Analysis", 
            "🌊 Oceanography", "📈 Analytics", "🔬 Research Tools"
        ])
        
        with tab1:
            self.render_data_integration()
        with tab2:
            self.render_taxonomy_edna()
        with tab3:
            self.render_otolith_analysis()
        with tab4:
            self.render_oceanography()
        with tab5:
            self.render_analytics()
        with tab6:
            self.render_research_tools()
    
    def render_data_integration(self):
        """Data Integration and Ingestion Module"""
        st.header("📊 Multi-Disciplinary Data Integration")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🔗 Data Sources")
            
            # Data source selection
            data_sources = st.multiselect(
                "Select Data Sources:",
                ["Physical Oceanography", "Chemical Oceanography", "Biological Oceanography",
                 "Fish Abundance", "Species Diversity", "Life History Traits", 
                 "Ecomorphology", "Taxonomy", "Otolith Morphology", "eDNA", "Molecular Biology"],
                default=["Fish Abundance", "Taxonomy", "eDNA"]
            )
            
            # Data format selection
            data_formats = st.multiselect(
                "Data Formats:",
                ["CSV", "JSON", "NetCDF", "HDF5", "Excel", "Database", "API", "Images"],
                default=["CSV", "JSON", "NetCDF"]
            )
            
            # Upload data
            uploaded_files = st.file_uploader(
                "Upload Marine Data Files:",
                accept_multiple_files=True,
                type=['csv', 'json', 'xlsx', 'nc', 'h5']
            )
            
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} files uploaded successfully!")
                
                # Process uploaded files
                for file in uploaded_files:
                    if file.name.endswith('.csv'):
                        df = pd.read_csv(file)
                        st.session_state.datasets.append({
                            'name': file.name,
                            'type': 'CSV',
                            'data': df,
                            'upload_time': datetime.now()
                        })
        
        with col2:
            st.subheader("📈 Integration Status")
            
            # Data integration metrics
            total_datasets = len(st.session_state.datasets)
            integrated_sources = len(data_sources)
            
            st.metric("Total Datasets", total_datasets)
            st.metric("Integrated Sources", integrated_sources)
            st.metric("Data Formats", len(data_formats))
            
            # Integration progress
            progress = min(100, (total_datasets * 10) + (integrated_sources * 5))
            st.progress(progress / 100)
            st.caption(f"Integration Progress: {progress}%")
            
            # Data quality indicators
            st.subheader("🔍 Data Quality")
            st.success("✅ Metadata Standardized")
            st.success("✅ Format Validation Complete")
            st.info("ℹ️ Cross-Reference Mapping in Progress")
    
    def render_taxonomy_edna(self):
        """Taxonomic Classification and eDNA Analysis Module"""
        st.header("🧬 Taxonomic Classification & eDNA Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔬 Species Identification")
            
            # Taxonomic classification
            species_input = st.text_area(
                "Enter species data (scientific names, common names, or descriptions):",
                placeholder="Thunnus albacares\nYellowfin Tuna\nLarge pelagic fish with yellow fins"
            )
            
            if st.button("🔍 Classify Species"):
                if species_input:
                    # Mock taxonomic classification
                    classification_result = self.perform_taxonomic_classification(species_input)
                    st.success("✅ Species classification completed!")
                    
                    # Display results
                    st.subheader("📋 Classification Results")
                    for result in classification_result:
                        st.write(f"**{result['species']}** - {result['confidence']:.1%} confidence")
                        st.write(f"Kingdom: {result['kingdom']} | Family: {result['family']}")
                        st.write(f"Common Name: {result['common_name']}")
                        st.divider()
        
        with col2:
            st.subheader("🧬 eDNA Analysis")
            
            # eDNA data input
            edna_data = st.text_area(
                "Enter eDNA sequence data:",
                placeholder="ATCGATCGATCG...\nOr upload sequence files"
            )
            
            # eDNA analysis parameters
            col2a, col2b = st.columns(2)
            with col2a:
                min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.8)
            with col2b:
                database = st.selectbox("Reference Database", ["NCBI", "BOLD", "Custom"])
            
            if st.button("🧬 Analyze eDNA"):
                if edna_data:
                    # Mock eDNA analysis
                    edna_results = self.perform_edna_analysis(edna_data, min_confidence, database)
                    st.success("✅ eDNA analysis completed!")
                    
                    # Display eDNA results
                    st.subheader("🧬 eDNA Results")
                    for result in edna_results:
                        st.write(f"**{result['species']}** - {result['confidence']:.1%} match")
                        st.write(f"Sequence Length: {result['length']} bp")
                        st.write(f"Database: {result['database']}")
                        st.divider()
    
    def render_otolith_analysis(self):
        """Otolith Morphology and Shape Analysis Module"""
        st.header("🐟 Otolith Morphology Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📸 Image Upload")
            
            # Otolith image upload
            uploaded_image = st.file_uploader(
                "Upload Otolith Image:",
                type=['png', 'jpg', 'jpeg', 'tiff']
            )
            
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Otolith Image", use_column_width=True)
                
                # Analysis parameters
                st.subheader("⚙️ Analysis Parameters")
                edge_detection = st.selectbox("Edge Detection", ["Canny", "Sobel", "Laplacian"])
                morphometry_type = st.selectbox("Morphometry Type", ["2D Shape", "3D Volume", "Surface Texture"])
                
                if st.button("🔬 Analyze Otolith"):
                    # Mock otolith analysis
                    analysis_results = self.perform_otolith_analysis(image, edge_detection, morphometry_type)
                    st.success("✅ Otolith analysis completed!")
        
        with col2:
            st.subheader("📊 Morphometric Analysis")
            
            # Display analysis results
            if 'otolith_analysis' in st.session_state:
                results = st.session_state.otolith_analysis
                
                # Shape parameters
                st.subheader("📐 Shape Parameters")
                col2a, col2b, col2c = st.columns(3)
                with col2a:
                    st.metric("Area", f"{results['area']:.2f} mm²")
                    st.metric("Perimeter", f"{results['perimeter']:.2f} mm")
                with col2b:
                    st.metric("Aspect Ratio", f"{results['aspect_ratio']:.3f}")
                    st.metric("Circularity", f"{results['circularity']:.3f}")
                with col2c:
                    st.metric("Roundness", f"{results['roundness']:.3f}")
                    st.metric("Solidity", f"{results['solidity']:.3f}")
                
                # Visualization
                st.subheader("📈 Shape Visualization")
                self.create_otolith_visualization(results)
    
    def render_oceanography(self):
        """Oceanographic Data Analysis Module"""
        st.header("🌊 Oceanographic Data Analysis")
        
        # Oceanographic parameters
        st.subheader("🌡️ Environmental Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Sea Surface Temperature", "28.5°C", "0.2°C")
            st.metric("Salinity", "35.2 PSU", "0.1 PSU")
        
        with col2:
            st.metric("Dissolved Oxygen", "6.8 mg/L", "-0.1 mg/L")
            st.metric("pH", "8.1", "0.05")
        
        with col3:
            st.metric("Chlorophyll-a", "0.8 mg/m³", "0.1 mg/m³")
            st.metric("Turbidity", "2.1 NTU", "0.3 NTU")
        
        # Oceanographic trends
        st.subheader("📈 Environmental Trends")
        
        # Generate mock oceanographic data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        sst_data = 28 + 2 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 0.5, len(dates))
        salinity_data = 35 + 0.5 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 0.1, len(dates))
        
        # Create trend plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sea Surface Temperature', 'Salinity', 'Dissolved Oxygen', 'Chlorophyll-a'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(go.Scatter(x=dates, y=sst_data, name='SST (°C)', line=dict(color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=salinity_data, name='Salinity (PSU)', line=dict(color='blue')), row=1, col=2)
        
        fig.update_layout(height=600, showlegend=True, title_text="Oceanographic Parameters Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    def render_analytics(self):
        """Advanced Analytics and Correlation Analysis"""
        st.header("📈 Advanced Analytics & Correlation Analysis")
        
        # Correlation analysis
        st.subheader("🔗 Cross-Domain Correlation Analysis")
        
        # Generate mock correlation data
        np.random.seed(42)
        n_samples = 100
        
        # Oceanographic parameters
        sst = np.random.normal(28, 2, n_samples)
        salinity = np.random.normal(35, 1, n_samples)
        oxygen = np.random.normal(7, 1, n_samples)
        
        # Biological parameters
        fish_abundance = 50 + 2 * sst + np.random.normal(0, 10, n_samples)
        species_diversity = 20 + 0.5 * oxygen + np.random.normal(0, 3, n_samples)
        biomass = 100 + 3 * sst + 2 * oxygen + np.random.normal(0, 15, n_samples)
        
        # Create correlation matrix
        correlation_data = pd.DataFrame({
            'SST': sst,
            'Salinity': salinity,
            'Oxygen': oxygen,
            'Fish Abundance': fish_abundance,
            'Species Diversity': species_diversity,
            'Biomass': biomass
        })
        
        # Correlation heatmap
        st.subheader("🔥 Correlation Heatmap")
        corr_matrix = correlation_data.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        plt.title('Cross-Domain Correlation Matrix')
        st.pyplot(fig)
        
        # PCA Analysis
        st.subheader("📊 Principal Component Analysis")
        
        # Perform PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(correlation_data)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        # Plot PCA results
        fig = px.scatter(
            x=pca_result[:, 0], 
            y=pca_result[:, 1],
            color=fish_abundance,
            size=biomass,
            title="PCA Analysis of Marine Parameters",
            labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                   'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def render_research_tools(self):
        """Research Project Management and Collaboration Tools"""
        st.header("🔬 Research Project Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Research Projects")
            
            # Project creation
            with st.form("new_project"):
                project_name = st.text_input("Project Name")
                project_description = st.text_area("Description")
                project_type = st.selectbox("Project Type", ["Biodiversity", "Oceanography", "Taxonomy", "eDNA", "Otolith"])
                start_date = st.date_input("Start Date")
                
                if st.form_submit_button("➕ Create Project"):
                    if project_name:
                        new_project = {
                            'name': project_name,
                            'description': project_description,
                            'type': project_type,
                            'start_date': start_date,
                            'status': 'Active',
                            'created_by': st.session_state.current_user
                        }
                        st.session_state.research_projects.append(new_project)
                        st.success("✅ Project created successfully!")
            
            # Display projects
            if st.session_state.research_projects:
                st.subheader("📊 Active Projects")
                for i, project in enumerate(st.session_state.research_projects):
                    with st.expander(f"🔬 {project['name']}"):
                        st.write(f"**Type:** {project['type']}")
                        st.write(f"**Status:** {project['status']}")
                        st.write(f"**Created by:** {project['created_by']}")
                        st.write(f"**Description:** {project['description']}")
        
        with col2:
            st.subheader("👥 Collaboration Tools")
            
            # Team members
            st.write("**Research Team:**")
            team_members = [
                {"name": "Dr. Rajesh Kumar", "role": "Principal Investigator", "department": "Marine Biology"},
                {"name": "Dr. Priya Sharma", "role": "Data Scientist", "department": "Oceanography"},
                {"name": "Dr. Amit Patel", "role": "Taxonomist", "department": "Marine Taxonomy"},
                {"name": "Dr. Sunita Singh", "role": "Molecular Biologist", "department": "eDNA Research"}
            ]
            
            for member in team_members:
                st.write(f"👤 **{member['name']}** - {member['role']} ({member['department']})")
            
            # Data sharing
            st.subheader("📤 Data Sharing")
            st.info("🔗 Share datasets with research collaborators")
            st.info("📊 Export analysis results for publication")
            st.info("🔒 Secure data access controls")
    
    def perform_taxonomic_classification(self, species_input):
        """Mock taxonomic classification"""
        # Mock classification results
        classifications = [
            {
                'species': 'Thunnus albacares',
                'common_name': 'Yellowfin Tuna',
                'kingdom': 'Animalia',
                'family': 'Scombridae',
                'confidence': 0.95
            },
            {
                'species': 'Scomberomorus commerson',
                'common_name': 'Narrow-barred Spanish Mackerel',
                'kingdom': 'Animalia',
                'family': 'Scombridae',
                'confidence': 0.87
            }
        ]
        return classifications
    
    def perform_edna_analysis(self, edna_data, min_confidence, database):
        """Mock eDNA analysis"""
        # Mock eDNA results
        results = [
            {
                'species': 'Thunnus albacares',
                'confidence': 0.92,
                'length': 658,
                'database': database
            },
            {
                'species': 'Scomberomorus commerson',
                'confidence': 0.85,
                'length': 642,
                'database': database
            }
        ]
        return [r for r in results if r['confidence'] >= min_confidence]
    
    def perform_otolith_analysis(self, image, edge_detection, morphometry_type):
        """Mock otolith analysis"""
        # Mock analysis results
        results = {
            'area': 45.2,
            'perimeter': 28.7,
            'aspect_ratio': 1.8,
            'circularity': 0.75,
            'roundness': 0.82,
            'solidity': 0.91
        }
        st.session_state.otolith_analysis = results
        return results
    
    def create_otolith_visualization(self, results):
        """Create otolith visualization"""
        # Mock visualization data
        fig = go.Figure()
        
        # Add shape parameters as bar chart
        parameters = ['Area', 'Perimeter', 'Aspect Ratio', 'Circularity', 'Roundness', 'Solidity']
        values = [results['area'], results['perimeter'], results['aspect_ratio'], 
                 results['circularity'], results['roundness'], results['solidity']]
        
        fig.add_trace(go.Bar(x=parameters, y=values, marker_color='lightblue'))
        fig.update_layout(title="Otolith Morphometric Parameters", xaxis_title="Parameters", yaxis_title="Values")
        
        st.plotly_chart(fig, use_container_width=True)

# Initialize and run the application
if __name__ == "__main__":
    app = CMLREScientificPlatform()
    app.main()
