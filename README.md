# Fork & Friends Yelp Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Apache Spark](https://img.shields.io/badge/Apache_Spark-3.0+-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626)
![Flask](https://img.shields.io/badge/Flask-3.1+-green)
![DeepSeek AI](https://img.shields.io/badge/DeepSeek-AI-purple)
![Langchain](https://img.shields.io/badge/Langchain-Framework-teal)

Fork & Friends Yelp Analysis is a comprehensive big data analytics platform that processes over 6.6 million Yelp reviews and 192,000+ businesses to deliver intelligent insights and AI-powered recommendations. This repository contains the data analysis pipeline, AI recommendation engine, and analytics notebooks that power the Fork & Friends platform's backend intelligence.

## Important Links

[![Live Demo](https://img.shields.io/badge/Live_Demo-Visit_Site-2ea44f?style=for-the-badge&logo=vercel)](https://fork-and-friends.onrender.com/)
[![Client Repository](https://img.shields.io/badge/Client_Code-GitHub-blue?style=for-the-badge&logo=github)](https://github.com/NasimRanaFeroz/Fork-Friends_Front-End)
[![Server Repository](https://img.shields.io/badge/Server_Code-GitHub-blue?style=for-the-badge&logo=github)](https://github.com/mdnezam-uddin/Fork-Friends_Back-End)
[![Data Analysis](https://img.shields.io/badge/Big_Data_Analysis-GitHub-orange?style=for-the-badge&logo=github)](https://github.com/azizerorahman/Fork-Friends_Yelp-Analysis)

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
- [Analysis Modules](#analysis-modules)
- [AI Recommendation Engine](#ai-recommendation-engine)
- [Project Structure](#project-structure)

## Features

- **Big Data Processing**: Apache Spark-based analysis of millions of reviews and business records
- **AI-Powered Recommendations**: DeepSeek AI integration for intelligent friend and business suggestions
- **Comprehensive Analytics**: Six different analysis modules covering all aspects of the Yelp dataset
- **Real-time API**: Flask-based API server for serving analytics and AI recommendations
- **Interactive Notebooks**: Jupyter notebooks for data exploration and visualization
- **Vector Database**: Chroma DB integration for semantic search and retrieval
- **Sentiment Analysis**: Advanced NLP processing for review sentiment analysis
- **Geographic Analysis**: Location-based insights across 10 metropolitan areas
- **Collaborative Filtering**: Advanced recommendation algorithms using LangChain
- **RAG System**: Retrieval-Augmented Generation for contextual recommendations

## Technologies Used

- **Apache Spark**: Distributed big data processing framework
- **Python**: Core programming language for data analysis
- **Jupyter Notebooks**: Interactive data exploration and visualization
- **Flask**: Web framework for API server
- **DeepSeek AI**: Advanced AI model for recommendations via OpenRouter
- **LangChain**: Framework for building AI applications with LLMs
- **Chroma DB**: Vector database for semantic search and retrieval
- **HuggingFace**: Embeddings and transformer models
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Matplotlib & Seaborn**: Data visualization libraries

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- Apache Spark 3.0+
- Jupyter Notebook or JupyterLab
- Git
- 16GB+ RAM (recommended for processing large datasets)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/azizerorahman/Fork-Friends_Yelp-Analysis.git
   cd Fork-Friends_Yelp-Analysis
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:

   ```bash
   # Create .env file in DeepSeek directory
   cd DeepSeek
   echo OPENROUTER_API_KEY=your_openrouter_api_key > .env
   ```

5. Download the Yelp dataset (optional - for full analysis):
   - Visit [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/)
   - Download the dataset files
   - Place them in the appropriate data directory

## Environment Variables

| Variable | Description |
|----------|-------------|
| OPENROUTER_API_KEY | API key for accessing DeepSeek AI through OpenRouter |

## Usage

### Running Jupyter Notebooks

1. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

2. Navigate to the `Analysis/` directory and open any of the analysis notebooks:
   - `Business.ipynb` - Business and merchant analysis
   - `User.ipynb` - User behavior and demographics analysis
   - `Review.ipynb` - Review content and sentiment analysis
   - `Rating.ipynb` - Rating patterns and distribution analysis
   - `Check-in.ipynb` - Check-in trends and temporal analysis
   - `Comprehensive.ipynb` - Integrated multi-dimensional analysis

### Running the AI Recommendation Server

1. Navigate to the DeepSeek directory:

   ```bash
   cd DeepSeek
   ```

2. Start the Flask server:

   ```bash
   python main.py
   ```

3. The API server will be available at `http://localhost:5000`

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/friend-recommendation` | Get AI-powered friend recommendations |
| POST | `/business-recommendation` | Get personalized business recommendations |
| POST | `/chat` | General AI chat for Yelp-related queries |

## Analysis Modules

### Business Analysis (`Business.ipynb`)

- **Top Merchants Identification**: Analysis of the 20 most common merchants across the U.S.
- **Geographic Distribution**: City and state rankings by merchant density
- **Rating Analysis**: Merchant ratings and performance metrics
- **Category Insights**: Restaurant type exploration and trends
- **Market Penetration**: Business expansion patterns across regions

### User Analysis (`User.ipynb`)

- **User Growth Trends**: Yearly user registration and engagement statistics
- **Elite vs Regular Users**: Comparative analysis of user types
- **Top Reviewers**: Identification of most active and influential users
- **Engagement Metrics**: User activity patterns and contribution analysis
- **Demographics**: User distribution across different markets

### Review Analysis (`Review.ipynb`)

- **Temporal Trends**: Yearly review volume and growth patterns
- **Sentiment Analysis**: Advanced NLP processing for review sentiment
- **Content Analysis**: Word frequency studies and text mining
- **Language Patterns**: Common phrases and review structure analysis
- **Quality Metrics**: Review helpfulness and usefulness analysis

### Rating Analysis (`Rating.ipynb`)

- **Distribution Analysis**: Rating patterns across 1-5 star scale
- **Business Performance**: Five-star business identification and trends
- **Temporal Patterns**: Weekly and seasonal rating variations
- **Correlation Studies**: Rating vs. other business metrics
- **Quality Assessment**: Rating reliability and consistency analysis

### Check-in Analysis (`Check-in.ipynb`)

- **Temporal Patterns**: Check-in trends across different time periods
- **Geographic Analysis**: City-wise check-in distribution and patterns
- **Growth Metrics**: Yearly check-in volume and trend analysis
- **Business Impact**: Check-in correlation with business success
- **User Behavior**: Check-in patterns and frequency analysis

### Comprehensive Analysis (`Comprehensive.ipynb`)

- **Integrated Analytics**: Cross-dataset insights and correlations
- **Multi-dimensional Analysis**: Combined metrics from all data sources
- **Predictive Modeling**: Business success prediction algorithms
- **Market Intelligence**: Comprehensive market trend analysis
- **Strategic Insights**: Data-driven business recommendations

## AI Recommendation Engine

The DeepSeek AI integration provides intelligent recommendations through:

### Friend Recommendations

- **Collaborative Filtering**: User similarity based on rating patterns
- **Interest Matching**: AI analysis of review content and preferences
- **Geographic Proximity**: Location-based friend suggestions
- **Social Network Analysis**: Connection recommendations through mutual interests

### Business Recommendations

- **Personalized Suggestions**: AI-powered restaurant recommendations
- **Context-Aware**: Location, time, and preference-based suggestions
- **Sentiment-Based**: Recommendations based on review sentiment analysis
- **Trend Analysis**: Popular and emerging business recommendations

### AI Engine Features

- **RAG System**: Retrieval-Augmented Generation for contextual responses
- **Vector Search**: Semantic similarity using HuggingFace embeddings
- **Caching**: LRU cache for improved response times
- **Fallback Mechanisms**: Graceful degradation when data is unavailable
- **Rate Limiting**: API protection and usage optimization

## Project Structure

```text
Fork-Friends_Yelp-Analysis/
├── Analysis/
│   ├── Business.ipynb
│   ├── Check-in.ipynb
│   ├── Comprehensive.ipynb
│   ├── Rating.ipynb
│   ├── Review.ipynb
│   └── User.ipynb
├── DeepSeek/
│   ├── main.py
│   ├── data/
│   └── .env
├── requirements.txt
├── .gitignore
└── README.md
```

## Contributors

- **[Azizur Rahman](https://github.com/azizerorahman/)** - Project Lead & Data Scientist
- **[Nasim Rana Feroz](https://github.com/nasimranaferoz/)** - Frontend Developer & UI/UX Designer
- **[MD Nezam Uddin](https://github.com/mdnezamuddin/)** - Backend Developer & Database Engineer

[![Contributors](https://contrib.rocks/image?repo=azizerorahman/Fork-Friends_Yelp-Analysis)](https://github.com/azizerorahman/Fork-Friends_Yelp-Analysis/graphs/contributors)

---

**Note**: This project processes the publicly available [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/) for educational and research purposes. All data analysis and recommendations are based on anonymized user data and publicly accessible business information. Please ensure compliance with Yelp's Terms of Service when using their dataset.
