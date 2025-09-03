import os
from typing import Dict, List, Any
from datetime import datetime, timedelta
import json
import requests
import openai

from dotenv import load_dotenv
load_dotenv()

class News:
    def __init__(self, config: Dict[str, Any], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = config
        self._api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self._api_key
    
    def process_data(self, context: str) -> Dict[str, Any]:
        """
        Process news data for a given context and analyze sentiment and risk levels.
        
        Args:
            context: The search context (e.g., "bitcoin", "aave", "uniswap")
            
        Returns:
            Dictionary containing news data, sentiment analysis, and risk assessment
        """
        try:
            news_data = self.get_news(context)
            sentiment_analysis = self.analyze_sentiment(news_data)
            risk_assessment = self.assess_risk_level(sentiment_analysis)
            
            return {
                'context': context,
                'sentiment_analysis': sentiment_analysis,
                'risk_assessment': risk_assessment,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'error': str(e),
                'context': context,
                'timestamp': datetime.now().isoformat()
            }

    def get_news(self, context: str) -> Dict[str, Any]:
        """
        Fetch news data from Hacker News API for the given context.
        
        Args:
            context: Search query for news articles
            
        Returns:
            Dictionary containing news articles and metadata
        """
        # Calculate timestamp for last 30 days
        thirty_days_ago = int((datetime.now() - timedelta(days=30)).timestamp())
        
        url = f"{self._config['url']}/search_by_date"
        params = {
            'query': context,
            'tags': 'story',
            'hitsPerPage': 3,
            'numericFilters': f'created_at_i>{thirty_days_ago}'
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def analyze_sentiment(self, news_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment of news articles using OpenAI.
        
        Args:
            news_data: News data from get_news method
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        if not news_data.get('hits'):
            return {'sentiment': 'neutral', 'confidence': 0.5, 'summary': 'No news articles found'}
        
        # Extract titles and content from news articles
        articles_text = []
        for hit in news_data['hits'][:3]:  # Analyze top 3 articles
            title = hit.get('title', '')
            content = hit.get('content', '')
            url = hit.get('url', '')
            articles_text.append(f"Title: {title}\nContent: {content}\nURL: {url}\n")
        
        combined_text = "\n---\n".join(articles_text)
        
        try:
            # Try gpt-4o-mini first, fallback to gpt-3.5-turbo if needed
            models_to_try = ['gpt-4o-mini', 'gpt-3.5-turbo']
            
            for model in models_to_try:
                try:
                    response = openai.ChatCompletion.create(
                        api_key = self._api_key,
                        model=model,
                        messages=[
                            {
                                "role": "system", 
                                "content": "You are a financial sentiment analysis expert. Analyze the sentiment of the provided news articles and return a JSON response with the following structure: {\"sentiment\": \"strong_positive/positive/neutral/negative/strong_negative\", \"confidence\": 0.0-1.0, \"summary\": \"brief summary of sentiment\", \"key_points\": [\"point1\", \"point2\"]}"
                            },
                            {
                                "role": "user", 
                                "content": f"Analyze the sentiment of these news articles related to cryptocurrency and DeFi:\n\n{combined_text}"
                            }
                        ],
                        temperature=0.3
                    )
                    
                    result = response.choices[0].message.content
                    # Try to parse JSON response
                    try:
                        return json.loads(result)
                    except json.JSONDecodeError:
                        return {
                            'sentiment': 'neutral',
                            'confidence': 0.5,
                            'summary': result,
                            'key_points': []
                        }
                        
                except Exception as model_error:
                    print(f"Failed with model {model}: {str(model_error)}")
                    if model == models_to_try[-1]:  # Last model failed
                        raise model_error
                    continue
                    
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'summary': f'Error in sentiment analysis: {str(e)}',
                'key_points': []
            }

    def assess_risk_level(self, sentiment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess risk level based on news sentiment and context.
        
        Args:
            news_data: News data from get_news method
            sentiment_analysis: Sentiment analysis results
            context: The context being analyzed (coin/protocol name)
            
        Returns:
            Dictionary containing risk assessment
        """
        sentiment = sentiment_analysis.get('sentiment', 'neutral')
        base_score = 50
        
        sentiment_adjustment = {
            'strong_positive': -20,
            'positive': -10,
            'neutral': 0,
            'negative': 10,
            'strong_negative': 20
        }
        
        risk_score = sentiment_adjustment.get(sentiment, 0)
        
        return {
            'risk': risk_score,
            'risk_factors': [f'Sentiment adjustment: {risk_score}'],
            'recommendations': ['Monitor closely', 'Conduct additional research'],
            'market_impact': 'neutral',
            'confidence': 0.3,
            'sentiment_adjustment': {
                'sentiment': sentiment,
                'base_score': base_score,
            }
        }

    def analyze_multiple_contexts(self, contexts: List[str]) -> Dict[str, Any]:
        """
        Analyze multiple contexts (coins/protocols) and compare their risk levels.
        
        Args:
            contexts: List of contexts to analyze
            
        Returns:
            Dictionary containing analysis for all contexts
        """
        results = {}
        for context in contexts:
            results[context] = self.process_data(context)
        
        # Add comparative analysis
        results['comparative_analysis'] = self._compare_risk_levels(results)
        return results

    def _compare_risk_levels(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare risk levels across different contexts.
        
        Args:
            results: Results from analyze_multiple_contexts
            
        Returns:
            Comparative analysis
        """
        risk_scores = {}
        for context, data in results.items():
            if context != 'comparative_analysis' and 'risk_assessment' in data:
                risk_scores[context] = data['risk_assessment'].get('risk_score', 50)
        
        if not risk_scores:
            return {'message': 'No risk scores available for comparison'}
        
        # Find highest and lowest risk
        highest_risk = max(risk_scores.items(), key=lambda x: x[1])
        lowest_risk = min(risk_scores.items(), key=lambda x: x[1])
        
        return {
            'highest_risk': {'context': highest_risk[0], 'score': highest_risk[1]},
            'lowest_risk': {'context': lowest_risk[0], 'score': lowest_risk[1]},
            'average_risk': sum(risk_scores.values()) / len(risk_scores),
            'risk_distribution': risk_scores
        }
        