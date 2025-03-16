import os
import requests
import json
import warnings
import re
from collections import defaultdict
from typing import Any, List, Optional
import time
import concurrent.futures
from functools import lru_cache
import logging
from tqdm import tqdm
from flask import Flask, request, jsonify
from flask_cors import CORS


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


app = Flask(__name__)
CORS(app)


class DeepSeekChat(BaseChatModel):
    openrouter_api_key: str
    model_name: str = "deepseek/deepseek-r1:free"
    temperature: float = 0.7
    
    def _generate(
        self, messages: List[Any], stop: Optional[List[str]] = None, 
        run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> ChatResult:
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "X-Title": "Fork and Friends Chatbot",
            "Content-Type": "application/json"
        }
        
        formatted_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                formatted_messages.append({"role": "assistant", "content": message.content})
            elif isinstance(message, SystemMessage):
                formatted_messages.append({"role": "system", "content": message.content})
            else:
                formatted_messages.append({"role": "user", "content": str(message)})
        
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": self.temperature
        }
        
        if stop:
            payload["stop"] = stop
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                response_data = response.json()
                break
            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                if attempt == max_retries - 1:
                    logger.error(f"API request failed after {max_retries} attempts: {str(e)}")
                    raise
                wait_time = 2 ** attempt
                logger.warning(f"API request failed, retrying in {wait_time}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
        
        message_content = response_data["choices"][0]["message"]["content"]
        
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=message_content))],
            llm_output=response_data
        )
    
    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"



BUSINESS_CACHE = {}
CITY_INDEX = defaultdict(set)
CATEGORY_INDEX = defaultdict(set)
BUSINESS_ID_TO_CATEGORIES = {}
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "default_api_key")
CACHE_INITIALIZED = False
RAG_CHAIN = None


CITY_PATTERN1 = re.compile(r"(?:business(?:es)? in|places in|recommend in|suggest in) ([a-zA-Z\s]+)(?:city)?")
CITY_PATTERN2 = re.compile(r"in ([a-zA-Z\s]+)(?:city)?")
CITY_PATTERN3 = re.compile(r"([a-zA-Z]+) city")
CATEGORY_PATTERN = re.compile(r"(?:recommend|suggest|find|looking for|want) (?:a|an|some) ([a-zA-Z\s]+) (?:and|&) ([a-zA-Z\s]+) in")


CITY_ALIASES = {
    "phonix": "phoenix", "pheonix": "phoenix", "phenix": "phoenix", "phx": "phoenix",
    "vegas": "las vegas", "lv": "las vegas", "sb": "santa barbara", "pitt": "pittsburgh",
    "pgh": "pittsburgh", "char": "charlotte", "tuc": "tucson", "tucson az": "tucson",
    "phoenix az": "phoenix", "nyc": "new york", "la": "los angeles", "sf": "san francisco"
}

CATEGORY_ALIASES = {
    "resturent": "restaurant", "resturants": "restaurant", "resturant": "restaurant",
    "restraunt": "restaurant", "restaraunt": "restaurant", "eatery": "restaurant",
    "dining": "restaurant", "diner": "restaurant", "coffeeshop": "coffee",
    "coffee shop": "coffee", "bakeries": "bakery", "bake shop": "bakery",
    "pastry": "bakery", "pastries": "bakery", "cafe": "coffee", "pizzeria": "pizza"
}

COMMON_CATEGORIES = {
    "restaurant", "cafe", "coffee", "pizza", "italian", "chinese", "mexican", 
    "japanese", "sushi", "thai", "indian", "bar", "pub", "bakery", "breakfast",
    "brunch", "lunch", "dinner", "fast food", "dessert", "ice cream", "salon",
    "spa", "gym", "fitness", "yoga", "hotel", "motel", "shopping", "retail",
    "clothing", "fashion", "grocery", "market", "pharmacy", "doctor", "dentist", "hospital",
    "automotive", "car repair", "gas station", "bank", "atm", "school", "university",
    "library", "bookstore", "movie theater", "theater", "museum", "art gallery",
    "park", "playground", "zoo", "aquarium", "nightlife", "club", "dance"
}


for alias in CATEGORY_ALIASES:
    COMMON_CATEGORIES.add(alias)

TOP_RATED_KEYWORDS = {
    "top rated", "highest rated", "best rated", "top star", "highest star", 
    "5 star", "five star", "top businesses", "best businesses", "top 5", 
    "top five", "most rated", "highest rating", "best rating", "highly rated",
    "highly-rated", "best", "great", "excellent", "outstanding", "recommend",
    "good", "nice", "popular", "famous", "well-known", "well known"
}

BUSINESS_KEYWORDS = {
    "restaurant", "food", "eat", "dining", "cafe", "coffee", "shop", "store", 
    "business", "place", "recommend", "suggestion", "best", "top", "rated", 
    "review", "star", "yelp", "location", "address", "open", "hour", "price",
    "expensive", "cheap", "affordable", "menu", "reservation", "booking",
    "takeout", "delivery", "dine-in", "cuisine", "bar", "pub", "nightlife",
    "entertainment", "service", "mall", "shopping", "retail", "salon", "spa",
    "gym", "fitness", "hotel", "motel", "lodging", "stay", "accommodation"
}


for alias in CATEGORY_ALIASES:
    BUSINESS_KEYWORDS.add(alias)

CITY_NAMES = {
    "tucson", "phoenix", "las vegas", "santa barbara", "toronto", "pittsburgh", 
    "charlotte", "new york", "los angeles", "chicago", "houston", "philadelphia",
    "san antonio", "san diego", "dallas", "san jose", "austin", "jacksonville",
    "fort worth", "columbus", "san francisco", "seattle", "denver", "boston"
}


for alias in CITY_ALIASES:
    CITY_NAMES.add(alias)

BUSINESS_PATTERNS = [
    re.compile(r"where (can|should) i"),
    re.compile(r"looking for"),
    re.compile(r"recommend"),
    re.compile(r"suggest"),
    re.compile(r"find me"),
    re.compile(r"what('s| is) the best"),
    re.compile(r"top \d+"),
    re.compile(r"highest rated"),
    re.compile(r"near me"),
    re.compile(r"in the area")
]

NEGATIVE_PHRASES = {
    "i don't have", "i couldn't find", "no information", "no data", "not available", 
    "no businesses", "no restaurants", "no cafes", "no bakeries"
}

PEOPLE_KEYWORDS = {
    "friend", "friends", "mate", "mates", "buddy", "buddies", "pal", "pals",
    "people", "person", "meet", "connect", "connection", "social", "networking",
    "dating", "date", "relationship", "relationships", "partner", "companion",
    "acquaintance", "colleague", "coworker", "neighbor", "classmate"
}


def initialize_cache(yelp_file_path, limit=20000):
    """Initialize cache with business data for faster lookups using parallel processing"""
    global BUSINESS_CACHE, CITY_INDEX, CATEGORY_INDEX, CACHE_INITIALIZED, BUSINESS_ID_TO_CATEGORIES
    
    if CACHE_INITIALIZED:
        return
    
    logger.info("Initializing business data cache...")
    
    
    batch_size = 1000
    businesses_processed = 0
    
    with open(yelp_file_path, 'r', encoding='utf-8') as file:
        with tqdm(total=min(limit, 20000), desc="Processing businesses", unit="business") as pbar:
            while businesses_processed < limit:
                batch_lines = []
                for _ in range(batch_size):
                    line = file.readline()
                    if not line:
                        break
                    batch_lines.append(line)
                
                if not batch_lines:
                    break
                
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, os.cpu_count() + 4)) as executor:
                    for line in batch_lines:
                        try:
                            business = json.loads(line.strip())
                            business_id = business.get('business_id')
                            
                            if not business_id:
                                continue
                            
                            BUSINESS_CACHE[business_id] = business
                            
                            city = business.get('city', '').lower()
                            if city:
                                CITY_INDEX[city].add(business_id)
                            
                            categories_str = business.get('categories', '')
                            if categories_str:
                                categories_list = [cat.strip().lower() for cat in categories_str.split(',')]
                                BUSINESS_ID_TO_CATEGORIES[business_id] = set(categories_list)
                                for category in categories_list:
                                    CATEGORY_INDEX[category].add(business_id)
                            
                            businesses_processed += 1
                            pbar.update(1)
                            
                            if businesses_processed >= limit:
                                break
                        except json.JSONDecodeError:
                            continue
    
    logger.info(f"Cache initialized with {len(BUSINESS_CACHE)} businesses")
    logger.info(f"Cities indexed: {len(CITY_INDEX)}")
    logger.info(f"Categories indexed: {len(CATEGORY_INDEX)}")
    
    CACHE_INITIALIZED = True


@lru_cache(maxsize=512)  
def normalize_city_name(city_name):
    """Normalize city name using aliases and spelling corrections with caching"""
    if not city_name:
        return None
    
    city_name = city_name.lower().strip()
    
    if city_name in CITY_ALIASES:
        return CITY_ALIASES[city_name]
    
    for alias, normalized in CITY_ALIASES.items():
        if city_name in alias or alias in city_name:
            return normalized
    
    return city_name


@lru_cache(maxsize=512)  
def normalize_category(category):
    """Normalize category name using aliases and spelling corrections with caching"""
    if not category:
        return None
    
    category = category.lower().strip()
    
    if category in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[category]
    
    for alias, normalized in CATEGORY_ALIASES.items():
        if category == alias or (len(category) > 4 and (category in alias or alias in category)):
            return normalized
    
    return category


def extract_city_name(query):
    """Extract city name from query using precompiled regex patterns"""
    query_lower = query.lower()
    
    
    for city in CITY_NAMES:
        if city in query_lower:
            return city
    
    for alias, city in CITY_ALIASES.items():
        if alias in query_lower:
            return city
    
    
    match = CITY_PATTERN1.search(query_lower)
    if match:
        return normalize_city_name(match.group(1).strip())
    
    match = CITY_PATTERN2.search(query_lower)
    if match:
        return normalize_city_name(match.group(1).strip())
    
    match = CITY_PATTERN3.search(query_lower)
    if match:
        return normalize_city_name(match.group(1).strip())
    
    return None


def extract_categories(query):
    """Extract multiple categories from query"""
    query_lower = query.lower()
    
    found_categories = set()
    
    
    for category in COMMON_CATEGORIES:
        if f" {category} " in f" {query_lower} " or f" {category}," in f" {query_lower} ":
            normalized = normalize_category(category)
            if normalized:
                found_categories.add(normalized)
    
    
    if not found_categories:
        match = CATEGORY_PATTERN.search(query_lower)
        if match:
            cat1 = normalize_category(match.group(1).strip())
            cat2 = normalize_category(match.group(2).strip())
            if cat1:
                found_categories.add(cat1)
            if cat2:
                found_categories.add(cat2)
    
    return list(found_categories) if found_categories else None


def is_top_rated_query(query):
    """Check if the query is asking for top rated businesses"""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in TOP_RATED_KEYWORDS)


def get_safe_string(value, default=""):
    """Safely get a string value, handling None values"""
    if value is None:
        return default
    return str(value).lower()


def generate_human_recommendation(businesses, category, city):
    """Generate a human-like recommendation for businesses"""
    if not businesses:
        return f"I couldn't find any {category} places in {city}. Perhaps try another category or city?"
    
    
    sorted_businesses = sorted(businesses, key=lambda x: (float(x.get("stars", 0) or 0), int(x.get("review_count", 0) or 0)), reverse=True)
    top_businesses = sorted_businesses[:3]
    
    multiple_categories = " and " in category
    
    if multiple_categories:
        response = f"For places that offer both {category} in {city}, I'd recommend checking out "
    else:
        response = f"For {category} in {city}, I'd recommend checking out "
    
    if len(top_businesses) == 1:
        business = top_businesses[0]
        response += f"{business['name']}. "
        response += f"It's rated {business.get('stars', 'N/A')} stars "
        if business.get('review_count'):
            response += f"based on {business.get('review_count')} reviews. "
        response += f"You can find it at {business.get('address', 'N/A')}. "
        if business.get('categories'):
            response += f"They specialize in {business.get('categories')}."
    
    elif len(top_businesses) == 2:
        business1 = top_businesses[0]
        business2 = top_businesses[1]
        response += f"{business1['name']} or {business2['name']}. "
        response += f"{business1['name']} has {business1.get('stars', 'N/A')} stars "
        if business1.get('review_count'):
            response += f"from {business1.get('review_count')} reviews, "
        response += f"while {business2['name']} is rated {business2.get('stars', 'N/A')} stars. "
        response += f"You can find {business1['name']} at {business1.get('address', 'N/A')} "
        response += f"and {business2['name']} at {business2.get('address', 'N/A')}."
    
    else:
        response += f"{top_businesses[0]['name']}, {top_businesses[1]['name']}, or {top_businesses[2]['name']}. "
        response += f"{top_businesses[0]['name']} is the highest rated at {top_businesses[0].get('stars', 'N/A')} stars. "
        response += f"All three are well-reviewed and popular choices for {category} in {city}."
    
    return response


def find_businesses_with_all_categories(businesses, categories):
    """Find businesses that have ALL the specified categories"""
    
    if len(businesses) > 100:  
        matching_business_ids = None
        categories_lower = [cat.lower() for cat in categories]
        
        for category in categories_lower:
            if category in CATEGORY_INDEX:
                category_business_ids = CATEGORY_INDEX[category]
                if matching_business_ids is None:
                    matching_business_ids = category_business_ids.copy()
                else:
                    matching_business_ids &= category_business_ids
        
        if matching_business_ids:
            return [BUSINESS_CACHE[bid] for bid in matching_business_ids if bid in BUSINESS_CACHE]
    
    
    matching_businesses = []
    categories_lower = [cat.lower() for cat in categories]
    
    for business in businesses:
        business_id = business.get('business_id')
        
        if business_id in BUSINESS_ID_TO_CATEGORIES:
            business_categories = BUSINESS_ID_TO_CATEGORIES[business_id]
            if all(any(category in cat for cat in business_categories) for category in categories_lower):
                matching_businesses.append(business)
        else:
            business_categories = get_safe_string(business.get('categories', ''))
            if all(category in business_categories for category in categories_lower):
                matching_businesses.append(business)
    
    return matching_businesses


def get_businesses_by_city(city_name):
    """Get businesses in a specific city"""
    global BUSINESS_CACHE, CITY_INDEX
    
    matching_businesses = []
    if not city_name:
        return matching_businesses
    
    city_name = city_name.lower()
    
    
    if city_name in CITY_INDEX:
        business_ids = CITY_INDEX[city_name]
        return [BUSINESS_CACHE[bid] for bid in business_ids if bid in BUSINESS_CACHE]
    
    
    for city in CITY_INDEX:
        if city_name in city or city in city_name:
            business_ids = CITY_INDEX[city]
            city_businesses = [BUSINESS_CACHE[bid] for bid in business_ids if bid in BUSINESS_CACHE]
            matching_businesses.extend(city_businesses)
            
            
            if len(matching_businesses) > 100:
                break
    
    return matching_businesses


def get_businesses_by_category(category, city_businesses=None):
    """Get businesses in a specific category, optionally filtered by city"""
    global BUSINESS_CACHE, CATEGORY_INDEX, BUSINESS_ID_TO_CATEGORIES
    
    category_lower = category.lower()
    
    
    if city_businesses is None:
        business_ids = CATEGORY_INDEX.get(category_lower, set())
        return [BUSINESS_CACHE[bid] for bid in business_ids if bid in BUSINESS_CACHE]
    
    
    if len(city_businesses) > 100:
        city_business_ids = {b.get('business_id') for b in city_businesses if b.get('business_id')}
        category_business_ids = CATEGORY_INDEX.get(category_lower, set())
        matching_ids = city_business_ids.intersection(category_business_ids)
        return [BUSINESS_CACHE[bid] for bid in matching_ids if bid in BUSINESS_CACHE]
    
    
    matching = []
    for b in city_businesses:
        business_id = b.get('business_id')
        if business_id in BUSINESS_ID_TO_CATEGORIES:
            if any(category_lower in cat for cat in BUSINESS_ID_TO_CATEGORIES[business_id]):
                matching.append(b)
        elif category_lower in get_safe_string(b.get('categories', '')):
            matching.append(b)
    
    return matching


def is_business_related_query(query):
    """Determine if a query is related to business recommendations"""
    query_lower = query.lower()
    
    
    if any(keyword in query_lower for keyword in BUSINESS_KEYWORDS):
        return True
    
    
    if any(city in query_lower for city in CITY_NAMES):
        return True
    
    
    return any(pattern.search(query_lower) for pattern in BUSINESS_PATTERNS)


def is_people_related_query(query):
    """Determine if a query is asking for friend recommendations or people connections"""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in PEOPLE_KEYWORDS)


@lru_cache(maxsize=128)  
def ask_deepseek(query, api_key):
    """Send a query directly to DeepSeek via OpenRouter API with caching"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek/deepseek-r1:free",
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ]
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
            if attempt == max_retries - 1:
                logger.error(f"DeepSeek API error: {str(e)}")
                return f"I'm sorry, I couldn't process your request at the moment. Please try again later."
            wait_time = 2 ** attempt
            logger.warning(f"DeepSeek API request failed, retrying in {wait_time}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait_time)


def get_friend_recommendation(query, city_name=None):
    """Generate a response for queries asking about friends or people connections"""
    prompt = f"The user is looking for friend recommendations or ways to meet people"
    if city_name:
        prompt += f" in {city_name}"
    prompt += f". Their query was: '{query}'. "
    prompt += "Please provide a helpful response with specific suggestions for how they could meet people, "
    prompt += "make friends, or connect with others in this location. Include specific venues, apps, or events "
    prompt += "where possible. If the Yelp dataset might contain relevant community spaces, mention those too. "
    prompt += "Please provide a concise list of names of friends "
    prompt += "No additional information or suggestions is necessary—just the names. "
    prompt += "Be conversational and friendly."
    prompt += "Please provide a helpful response with specific suggestions for how they could meet people, "
    prompt += "make friends, or connect with others in this location. Include specific venues, apps, or events "
    prompt += "where possible. If the Yelp dataset might contain relevant community spaces, mention those too. "
    
    return ask_deepseek(prompt, OPENROUTER_API_KEY)


def get_generic_business_response(query, city_name=None, categories=None):
    """Generate a generic business response using DeepSeek when we don't have data"""
    prompt = f"I need information about {', '.join(categories) if categories else 'businesses'} "
    prompt += f"in {city_name.title() if city_name else 'various cities'}. "
    prompt += f"The user asked: '{query}'. "
    prompt += "Please provide a helpful response about what kinds of businesses they might find there, "
    prompt += "popular options, and suggestions for how they could find more specific information. "
    prompt += "Don't mention that you don't have data - just be helpful and informative."
    
    return ask_deepseek(prompt, OPENROUTER_API_KEY)


def process_direct_query(query):
    """Process direct queries without using vector search"""
    query_lower = query.lower()
    is_top_rated = is_top_rated_query(query)
    
    city_name = extract_city_name(query)
    categories = extract_categories(query)
    
    logger.info(f"Query: {query}")
    logger.info(f"Detected city: {city_name}")
    logger.info(f"Detected categories: {categories}")
    logger.info(f"Is top rated query: {is_top_rated}")
    
    if categories and city_name:
        city_businesses = get_businesses_by_city(city_name)
        
        if not city_businesses:
            return get_generic_business_response(query, city_name, categories)
        
        matching_businesses = find_businesses_with_all_categories(city_businesses, categories)
        
        if matching_businesses:
            categories_text = " and ".join(categories)
            return generate_human_recommendation(matching_businesses, categories_text, city_name.title())
        else:
            any_category_businesses = []
            for category in categories:
                category_businesses = get_businesses_by_category(category, city_businesses)
                any_category_businesses.extend(category_businesses)
            
            if any_category_businesses:
                categories_text = " or ".join(categories)
                return f"I couldn't find places that offer both {' and '.join(categories)} together in {city_name.title()}, but I found some that offer either {categories_text}:\n\n" + \
                       generate_human_recommendation(any_category_businesses, categories_text, city_name.title())
            else:
                return get_generic_business_response(query, city_name, categories)
    
    elif is_top_rated:
        if city_name:
            businesses = get_businesses_by_city(city_name)
            if not businesses:
                return get_generic_business_response(query, city_name, categories)
        else:
            
            businesses = list(BUSINESS_CACHE.values())[:2000]
        
        if categories:
            filtered_businesses = []
            for category in categories:
                category_businesses = get_businesses_by_category(category, businesses)
                filtered_businesses.extend(category_businesses)
            businesses = filtered_businesses
        
        businesses = [b for b in businesses if "stars" in b and b["stars"] is not None]
        
        if businesses:
            
            sorted_businesses = sorted(
                businesses, 
                key=lambda x: (float(x.get("stars", 0) or 0), int(x.get("review_count", 0) or 0)), 
                reverse=True
            )
            
            top_businesses = sorted_businesses[:5]
            
            location_text = f" in {city_name.title()}" if city_name else ""
            category_text = f" {' and '.join(categories)}" if categories else ""
            result = f"🌟 Top rated{category_text} businesses{location_text}:\n\n"
            for i, business in enumerate(top_businesses, 1):
                result += f"{i}. {business['name']} - {business.get('stars', 'N/A')} stars\n"
                result += f"   Categories: {business.get('categories', 'N/A')}\n"
                result += f"   Address: {business.get('address', 'N/A')}, {business.get('city', 'N/A')}\n\n"
            
            return result
        else:
            return get_generic_business_response(query, city_name, categories)
    
    elif city_name:
        businesses_in_city = get_businesses_by_city(city_name)
        
        if businesses_in_city:
            
            sorted_businesses = sorted(
                businesses_in_city, 
                key=lambda x: (float(x.get("stars", 0) or 0), int(x.get("review_count", 0) or 0)), 
                reverse=True
            )
            
            top_businesses = sorted_businesses[:10]
            
            result = f"📍 Top businesses in {city_name.title()}:\n\n"
            for i, business in enumerate(top_businesses, 1):
                result += f"{i}. {business['name']} - {business.get('stars', 'N/A')} stars\n"
                result += f"   Categories: {business.get('categories', 'N/A')}\n"
                result += f"   Address: {business.get('address', 'N/A')}\n\n"
            
            return result
        else:
            return get_generic_business_response(query, city_name, None)
    
    return None


def setup_rag_system(yelp_file_path):
    """Set up the RAG system with vectorstore and retrieval chain"""
    logger.info("Setting up RAG system...")
    
    try:
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        
        business_ids = list(BUSINESS_CACHE.keys())[:2000]
        
        def process_business(business_id):
            business = BUSINESS_CACHE[business_id]
            
            content = f"Business Name: {business.get('name', '')}\n"
            content += f"Address: {business.get('address', '')}, {business.get('city', '')}, {business.get('state', '')} {business.get('postal_code', '')}\n"
            content += f"Categories: {business.get('categories', '')}\n"
            content += f"Rating: {business.get('stars', 'N/A')} stars based on {business.get('review_count', 'N/A')} reviews\n"
            
            return Document(
                page_content=content,
                metadata={
                    "source": yelp_file_path, 
                    "city": business.get("city", ""),
                    "state": business.get("state", ""),
                    "stars": business.get("stars", 0),
                    "categories": get_safe_string(business.get("categories"))
                }
            )
        
        
        batch_size = 500
        documents = []
        
        for i in range(0, len(business_ids), batch_size):
            batch_ids = business_ids[i:i+batch_size]
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, os.cpu_count() + 4)) as executor:
                batch_docs = list(executor.map(process_business, batch_ids))
                documents.extend(batch_docs)
        
        
        vectorstore = Chroma.from_documents(
            documents=documents, 
            embedding=embeddings
        )
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
        
        llm = DeepSeekChat(
            openrouter_api_key=OPENROUTER_API_KEY,
            model_name="deepseek/deepseek-r1:free",
            temperature=0.7
        )
        
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=index.vectorstore.as_retriever(search_kwargs={"k": 3}),
        )
        
        logger.info("RAG system ready")
        return chain
    
    except Exception as e:
        logger.error(f"Error setting up RAG system: {str(e)}")
        return None


def process_query(query):
    """Process a query and return a response"""
    global RAG_CHAIN
    
    
    if is_people_related_query(query):
        logger.info("Processing people-related question")
        city_name = extract_city_name(query)
        try:
            response = get_friend_recommendation(query, city_name)
            return {"response": response, "type": "people"}
        except Exception as e:
            logger.error(f"Error processing people-related question: {str(e)}")
            return {"response": "I'm sorry, I couldn't process your request. Please try again later.", "error": str(e)}
    
    
    if not is_business_related_query(query):
        logger.info("Processing general question")
        try:
            response = ask_deepseek(query, OPENROUTER_API_KEY)
            return {"response": response, "type": "general"}
        except Exception as e:
            logger.error(f"Error processing general question: {str(e)}")
            return {"response": "I'm sorry, I couldn't process your request. Please try again later.", "error": str(e)}
    
    
    yelp_file_path = "data/dataset.json"
    
    
    if not os.path.exists(yelp_file_path):
        logger.error(f"Data file not found: {yelp_file_path}")
        try:
            response = ask_deepseek(query, OPENROUTER_API_KEY)
            return {"response": response, "type": "fallback", "reason": "data_file_missing"}
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            return {"response": "I'm sorry, I couldn't process your request. Please try again later.", "error": str(e)}
    
    
    try:
        if not CACHE_INITIALIZED:
            initialize_cache(yelp_file_path)
    except Exception as e:
        logger.error(f"Error initializing cache: {str(e)}")
        try:
            response = ask_deepseek(query, OPENROUTER_API_KEY)
            return {"response": response, "type": "fallback", "reason": "cache_init_error", "error": str(e)}
        except Exception as e2:
            logger.error(f"API error: {str(e2)}")
            return {"response": "I'm sorry, I couldn't process your request. Please try again later.", "error": str(e2)}
    
    
    try:
        direct_result = process_direct_query(query)
        if direct_result:
            return {"response": direct_result, "type": "direct"}
    except Exception as e:
        logger.error(f"Error in direct query processing: {str(e)}")
    
    
    try:
        
        if RAG_CHAIN is None:
            RAG_CHAIN = setup_rag_system(yelp_file_path)
        
        if RAG_CHAIN:
            chat_history = []
            result = RAG_CHAIN.invoke({"question": query, "chat_history": chat_history})
            answer = result["answer"]
            
            
            if any(phrase in answer.lower() for phrase in NEGATIVE_PHRASES):
                logger.info("RAG response was not helpful, falling back to DeepSeek")
                answer = get_generic_business_response(query, extract_city_name(query), extract_categories(query))
            
            return {"response": answer, "type": "rag"}
        else:
            
            response = get_generic_business_response(query, extract_city_name(query), extract_categories(query))
            return {"response": response, "type": "fallback", "reason": "rag_setup_failed"}
            
    except Exception as e:
        logger.error(f"Error during RAG processing: {str(e)}")
        
        
        try:
            response = get_generic_business_response(query, extract_city_name(query), extract_categories(query))
            return {"response": response, "type": "fallback", "reason": "rag_error", "error": str(e)}
        except Exception as e2:
            logger.error(f"Final fallback error: {str(e2)}")
            return {"response": "I'm sorry, I couldn't process your request. Please try again with a different query.", "error": str(e2)}




@app.route('/')
def home():
    """Home page with simple instructions"""
    return """
    <html>
        <head>
            <title>Business Recommendation API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: 
                code { background-color: 
                pre { background-color: 
                .endpoint { margin-bottom: 30px; }
            </style>
        </head>
        <body>
            <h1>🔍 Business Recommendation API</h1>
            <p>Welcome to the Business Recommendation API. Use the endpoints below to query business information.</p>
            
            <div class="endpoint">
                <h2>Query Endpoint</h2>
                <p>Send a POST request to <code>/api/query</code> with a JSON body containing your query:</p>
                <pre>
{
  "query": "Find me the best Italian restaurants in Phoenix"
}
                </pre>
                <p>Example using curl:</p>
                <pre>curl -X POST http://localhost:5000/api/query -H "Content-Type: application/json" -d '{"query": "Find me the best Italian restaurants in Phoenix"}'</pre>
            </div>
            
            <div class="endpoint">
                <h2>Simple Query Endpoint</h2>
                <p>You can also use a GET request with a query parameter:</p>
                <pre>GET /api/simple-query?q=Find+me+the+best+Italian+restaurants+in+Phoenix</pre>
                <p>Example in browser: <a href="/api/simple-query?q=Find+me+the+best+Italian+restaurants+in+Phoenix">/api/simple-query?q=Find+me+the+best+Italian+restaurants+in+Phoenix</a></p>
            </div>
        </body>
    </html>
    """

@app.route('/api/query', methods=['POST'])
def api_query():
    """Main API endpoint for processing queries"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing query parameter",
                "status": "error"
            }), 400
        
        query = data['query']
        
        if not query or not isinstance(query, str):
            return jsonify({
                "error": "Query must be a non-empty string",
                "status": "error"
            }), 400
        
        result = process_query(query)
        
        
        result["query"] = query
        result["status"] = "success"
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error",
            "response": "I'm sorry, I encountered an error processing your request."
        }), 500

@app.route('/api/simple-query', methods=['GET'])
def simple_query():
    """Simplified GET endpoint for queries"""
    try:
        query = request.args.get('q', '')
        
        if not query:
            return jsonify({
                "error": "Missing query parameter 'q'",
                "status": "error"
            }), 400
        
        result = process_query(query)
        
        
        result["query"] = query
        result["status"] = "success"
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error",
            "response": "I'm sorry, I encountered an error processing your request."
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "cache_initialized": CACHE_INITIALIZED,
        "rag_initialized": RAG_CHAIN is not None,
        "businesses_loaded": len(BUSINESS_CACHE),
        "cities_indexed": len(CITY_INDEX),
        "categories_indexed": len(CATEGORY_INDEX)
    })

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "status": "error"
    }), 404

@app.errorhandler(405)
def method_not_allowed(e):
    """Handle 405 errors"""
    return jsonify({
        "error": "Method not allowed",
        "status": "error"
    }), 405



def init_app():
    """Initialize the application data"""
    global CACHE_INITIALIZED, RAG_CHAIN
    
    yelp_file_path = "data/dataset.json"
    
    if os.path.exists(yelp_file_path):
        try:
            
            initialize_cache(yelp_file_path, limit=10000)
            
            
            def setup_rag_background():
                global RAG_CHAIN
                RAG_CHAIN = setup_rag_system(yelp_file_path)
            
            import threading
            rag_thread = threading.Thread(target=setup_rag_background)
            rag_thread.daemon = True
            rag_thread.start()
            
            logger.info("Application initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing application: {str(e)}")
    else:
        logger.warning(f"Data file not found at {yelp_file_path}. Some functionality will be limited.")



if __name__ == "__main__":
    
    init_app()
    
    
    port = int(os.environ.get("PORT", 5000))
    
    
    app.run(host="0.0.0.0", port=port, debug=False)
