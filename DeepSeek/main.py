import os
import sys
import requests
import json
import warnings
import re
from collections import defaultdict
from typing import Any, List, Optional
import time

# Suppress warnings
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

# Custom OpenRouter Chat model implementation for DeepSeek
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
            "HTTP-Referer": "https://yourapp.com",
            "X-Title": "LangChain RAG Application",
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
        
        # Add retry logic for API calls
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30  # Add timeout
                )
                response.raise_for_status()
                response_data = response.json()
                break
            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        message_content = response_data["choices"][0]["message"]["content"]
        
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=message_content))],
            llm_output=response_data
        )
    
    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"

# Cache for business data to avoid repeated file reads
BUSINESS_CACHE = {}
CITY_INDEX = defaultdict(list)
CATEGORY_INDEX = defaultdict(list)
CITY_ALIASES = {
    "phonix": "phoenix",
    "pheonix": "phoenix",
    "phenix": "phoenix",
    "phx": "phoenix",
    "vegas": "las vegas",
    "lv": "las vegas",
    "sb": "santa barbara",
    "pitt": "pittsburgh",
    "pgh": "pittsburgh",
    "char": "charlotte",
    "tuc": "tucson",
    "tucson az": "tucson",
    "phoenix az": "phoenix"
}
CATEGORY_ALIASES = {
    "resturent": "restaurant",
    "resturants": "restaurant",
    "resturant": "restaurant",
    "restraunt": "restaurant",
    "restaraunt": "restaurant",
    "eatery": "restaurant",
    "dining": "restaurant",
    "diner": "restaurant",
    "coffeeshop": "coffee",
    "coffee shop": "coffee",
    "bakeries": "bakery",
    "bake shop": "bakery",
    "pastry": "bakery",
    "pastries": "bakery"
}
CACHE_INITIALIZED = False

def initialize_cache(yelp_file_path, limit=10000):
    """Initialize cache with business data for faster lookups"""
    global BUSINESS_CACHE, CITY_INDEX, CATEGORY_INDEX, CACHE_INITIALIZED
    
    if CACHE_INITIALIZED:
        return
    
    print("Initializing business data cache...")
    
    with open(yelp_file_path, 'r', encoding='utf-8') as file:
        count = 0
        for line in file:
            try:
                business = json.loads(line.strip())
                business_id = business.get('business_id')
                
                if business_id:
                    BUSINESS_CACHE[business_id] = business
                
                # Index by city for quick lookups - use exact city name
                city = business.get('city', '').lower()
                if city:
                    CITY_INDEX[city].append(business_id)
                
                # Index by categories for quick lookups
                categories_str = business.get('categories', '')
                if categories_str:
                    categories_list = [cat.strip().lower() for cat in categories_str.split(',')]
                    for category in categories_list:
                        CATEGORY_INDEX[category].append(business_id)
                
                count += 1
                if count >= limit:
                    break
            except json.JSONDecodeError:
                continue
    
    print(f"Cache initialized with {len(BUSINESS_CACHE)} businesses")
    print(f"Cities indexed: {len(CITY_INDEX)}")
    print(f"Categories indexed: {len(CATEGORY_INDEX)}")
    
    CACHE_INITIALIZED = True

def normalize_city_name(city_name):
    """Normalize city name using aliases and spelling corrections"""
    if not city_name:
        return None
    
    city_name = city_name.lower().strip()
    
    # Check for direct aliases
    if city_name in CITY_ALIASES:
        return CITY_ALIASES[city_name]
    
    # Check for partial matches in aliases
    for alias, normalized in CITY_ALIASES.items():
        if city_name in alias or alias in city_name:
            return normalized
    
    return city_name

def normalize_category(category):
    """Normalize category name using aliases and spelling corrections"""
    if not category:
        return None
    
    category = category.lower().strip()
    
    # Check for direct aliases
    if category in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[category]
    
    # Check for partial matches in aliases
    for alias, normalized in CATEGORY_ALIASES.items():
        if category == alias or (len(category) > 4 and (category in alias or alias in category)):
            return normalized
    
    return category

def extract_city_name(query):
    """Extract city name from query using regex patterns"""
    query_lower = query.lower()
    
    # Pattern 1: "business in X city" or "businesses in X city"
    pattern1 = r"(?:business(?:es)? in|places in|recommend in|suggest in) ([a-zA-Z\s]+)(?:city)?"
    match = re.search(pattern1, query_lower)
    if match:
        return normalize_city_name(match.group(1).strip())
    
    # Pattern 2: "in X city"
    pattern2 = r"in ([a-zA-Z\s]+)(?:city)?"
    match = re.search(pattern2, query_lower)
    if match:
        return normalize_city_name(match.group(1).strip())
    
    # Pattern 3: "X city"
    pattern3 = r"([a-zA-Z]+) city"
    match = re.search(pattern3, query_lower)
    if match:
        return normalize_city_name(match.group(1).strip())
    
    # Direct city mention
    cities = ["tucson", "phoenix", "las vegas", "santa barbara", "toronto", "pittsburgh", "charlotte"]
    for city in cities:
        if city in query_lower:
            return city
    
    # Check for city aliases
    for alias, city in CITY_ALIASES.items():
        if alias in query_lower:
            return city
    
    return None

def extract_categories(query):
    """Extract multiple categories from query"""
    query_lower = query.lower()
    
    # Common categories to look for - moved to a set for O(1) lookups
    common_categories = {
        "restaurant", "cafe", "coffee", "pizza", "italian", "chinese", "mexican", 
        "japanese", "sushi", "thai", "indian", "bar", "pub", "bakery", "breakfast",
        "brunch", "lunch", "dinner", "fast food", "dessert", "ice cream", "salon",
        "spa", "gym", "fitness", "yoga", "hotel", "motel", "shopping", "retail",
        "clothing", "fashion", "grocery", "market", "pharmacy", "doctor", "dentist", "hospital",
        "automotive", "car repair", "gas station", "bank", "atm", "school", "university",
        "library", "bookstore", "movie theater", "theater", "museum", "art gallery",
        "park", "playground", "zoo", "aquarium", "nightlife", "club", "dance"
    }
    
    # Add category aliases
    for alias in CATEGORY_ALIASES:
        common_categories.add(alias)
    
    # Find all categories mentioned in the query
    found_categories = []
    for category in common_categories:
        if category in query_lower:
            normalized = normalize_category(category)
            if normalized and normalized not in found_categories:
                found_categories.append(normalized)
    
    # Pattern for "recommend a {category} and {category} in {city}"
    pattern = r"(?:recommend|suggest|find|looking for|want) (?:a|an|some) ([a-zA-Z\s]+) (?:and|&) ([a-zA-Z\s]+) in"
    match = re.search(pattern, query_lower)
    if match:
        cat1 = normalize_category(match.group(1).strip())
        cat2 = normalize_category(match.group(2).strip())
        if cat1 and cat1 not in found_categories:
            found_categories.append(cat1)
        if cat2 and cat2 not in found_categories:
            found_categories.append(cat2)
    
    # If we found categories through pattern matching or direct mention
    return found_categories if found_categories else None

def is_top_rated_query(query):
    """Check if the query is asking for top rated businesses"""
    query_lower = query.lower()
    # Use a set for O(1) lookups
    top_rated_keywords = {
        "top rated", "highest rated", "best rated", "top star", "highest star", 
        "5 star", "five star", "top businesses", "best businesses", "top 5", 
        "top five", "most rated", "highest rating", "best rating", "highly rated",
        "highly-rated", "best", "great", "excellent", "outstanding", "recommend",
        "good", "nice", "popular", "famous", "well-known", "well known"
    }
    
    return any(keyword in query_lower for keyword in top_rated_keywords)

def get_safe_string(value, default=""):
    """Safely get a string value, handling None values"""
    if value is None:
        return default
    return str(value).lower()

def generate_human_recommendation(businesses, category, city):
    """Generate a human-like recommendation for businesses"""
    if not businesses:
        return f"I couldn't find any {category} places in {city}. Perhaps try another category or city?"
    
    # Sort by rating and review count
    sorted_businesses = sorted(businesses, key=lambda x: (x.get("stars", 0), x.get("review_count", 0)), reverse=True)
    top_businesses = sorted_businesses[:3]  # Get top 3 for more natural recommendations
    
    # Check if we're dealing with multiple categories
    multiple_categories = " and " in category
    
    # Generate a conversational response
    if multiple_categories:
        response = f"If you're looking for places that offer both {category} in {city}, I'd recommend checking out "
    else:
        response = f"If you're looking for {category} in {city}, I'd recommend checking out "
    
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
    
    else:  # 3 or more
        response += f"{top_businesses[0]['name']}, {top_businesses[1]['name']}, or {top_businesses[2]['name']}. "
        response += f"{top_businesses[0]['name']} is the highest rated at {top_businesses[0].get('stars', 'N/A')} stars. "
        response += f"All three are well-reviewed and popular choices for {category} in {city}."
    
    return response

def find_businesses_with_all_categories(businesses, categories):
    """Find businesses that have ALL the specified categories"""
    matching_businesses = []
    
    for business in businesses:
        business_categories = get_safe_string(business.get('categories', ''))
        # Check if all requested categories are in this business's categories
        if all(category in business_categories for category in categories):
            matching_businesses.append(business)
    
    return matching_businesses

def get_businesses_by_city(city_name):
    """Get businesses in a specific city"""
    global BUSINESS_CACHE, CITY_INDEX
    
    matching_businesses = []
    if not city_name:
        return matching_businesses
    
    city_name = city_name.lower()
    
    # Try exact match first
    if city_name in CITY_INDEX:
        business_ids = CITY_INDEX[city_name]
        matching_businesses = [BUSINESS_CACHE[bid] for bid in business_ids if bid in BUSINESS_CACHE]
    else:
        # Try partial match
        for city in CITY_INDEX:
            if city_name in city or city in city_name:
                business_ids = CITY_INDEX[city]
                city_businesses = [BUSINESS_CACHE[bid] for bid in business_ids if bid in BUSINESS_CACHE]
                matching_businesses.extend(city_businesses)
    
    return matching_businesses

def get_businesses_by_category(category, city_businesses=None):
    """Get businesses in a specific category, optionally filtered by city"""
    global BUSINESS_CACHE, CATEGORY_INDEX
    
    if city_businesses is None:
        # If no city filter, get all businesses in this category
        business_ids = CATEGORY_INDEX.get(category.lower(), [])
        return [BUSINESS_CACHE[bid] for bid in business_ids if bid in BUSINESS_CACHE]
    else:
        # Filter city businesses by category
        return [b for b in city_businesses if category.lower() in get_safe_string(b.get('categories', ''))]

def is_business_related_query(query):
    """Determine if a query is related to business recommendations"""
    query_lower = query.lower()
    
    # Business-related keywords
    business_keywords = [
        "restaurant", "food", "eat", "dining", "cafe", "coffee", "shop", "store", 
        "business", "place", "recommend", "suggestion", "best", "top", "rated", 
        "review", "star", "yelp", "location", "address", "open", "hour", "price",
        "expensive", "cheap", "affordable", "menu", "reservation", "booking",
        "takeout", "delivery", "dine-in", "cuisine", "bar", "pub", "nightlife",
        "entertainment", "service", "mall", "shopping", "retail", "salon", "spa",
        "gym", "fitness", "hotel", "motel", "lodging", "stay", "accommodation"
    ]
    
    # Add all category aliases as business keywords
    for alias in CATEGORY_ALIASES:
        business_keywords.append(alias)
    
    # City names that might indicate a business query
    city_names = [
        "tucson", "phoenix", "las vegas", "santa barbara", "toronto", "pittsburgh", 
        "charlotte", "new york", "los angeles", "chicago", "houston", "philadelphia",
        "san antonio", "san diego", "dallas", "san jose", "austin", "jacksonville",
        "fort worth", "columbus", "san francisco", "seattle", "denver", "boston"
    ]
    
    # Add all city aliases
    for alias in CITY_ALIASES:
        city_names.append(alias)
    
    # Check for business keywords
    if any(keyword in query_lower for keyword in business_keywords):
        return True
    
    # Check for city names
    if any(city in query_lower for city in city_names):
        return True
    
    # Check for common business query patterns
    business_patterns = [
        r"where (can|should) i",
        r"looking for",
        r"recommend",
        r"suggest",
        r"find me",
        r"what('s| is) the best",
        r"top \d+",
        r"highest rated",
        r"near me",
        r"in the area"
    ]
    
    if any(re.search(pattern, query_lower) for pattern in business_patterns):
        return True
    
    return False

def ask_deepseek(query, api_key):
    """Send a query directly to DeepSeek via OpenRouter API"""
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
    
    # Add retry logic
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
                return f"Sorry, I encountered an error while trying to answer your question: {str(e)}"
            time.sleep(2 ** attempt)  # Exponential backoff

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
    
    # Extract city name if present
    city_name = extract_city_name(query)
    
    # Extract multiple categories if present
    categories = extract_categories(query)
    
    # Debug info
    print(f"Query: {query}")
    print(f"Detected city: {city_name}")
    print(f"Detected categories: {categories}")
    print(f"Is top rated query: {is_top_rated}")
    
    # Handle category recommendation in a specific city
    if categories and city_name:
        # Get businesses in the specified city
        city_businesses = get_businesses_by_city(city_name)
        
        if not city_businesses:
            # Instead of returning a negative response, use DeepSeek to generate a helpful response
            return get_generic_business_response(query, city_name, categories)
        
        # Find businesses that match ALL categories
        matching_businesses = find_businesses_with_all_categories(city_businesses, categories)
        
        if matching_businesses:
            categories_text = " and ".join(categories)
            return generate_human_recommendation(matching_businesses, categories_text, city_name.title())
        else:
            # Try to find businesses that match ANY of the categories
            any_category_businesses = []
            for category in categories:
                category_businesses = get_businesses_by_category(category, city_businesses)
                any_category_businesses.extend(category_businesses)
            
            if any_category_businesses:
                categories_text = " or ".join(categories)
                return f"I couldn't find places that offer both {' and '.join(categories)} together in {city_name.title()}, but I found some that offer either {categories_text}:\n\n" + \
                       generate_human_recommendation(any_category_businesses, categories_text, city_name.title())
            else:
                # Use DeepSeek for a helpful response
                return get_generic_business_response(query, city_name, categories)
    
    # Handle top rated businesses (with or without city)
    elif is_top_rated:
        # Get businesses by city if specified
        if city_name:
            businesses = get_businesses_by_city(city_name)
            if not businesses:
                # Use DeepSeek for a helpful response
                return get_generic_business_response(query, city_name, categories)
        else:
            # If no city, use all businesses (limited sample)
            businesses = list(BUSINESS_CACHE.values())[:5000]  # Limit for performance
        
        # Filter by categories if specified
        if categories:
            filtered_businesses = []
            for category in categories:
                category_businesses = get_businesses_by_category(category, businesses)
                filtered_businesses.extend(category_businesses)
            businesses = filtered_businesses
        
        # Filter for businesses with ratings
        businesses = [b for b in businesses if "stars" in b and b["stars"] is not None]
        
        if businesses:
            # Sort by rating (stars) and review count
            sorted_businesses = sorted(
                businesses, 
                key=lambda x: (x.get("stars", 0), x.get("review_count", 0)), 
                reverse=True
            )
            
            # Take top 5 businesses
            top_businesses = sorted_businesses[:5]
            
            location_text = f" in {city_name.title()}" if city_name else ""
            category_text = f" {' and '.join(categories)}" if categories else ""
            result = f"Top rated{category_text} businesses{location_text}:\n\n"
            for i, business in enumerate(top_businesses, 1):
                result += f"{i}. {business['name']} - {business.get('stars', 'N/A')} stars\n"
                result += f"   Categories: {business.get('categories', 'N/A')}\n"
                result += f"   Address: {business.get('address', 'N/A')}, {business.get('city', 'N/A')}\n\n"
            
            return result
        else:
            # Use DeepSeek for a helpful response
            return get_generic_business_response(query, city_name, categories)
    
    # Handle city-based business query (if not already handled by top rated)
    elif city_name:
        businesses_in_city = get_businesses_by_city(city_name)
        
        if businesses_in_city:
            # Sort by rating and review count
            sorted_businesses = sorted(
                businesses_in_city, 
                key=lambda x: (x.get("stars", 0), x.get("review_count", 0)), 
                reverse=True
            )
            
            # Take top 10 businesses
            top_businesses = sorted_businesses[:10]
            
            result = f"Top businesses in {city_name.title()}:\n\n"
            for i, business in enumerate(top_businesses, 1):
                result += f"{i}. {business['name']} - {business.get('stars', 'N/A')} stars\n"
                result += f"   Categories: {business.get('categories', 'N/A')}\n"
                result += f"   Address: {business.get('address', 'N/A')}\n\n"
            
            return result
        else:
            # Use DeepSeek for a helpful response
            return get_generic_business_response(query, city_name, None)
    
    # If no direct match, return None to indicate we should use RAG
    return None

def setup_rag_system(yelp_file_path):
    """Set up the RAG system with vectorstore and retrieval chain"""
    # Use HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Process the Yelp JSON file (limited to 5000 records for speed)
    documents = []
    
    # Use cached data instead of reading the file again
    count = 0
    for business_id, business in BUSINESS_CACHE.items():
        # Create a document with relevant business info
        content = f"Business Name: {business.get('name', '')}\n"
        content += f"Address: {business.get('address', '')}, {business.get('city', '')}, {business.get('state', '')} {business.get('postal_code', '')}\n"
        content += f"Categories: {business.get('categories', '')}\n"
        content += f"Rating: {business.get('stars', 'N/A')} stars based on {business.get('review_count', 'N/A')} reviews\n"
        
        doc = Document(
            page_content=content,
            metadata={
                "source": yelp_file_path, 
                "city": business.get("city", ""),
                "state": business.get("state", ""),
                "stars": business.get("stars", 0),
                "categories": get_safe_string(business.get("categories"))
            }
        )
        documents.append(doc)
        
        count += 1
        if count >= 5000:  # Limit for performance
            break
    
    # Create vectorstore
    vectorstore = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings
    )
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    
    # Initialize DeepSeek chat model via OpenRouter
    llm = DeepSeekChat(
        openrouter_api_key=OPENROUTER_API_KEY,
        model_name="deepseek/deepseek-r1:free",
        temperature=0.7
    )
    
    # Create a system message that emphasizes formatting
    system_message = """
    You are a helpful assistant that provides information about businesses from the Yelp dataset.
    When responding about businesses, please format your response in a clear, numbered list with key details.
    Include the business name, rating, categories, and address for each business.
    """
    
    # Create the chain with the system message
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 5}),
    )
    
    return chain

def main():
    # Your OpenRouter API key
    global OPENROUTER_API_KEY
    OPENROUTER_API_KEY = "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"
    
    # Get query from command line
    query = None
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("Please provide a query as a command line argument.")
        sys.exit(1)
    
    # Check if the query is business-related
    if not is_business_related_query(query):
        print("This query appears to be a general question, not business-related.")
        # Use DeepSeek directly for non-business queries
        response = ask_deepseek(query, OPENROUTER_API_KEY)
        print(response)
        sys.exit(0)
    
    # File path to Yelp dataset
    yelp_file_path = "data/dataset.json"
    
    # Check if file exists
    if not os.path.exists(yelp_file_path):
        print(f"Error: File not found: {yelp_file_path}")
        # Fallback to DeepSeek for general response
        response = ask_deepseek(query, OPENROUTER_API_KEY)
        print(response)
        sys.exit(0)
    
    # Initialize cache
    initialize_cache(yelp_file_path)
    
    # Try direct query processing first
    direct_result = process_direct_query(query)
    if direct_result:
        print(direct_result)
        sys.exit(0)
    
    # If no direct match, use RAG approach
    print("No direct match found, using RAG approach...")
    
    # Set up RAG system
    chain = setup_rag_system(yelp_file_path)
    
    # Process query with RAG
    try:
        chat_history = []
        result = chain.invoke({"question": query, "chat_history": chat_history})
        answer = result["answer"]
        
        # If the answer is negative or doesn't provide useful information, use DeepSeek directly
        negative_phrases = ["I don't have", "I couldn't find", "no information", "no data", "not available", 
                           "no businesses", "no restaurants", "no cafes", "no bakeries"]
        
        if any(phrase in answer.lower() for phrase in negative_phrases):
            print("RAG response was not helpful, falling back to DeepSeek...")
            answer = get_generic_business_response(query, extract_city_name(query), extract_categories(query))
        
        print(answer)
    except Exception as e:
        print(f"Error during RAG processing: {str(e)}")
        # Fallback to DeepSeek
        response = get_generic_business_response(query, extract_city_name(query), extract_categories(query))
        print(response)

if __name__ == "__main__":
    main()
