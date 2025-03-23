import os
import requests
from bs4 import BeautifulSoup
import re
import time
from typing import List, Dict, Tuple, Optional, Any, Set
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from langchain_community.vectorstores import FAISS
import streamlit as st

load_dotenv()

# OpenAI credentials from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RETRIEVAL_ASSISTANT_ID = os.getenv("RETRIEVAL_ASSISTANT_ID")
ANALYSIS_ASSISTANT_ID = os.getenv("ANALYSIS_ASSISTANT_ID")   

class ConfluenceChatbot:
    def __init__(self, confluence_url: str, confluence_username: str, confluence_api_token: str):
        self.confluence_url = confluence_url
        self.auth = (confluence_username, confluence_api_token)
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=OPENAI_API_KEY
        )
        self.vector_store = None 
        self.chat_history = []
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.thread_id = None
        self.last_api_call = 0
        self.rate_limit_delay = 0.5
        self.document_metadata = {}
    
    # ----- Rate Limiting Helper -----
    
    def rate_limit(self):
        """Simple rate limiting to avoid hitting API limits"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_call
            time.sleep(sleep_time)
            
        self.last_api_call = time.time()
    
    # ----- Confluence API and Content Processing -----
    
    def fetch_space_pages(self, space_key: str) -> List[Dict]:
        """Fetch all pages from a Confluence space with improved handling"""
        url = f"{self.confluence_url}/rest/api/content"
        params = {
            "spaceKey": space_key,
            "expand": "body.storage,children.page",
            "limit": 100
        }
        
        all_pages = []
        start = 0
        
        while True:
            params["start"] = start
            
            # Rate limit API calls
            self.rate_limit()
            
            # Add retry logic for API resilience
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    response = requests.get(
                        url, 
                        auth=self.auth, 
                        headers=self.headers, 
                        params=params,
                        timeout=30  # Add timeout
                    )
                    
                    response.raise_for_status()
                    break
                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"Error fetching pages after {max_retries} attempts: {str(e)}")
                        return all_pages
                    print(f"Request error (attempt {retry_count}): {str(e)}. Retrying...")
                    time.sleep(2)
            
            data = response.json()
            results = data.get("results", [])
            
            if not results:
                break
                
            all_pages.extend(results)
            
            child_page_ids = []
            for page in results:
                if "children" in page and "page" in page["children"] and "results" in page["children"]["page"]:
                    for child in page["children"]["page"]["results"]:
                        child_page_ids.append(child["id"])
            
            # Fetch child pages in batch to reduce API calls
            if child_page_ids:
                for child_id in child_page_ids:
                    child_page = self.fetch_single_page(child_id)
                    if child_page:
                        all_pages.extend(child_page)
            
            if len(results) < params["limit"]:
                break
                
            start += params["limit"]
        
        print(f"Fetched a total of {len(all_pages)} pages from space {space_key}")
        return all_pages
        
    def fetch_single_page(self, page_id: str) -> Optional[List[Dict]]:
        """Fetch a single page from Confluence by ID with improved error handling"""
        url = f"{self.confluence_url}/rest/api/content/{page_id}"
        params = {
            "expand": "body.storage,children.page"
        }
        
        self.rate_limit()
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = requests.get(
                    url, 
                    auth=self.auth, 
                    headers=self.headers, 
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                
                result = [response.json()]
                
                # Get child pages with rate limiting
                page = response.json()
                if "children" in page and "page" in page["children"] and "results" in page["children"]["page"]:
                    child_pages = page["children"]["page"]["results"]
                    for child in child_pages:
                        # Rate limit between child page fetches
                        self.rate_limit()
                        child_result = self.fetch_single_page(child["id"])
                        if child_result:
                            result.extend(child_result)
                            
                return result
                
            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Error fetching page {page_id} after {max_retries} attempts: {str(e)}")
                    return None
                print(f"Request error (attempt {retry_count}): {str(e)}. Retrying...")
                time.sleep(2)
            
        return None
    
    def clean_html_content(self, html_content: str) -> str:
        """Clean HTML content from Confluence pages"""
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Handle special Confluence elements
        for macro in soup.find_all(class_=re.compile(r'macro')):
            macro.extract()
            
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def process_pages(self, pages: List[Dict]) -> List[Dict]:
        """Process and index Confluence pages"""
        documents = []
        
        for page in pages:
            title = page.get("title", "")
            body = page.get("body", {}).get("storage", {}).get("value", "")
            page_id = page.get("id", "")
            page_url = f"{self.confluence_url}/pages/viewpage.action?pageId={page_id}"
            
            # Clean HTML content
            text_content = self.clean_html_content(body)
            
            # Add metadata
            document = {
                "title": title,
                "content": text_content,
                "url": page_url,
                "id": page_id
            }
            
            # Store document metadata for quick reference
            self.document_metadata[page_id] = {
                "title": title,
                "url": page_url
            }
            
            documents.append(document)
        
        return documents
    
    # ----- Vector Store and Index Management -----
    
    def create_vector_store(self, documents: List[Dict]) -> Optional[FAISS]:
        """Create an in-memory FAISS vector store with optimized chunking for better retrieval"""
        # Use fewer, larger chunks to reduce embedding API calls
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,         
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        texts = []
        metadatas = []
        
        for doc in documents:
            # Add the title to the content for better context
            title_and_content = f"Title: {doc['title']}\n\n{doc['content']}"
            
            # Split into chunks
            chunks = text_splitter.split_text(title_and_content)
            
            # Create chunks
            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                metadatas.append({
                    "title": doc["title"],
                    "url": doc["url"],
                    "id": doc["id"],
                    "chunk_id": f"{doc['id']}-{i}",
                    "is_title_chunk": i == 0
                })
        
        # Create vector store if we have documents - but batch the embeddings to avoid rate limits
        if texts:
            batch_size = 50
            
            if self.vector_store:
                # Process in batches
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_metadatas = metadatas[i:i+batch_size]
                    self.vector_store.add_texts(batch_texts, batch_metadatas)
                    time.sleep(1)
            else:
                # Create new vector store with first batch
                first_batch_texts = texts[:batch_size]
                first_batch_metadatas = metadatas[:batch_size]
                
                self.vector_store = FAISS.from_texts(
                    texts=first_batch_texts,
                    embedding=self.embeddings,
                    metadatas=first_batch_metadatas
                )
                
                # Add remaining batches
                for i in range(batch_size, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_metadatas = metadatas[i:i+batch_size]
                    self.vector_store.add_texts(batch_texts, batch_metadatas)
                    time.sleep(1)
            
        return self.vector_store
    
    def clear_index(self):
        """Clear the in-memory vector store"""
        self.vector_store = None
        self.document_metadata = {}
        print("Vector index cleared - all data was in memory only")
        
    def index_space(self, space_key: str) -> int:
        """Index a Confluence space"""
        try:
            pages = self.fetch_space_pages(space_key)
            documents = self.process_pages(pages)
            self.create_vector_store(documents)
            return len(documents)
        except Exception as e:
            print(f"Error indexing space: {str(e)}")
            raise
        
    def index_single_page(self, page_id: str) -> int:
        """Index a single Confluence page"""
        try:
            page = self.fetch_single_page(page_id)
            if not page:
                return 0
            documents = self.process_pages(page)
            self.create_vector_store(documents)
            return len(documents)
        except Exception as e:
            print(f"Error indexing page: {str(e)}")
            return 0
    
    # ----- Assistant Thread Management -----
    
    def initialize_assistant_thread(self):
        """Initialize the OpenAI Assistant thread for the main conversation"""
        if not self.thread_id:
            thread = self.client.beta.threads.create()
            self.thread_id = thread.id
    
    # ----- Document Search and Retrieval -----
    
    def extract_term_context(self, content: str, term: str, context_chars: int = 100) -> str:
        """Extract context around a term mention"""
        term_lower = term.lower()
        content_lower = content.lower()
        
        # Find the position of the term
        position = content_lower.find(term_lower)
        
        if position == -1:
            return ""
        
        # Calculate the context window
        start = max(0, position - context_chars)
        end = min(len(content), position + len(term) + context_chars)
        
        # Extract context and add ellipsis if needed
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(content) else ""
        
        context = prefix + content[start:end].strip() + suffix
        
        return context
    
    def count_mentions(self, term: str) -> Tuple[int, List[Dict]]:
        """
        Count mentions of a specific term across all indexed documents
        Returns count and detailed information about each unique document
        """
        if not self.vector_store:
            return 0, []
        
        # Search for documents containing the term
        docs = self.vector_store.similarity_search(
            term,
            k=100
        )
        
        # Track unique pages mentioning the term (case-insensitive)
        term = term.lower()
        unique_documents = {}
        
        for doc in docs:
            page_id = doc.metadata['id']
            # Check if term exists in the content and is not already added
            if term in doc.page_content.lower() and page_id not in unique_documents:
                unique_documents[page_id] = {
                    'id': page_id,
                    'title': doc.metadata['title'],
                    'url': doc.metadata['url']
                }
        
        # Convert to list and sort by title
        document_list = list(unique_documents.values())
        document_list.sort(key=lambda x: x['title'])
        
        return len(document_list), document_list
    
    def detect_document_title_in_question(self, question: str) -> Optional[str]:
        """Detect if the user is asking about a specific document by title"""
        # Look for document title patterns
        title_patterns = [
            r'in\s+["\']?([^"\']+?)["\']?\s+(document|page|confluence|guide|setup|overview)',
            r'from\s+["\']?([^"\']+?)["\']?\s+(document|page|confluence|guide|setup|overview)',
            r'(document|page|confluence|guide|setup|overview)\s+["\']?([^"\']+?)["\']?',
            r'["\']([^"\']+?)["\']',
        ]
        
        # Check each pattern
        for pattern in title_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        # For patterns with capture groups
                        potential_title = match[0] if len(match[0]) > len(match[1]) else match[1]
                    else:
                        potential_title = match
                    
                    # Validate this is a real document title
                    if self.is_valid_document_title(potential_title):
                        return potential_title
        
        return None

    def is_valid_document_title(self, potential_title: str) -> bool:
        """Check if a potential title exists in our document index"""
        if not self.vector_store:
            return False
        
        # Get some documents to check titles - with reduced number
        sample_docs = self.vector_store.similarity_search(
            "document information",
            k=100
        )
        
        # Look for exact or partial matches
        for doc in sample_docs:
            if doc.metadata['title'].lower() == potential_title.lower():
                return True
            if len(potential_title) > 10 and potential_title.lower() in doc.metadata['title'].lower():
                return True
        
        return False

    def retrieve_by_exact_title(self, title: str, question: str) -> Tuple[List, str]:
        """Specifically retrieve content from a document with an exact title match"""
        if not self.vector_store:
            return [], "Vector store not initialized"
        
        # Get documents from the vector store - reduced number
        docs = self.vector_store.similarity_search(
            title,
            k=100
        )
        
        # Filter for title match
        title_matched_docs = []
        for doc in docs:
            if doc.metadata['title'].lower() == title.lower() or title.lower() in doc.metadata['title'].lower():
                title_matched_docs.append(doc)
        
        if not title_matched_docs:
            return [], f"No document with title '{title}' found in the index"
        
        # Sort by relevance to question
        title_matched_docs.sort(key=lambda doc: 
            doc.page_content.lower().count(question.lower()), 
            reverse=True
        )
        
        # Limit to top 10 chunks to reduce token usage
        return title_matched_docs[:10], f"Found {len(title_matched_docs[:10])} chunks from document '{title}'"
    
    # ----- Dual-Assistant Question Answering Implementation -----
    
    def wait_for_run_completion(self, thread_id, run_id, max_wait_sec=60):
        """Wait for assistant run to complete with proper rate limiting and timeout"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_sec:
            # Rate limit API calls
            self.rate_limit()
            
            run_status = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            
            if run_status.status == 'completed':
                return True
            elif run_status.status in ['failed', 'cancelled', 'expired']:
                return False
                
            # Wait longer between status checks
            time.sleep(2)
        
        return False  # Timeout
    
    def ask_specific_document(self, title: str, question: str) -> str:
        """Ask a question about a specific document by title using dual-assistant approach"""
        if not self.vector_store:
            return "Please index Confluence pages first."
        
        # Get content from the specific document title
        document_chunks, status_message = self.retrieve_by_exact_title(title, question)
        
        if not document_chunks:
            return f"Could not find document with title '{title}'. {status_message}"
        
        # STAGE 1: RETRIEVAL ASSISTANT
        # Create a new thread for retrieval
        retrieval_thread = self.client.beta.threads.create()
        
        # Prepare context from document chunks
        document_context = ""
        for i, chunk in enumerate(document_chunks, 1):
            document_context += f"CHUNK {i}:\nTitle: {chunk.metadata['title']}\nContent: {chunk.page_content}\n\n"
        
        # Create prompt for the retrieval assistant
        retrieval_prompt = f"""
        The user is asking about a specific document titled "{title}":
        
        Question: {question}
        
        Here are relevant sections from this document:
        
        {document_context}
        
        Extract ALL relevant information that helps answer this specific question. Maintain original formatting and include ALL steps if this is about a workflow or process. Be thorough and comprehensive in your extraction.
        """
        
        # Rate limit API calls
        self.rate_limit()
        
        # Create message in the retrieval thread
        self.client.beta.threads.messages.create(
            thread_id=retrieval_thread.id,
            role="user",
            content=retrieval_prompt
        )
        
        # Run the retrieval assistant
        retrieval_run = self.client.beta.threads.runs.create(
            thread_id=retrieval_thread.id,
            assistant_id=RETRIEVAL_ASSISTANT_ID
        )
        
        # Wait for completion with timeout
        if not self.wait_for_run_completion(retrieval_thread.id, retrieval_run.id, max_wait_sec=120):
            # Clean up thread
            try:
                self.client.beta.threads.delete(thread_id=retrieval_thread.id)
            except Exception:
                pass
            return "The retrieval process took too long or encountered an error. Please try again with a more specific question."
        
        # Get the retrieval results
        retrieval_messages = self.client.beta.threads.messages.list(
            thread_id=retrieval_thread.id
        )
        
        # Extract the retrieved information
        retrieved_info = ""
        for message in retrieval_messages.data:
            if message.role == "assistant":
                retrieved_info = message.content[0].text.value
                break
        
        # Clean up the retrieval thread
        try:
            self.client.beta.threads.delete(thread_id=retrieval_thread.id)
        except Exception:
            pass
        
        # STAGE 2: ANALYSIS ASSISTANT
        # Initialize main thread if not already done
        self.initialize_assistant_thread()
        
        # Create synthesis prompt for the analysis assistant
        synthesis_prompt = f"""
        The user is asking about the document titled "{title}":
        
        Question: {question}
        
        Here is the relevant information extracted from this document:
        
        {retrieved_info}
        
        Please provide a well-structured answer based solely on this information. Follow the formatting guidelines especially for workflows/processes if applicable.
        """
        
        # Rate limit API calls
        self.rate_limit()
        
        # Create message in the main thread
        self.client.beta.threads.messages.create(
            thread_id=self.thread_id,
            role="user",
            content=synthesis_prompt
        )
        
        # Run the analysis assistant
        synthesis_run = self.client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=ANALYSIS_ASSISTANT_ID
        )
        
        # Wait for completion with timeout
        if not self.wait_for_run_completion(self.thread_id, synthesis_run.id, max_wait_sec=120):
            return "The answer synthesis took too long or encountered an error. Please try again with a more specific question."
        
        # Retrieve the final answer
        messages = self.client.beta.threads.messages.list(
            thread_id=self.thread_id
        )
        
        # Get the last assistant message
        for message in messages.data:
            if message.role == "assistant":
                answer = message.content[0].text.value
                return answer
        
        return "No response received from the assistant."
    
    def ask(self, question: str) -> str:
        """Dual-assistant approach for general questions"""
        if not self.vector_store:
            return "Please index Confluence pages first."
        
        # Initialize thread if not already done
        self.initialize_assistant_thread()
        
        # First, check if the user is asking about a specific document
        specific_document = self.detect_document_title_in_question(question)
        
        # If asking about a specific document, use document-specific retrieval
        if specific_document:
            return self.ask_specific_document(specific_document, question)
        
        # Check if this is a counting query
        is_count_query = any([
            re.search(r"how many (documents|pages|content|articles) (mention|reference|refer to)", question.lower()),
            re.search(r"how many (times|mentions|occurrences) of", question.lower()),
            re.search(r"count (of|the) (mentions|occurrences)", question.lower()),
            re.search(r"number of (documents|pages) (with|containing|mentioning)", question.lower())
        ])
        
        # If it's a count query, handle it differently
        if is_count_query:
            # Try to extract the term to count
            term_patterns = [
                r"mention[s]? ['\"](.+?)['\"]",
                r"mention[s]? (.+?)[\.\?]?$",
                r"refer to ['\"](.+?)['\"]",
                r"refer to (.+?)[\.\?]?$",
                r"reference ['\"](.+?)['\"]",
                r"reference (.+?)[\.\?]?$",
                r"occurrences of ['\"](.+?)['\"]",
                r"occurrences of (.+?)[\.\?]?$",
                r"containing ['\"](.+?)['\"]",
                r"containing (.+?)[\.\?]?$",
                r"mentioning ['\"](.+?)['\"]",
                r"mentioning (.+?)[\.\?]?$"
            ]
            
            extracted_term = None
            
            for pattern in term_patterns:
                match = re.search(pattern, question.lower())
                if match:
                    extracted_term = match.group(1).strip().strip("'\"")
                    break
            
            # If no pattern matched, try to extract from quotes
            if not extracted_term:
                quoted_match = re.search(r"['\"](.+?)['\"]", question)
                if quoted_match:
                    extracted_term = quoted_match.group(1).strip()
                else:
                    # Last resort: look at last few words
                    words = question.split()
                    if len(words) > 2:
                        potential_terms = [word.strip(".,?!;") for word in words[-3:]]
                        # Filter out common words
                        common_words = ["documents", "pages", "mentions", "many", "how", "the", "of"]
                        candidates = [term for term in potential_terms if term.lower() not in common_words]
                        if candidates:
                            extracted_term = max(candidates, key=len)
            
            if extracted_term:
                return self.handle_count_query(extracted_term)
        
        # STAGE 1: RETRIEVAL ASSISTANT
        # This will be used for general questions
        
        # Create a new thread for retrieval
        retrieval_thread = self.client.beta.threads.create()
        
        # Get relevant documents for the question - use higher k for more comprehensive retrieval
        docs = self.vector_store.similarity_search(
            question,
            k=30  # Increased to get broader coverage
        )
        
        # Prepare context for retrieval
        context = ""
        for i, doc in enumerate(docs, 1):
            context += f"DOCUMENT {i}:\nTitle: {doc.metadata['title']}\nURL: {doc.metadata['url']}\nContent: {doc.page_content}\n\n"
        
        # Create prompt for the retrieval assistant
        retrieval_prompt = f"""
        I need to answer this question from our Confluence knowledge base:
        
        Question: {question}
        
        Here are potentially relevant documents:
        
        {context}
        
        Extract ALL information that helps answer this question. Be thorough and comprehensive. Look for patterns and connections across documents. Focus on breadth of coverage rather than depth from a single document.
        """
        
        # Rate limit API calls
        self.rate_limit()
        
        # Create message in the retrieval thread
        self.client.beta.threads.messages.create(
            thread_id=retrieval_thread.id,
            role="user",
            content=retrieval_prompt
        )
        
        # Run the retrieval assistant
        retrieval_run = self.client.beta.threads.runs.create(
            thread_id=retrieval_thread.id,
            assistant_id=RETRIEVAL_ASSISTANT_ID
        )
        
        # Wait for completion with timeout
        if not self.wait_for_run_completion(retrieval_thread.id, retrieval_run.id, max_wait_sec=120):
            # Clean up thread
            try:
                self.client.beta.threads.delete(thread_id=retrieval_thread.id)
            except Exception:
                pass
            return "The retrieval process took too long or encountered an error. Please try again with a more specific question."
        
        # Get the retrieval results
        retrieval_messages = self.client.beta.threads.messages.list(
            thread_id=retrieval_thread.id
        )
        
        # Extract the retrieved information
        retrieved_info = ""
        for message in retrieval_messages.data:
            if message.role == "assistant":
                retrieved_info = message.content[0].text.value
                break
        
        # Clean up the retrieval thread
        try:
            self.client.beta.threads.delete(thread_id=retrieval_thread.id)
        except Exception:
            pass
        
        # STAGE 2: ANALYSIS ASSISTANT
        
        # Create synthesis prompt for the analysis assistant
        synthesis_prompt = f"""
        Question: {question}
        
        Here is the information extracted from our Confluence knowledge base:
        
        {retrieved_info}
        
        Please provide a well-structured answer based on this information. Follow the formatting guidelines especially for workflows/processes if applicable.
        """
        
        # Rate limit API calls
        self.rate_limit()
        
        # Create message in the main thread
        self.client.beta.threads.messages.create(
            thread_id=self.thread_id,
            role="user",
            content=synthesis_prompt
        )
        
        # Run the analysis assistant
        synthesis_run = self.client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=ANALYSIS_ASSISTANT_ID
        )
        
        # Wait for completion with timeout
        if not self.wait_for_run_completion(self.thread_id, synthesis_run.id, max_wait_sec=120):
            return "The answer synthesis took too long or encountered an error. Please try again with a more specific question."
        
        # Retrieve the final answer
        messages = self.client.beta.threads.messages.list(
            thread_id=self.thread_id
        )
        
        # Get the last assistant message
        for message in messages.data:
            if message.role == "assistant":
                answer = message.content[0].text.value
                return answer
        
        return "No response received from the assistant."
    
    def handle_count_query(self, term: str) -> str:
        """
        Specialized handler for count-type queries using dual-assistant approach.
        First uses direct vector search to find mentions, then passes to assistants for formatting.
        """
        # Get count and details directly
        count, page_details = self.count_mentions(term)
        
        # If no results, return directly
        if count == 0:
            return f"No pages were found that mention '{term}'."
        
        # Create context for retrieval assistant
        context = f"The user is asking how many documents mention '{term}'.\n\n"
        context += f"I found {count} Confluence pages that mention '{term}':\n\n"
        
        for i, page in enumerate(page_details, 1):
            context += f"DOCUMENT {i}:\nTitle: {page['title']}\nURL: {page['url']}\n\n"
        
        # STAGE 1: RETRIEVAL ASSISTANT
        # Create a new thread for retrieval
        retrieval_thread = self.client.beta.threads.create()
        
        # Create prompt for the retrieval assistant
        retrieval_prompt = f"""
        The user is asking how many documents mention '{term}'.
        
        Here is the count information:
        
        {context}
        
        Please format this information according to the guidelines for count/mention queries. Focus on providing a comprehensive list of ALL documents that mention '{term}'. Be brief but thorough.
        """
        
        # Rate limit API calls
        self.rate_limit()
        
        # Create message in the retrieval thread
        self.client.beta.threads.messages.create(
            thread_id=retrieval_thread.id,
            role="user",
            content=retrieval_prompt
        )
        
        # Run the retrieval assistant
        retrieval_run = self.client.beta.threads.runs.create(
            thread_id=retrieval_thread.id,
            assistant_id=RETRIEVAL_ASSISTANT_ID
        )
        
        # Wait for completion with timeout
        if not self.wait_for_run_completion(retrieval_thread.id, retrieval_run.id, max_wait_sec=60):
            # Clean up thread
            try:
                self.client.beta.threads.delete(thread_id=retrieval_thread.id)
            except Exception:
                pass
            
            # Fallback to direct response if retrieval fails
            response = f"There are {count} Confluence pages that mention '{term}':\n\n"
            for i, page in enumerate(page_details, 1):
                response += f"{i}. {page['title']}\n"
            
            return response
        
        # Get the retrieval results
        retrieval_messages = self.client.beta.threads.messages.list(
            thread_id=retrieval_thread.id
        )
        
        # Extract the retrieved information
        retrieved_info = ""
        for message in retrieval_messages.data:
            if message.role == "assistant":
                retrieved_info = message.content[0].text.value
                break
        
        # Clean up the retrieval thread
        try:
            self.client.beta.threads.delete(thread_id=retrieval_thread.id)
        except Exception:
            pass
        
        # STAGE 2: ANALYSIS ASSISTANT
        # Initialize main thread if not already done
        self.initialize_assistant_thread()
        
        # Create synthesis prompt for the analysis assistant
        synthesis_prompt = f"""
        The user asked how many documents mention '{term}'.
        
        Here is the information from our retrieval system:
        
        {retrieved_info}
        
        Please format this as a clean, well-structured response according to the guidelines for count queries. Verify that EVERY document is included in your response.
        """
        
        # Rate limit API calls
        self.rate_limit()
        
        # Create message in the main thread
        self.client.beta.threads.messages.create(
            thread_id=self.thread_id,
            role="user",
            content=synthesis_prompt
        )
        
        # Run the analysis assistant
        synthesis_run = self.client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=ANALYSIS_ASSISTANT_ID
        )
        
        # Wait for completion with timeout
        if not self.wait_for_run_completion(self.thread_id, synthesis_run.id, max_wait_sec=60):
            # Fallback to direct response if synthesis fails
            response = f"There are {count} Confluence pages that mention '{term}':\n\n"
            for i, page in enumerate(page_details, 1):
                response += f"{i}. {page['title']}\n"
            
            return response
        
        # Retrieve the final answer
        messages = self.client.beta.threads.messages.list(
            thread_id=self.thread_id
        )
        
        # Get the last assistant message
        for message in messages.data:
            if message.role == "assistant":
                answer = message.content[0].text.value
                return answer
        
        # Final fallback
        response = f"There are {count} Confluence pages that mention '{term}':\n\n"
        for i, page in enumerate(page_details, 1):
            response += f"{i}. {page['title']}\n"
        
        return response
    
def main():
    st.set_page_config(page_title="Confluence Knowledge Chatbot", layout="wide")
    
    st.title("Confluence Knowledge Chatbot")
    
    if "configured" not in st.session_state:
        st.session_state.configured = False
    
    # Configuration section
    if not st.session_state.configured:
        st.header("Configuration")
        
        with st.form("config_form"):
            # Get user inputs for Confluence credentials
            confluence_url = st.text_input("Confluence Base URL (e.g., https://your-domain.atlassian.net/wiki)")
            confluence_username = st.text_input("Confluence Username (email)")
            confluence_api_token = st.text_input("Confluence API Token", type="password")
            
            submit_button = st.form_submit_button("Save Configuration")
            
            if submit_button:
                if not confluence_url or not confluence_username or not confluence_api_token:
                    st.error("All fields are required.")
                else:
                    st.session_state.confluence_url = confluence_url
                    st.session_state.confluence_username = confluence_username
                    st.session_state.confluence_api_token = confluence_api_token
                    
                    st.session_state.chatbot = ConfluenceChatbot(
                        confluence_url=confluence_url,
                        confluence_username=confluence_username,
                        confluence_api_token=confluence_api_token
                    )
                    
                    # Set configured flag
                    st.session_state.configured = True
                    st.success("Configuration saved successfully!")
                    st.rerun()
    
    # Main application (only shown after configuration)
    if st.session_state.configured:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.header("Index Content")
            
            # Option to choose between single page or space
            index_option = st.radio("What would you like to index?", ["Single Page", "Entire Space"])
            
            if index_option == "Single Page":
                page_id = st.text_input("Page ID")
                
                if st.button("Index Page"):
                    with st.spinner("Indexing Confluence page..."):
                        try:
                            num_docs = st.session_state.chatbot.index_single_page(page_id)
                            if num_docs > 0:
                                st.success(f"Page indexed successfully! Added {num_docs} document(s) to the index.")
                            else:
                                st.error("Failed to index page. Check the page ID and your credentials.")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            else:
                space_key = st.text_input("Space Key")
                
                if st.button("Index Space"):
                    with st.spinner("Indexing Confluence space..."):
                        try:
                            num_docs = st.session_state.chatbot.index_space(space_key)
                            st.success(f"Indexed {num_docs} pages successfully!")
                        except Exception as e:
                            st.error(f"Error indexing space: {str(e)}")
            
            # Add option to reset configuration
            if st.button("Reset Configuration"):
                st.session_state.configured = False
                st.rerun()
            
            # Add option to clear index without saving
            if st.button("Clear Index"):
                if hasattr(st.session_state, 'chatbot'):
                    st.session_state.chatbot.clear_index()
                    st.success("Index cleared successfully! All data was in-memory only.")
            
            # Add a section for indexing stats
            if hasattr(st.session_state, 'chatbot') and hasattr(st.session_state.chatbot, 'vector_store') and st.session_state.chatbot.vector_store:
                st.subheader("Index Stats")
                try:
                    # Use a simple search to get an idea of index size
                    sample_docs = st.session_state.chatbot.vector_store.similarity_search(
                        "document information",
                        k=500
                    )
                    
                    # Count unique page IDs
                    unique_page_ids = set()
                    for doc in sample_docs:
                        unique_page_ids.add(doc.metadata['id'])
                    
                    st.write(f"Unique pages in index: {len(unique_page_ids)}")
                    st.write(f"Total chunks in index: {len(sample_docs)}")
                except Exception as e:
                    st.write(f"Could not retrieve index stats: {str(e)}")
        
        with col2:
            st.header("Ask me anything about your Confluence pages")
            
            chat_container = st.container()
            
            if "messages" not in st.session_state:
                st.session_state.messages = []
                if hasattr(st.session_state, 'chatbot') and hasattr(st.session_state.chatbot, 'chat_history'):
                    for q, a in st.session_state.chatbot.chat_history:
                        st.session_state.messages.append({"role": "user", "content": q})
                        st.session_state.messages.append({"role": "assistant", "content": a})
            
            # Display chat messages from session state
            with chat_container:
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        st.markdown(f"**You:** {message['content']}")
                    else:
                        st.markdown(f"**Bot:** {message['content']}")
                    st.markdown("---")
            
            # Create a unique form key based on the number of messages
            form_key = f"question_form_{len(st.session_state.messages)}"
            input_key = f"input_field_{len(st.session_state.messages)}"
            
            # Form for submitting question
            with st.form(key=form_key):
                user_question = st.text_input("Your question", key=input_key)
                submit_button = st.form_submit_button("Ask")
                
                if submit_button and user_question:
                    if not hasattr(st.session_state, 'chatbot') or not hasattr(st.session_state.chatbot, 'vector_store') or st.session_state.chatbot.vector_store is None:
                        st.warning("Please index a Confluence space or page first.")
                    else:
                        st.session_state.messages.append({"role": "user", "content": user_question})
                        
                        with st.spinner("Thinking..."):
                            try:
                                # Use the ask method
                                answer = st.session_state.chatbot.ask(user_question)
                                
                                # Add to chat history
                                st.session_state.chatbot.chat_history.append((user_question, answer))
                                st.session_state.messages.append({"role": "assistant", "content": answer})
                            except Exception as e:
                                error_message = f"Sorry, an error occurred: {str(e)}"
                                st.session_state.messages.append({"role": "assistant", "content": error_message})
                        
                        st.rerun()

if __name__ == "__main__":
    main()