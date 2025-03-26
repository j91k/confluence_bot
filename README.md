# Confluence Knowledge Chatbot

A Streamlit-based application that indexes your Confluence content and provides an AI-powered chatbot interface to query your organizational knowledge.

## Features

- Index entire Confluence spaces or individual pages
- Query your knowledge base using natural language
- Get answers based on your Confluence content
- Specialized handling for document-specific and count queries
- Rate-limited API calls to avoid hitting Confluence and OpenAI limits

## Prerequisites

- Python 3.8+
- Confluence instance (Cloud or Server)
- Confluence API token
- OpenAI API key

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/confluence-knowledge-chatbot.git
cd confluence-knowledge-chatbot
```

### 2. Install Dependencies

You can install dependencies directly:

```bash
pip install -r requirements.txt
```

**Note:** While not required, using a virtual environment is recommended for avoiding dependency conflicts:

```bash
# Optional: Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Then install dependencies
pip install -r requirements.txt
```

### 3. Create a `.env` File

Create a `.env` file in the project root directory with the following environment variables:

```
OPENAI_API_KEY=your_openai_api_key
RETRIEVAL_ASSISTANT_ID=your_retrieval_assistant_id
ANALYSIS_ASSISTANT_ID=your_analysis_assistant_id
```

**Note:** You do not need to add your Confluence API token to the `.env` file. You will provide this information directly through the application's configuration interface when you first run it.

### 4. Set Up OpenAI Assistants

You need to input two OpenAI Assistants ID shared within:

1. **Retrieval Assistant**: Focused on extracting relevant information from documents
2. **Analysis Assistant**: Focused on synthesizing and formatting answers

### 5. Get Confluence API Token

1. Log in to your Atlassian account at [id.atlassian.com](https://id.atlassian.com/)
2. Navigate to Security > API tokens
3. Click "Create API token"
4. Give your token a name (e.g., "Confluence Knowledge Chatbot")
5. Click "Create"
6. Copy the generated token and keep it secure

### 6. Run the Application

```bash
streamlit run confluence_chat_bot.py
```

Access the application at `http://localhost:8501` in your web browser.

## Usage

1. **Configure the Application**:
   - Enter your Confluence URL (e.g., `https://your-domain.atlassian.net/wiki`)
   - Enter your Confluence username (email address)
   - Enter your Confluence API token
   - Click "Save Configuration"

2. **Index Content**:
   - To index a single page: Enter the page ID and click "Index Page"
   - To index an entire space: Enter the space key and click "Index Space"

3. **Ask Questions**:
   - Type your question in the input field and click "Ask"
   - The chatbot will respond based on your indexed Confluence content

## Query Types

The chatbot handles several types of queries effectively:

- **General questions**: What is our vacation policy?
- **Document-specific questions**: What does the Onboarding Guide say about equipment setup?
- **Count queries**: How many documents mention "security"?

## Finding Confluence Page IDs and Space Keys

### Finding Page ID
1. Open the page in Confluence
2. Look at the URL, which typically follows this format:
   `https://your-domain.atlassian.net/wiki/spaces/SPACEKEY/pages/PAGEID/Page+Title`
3. The page ID is the number after "pages/" in the URL

### Finding Space Key
1. Open any page in the space
2. Look at the URL, which typically follows this format:
   `https://your-domain.atlassian.net/wiki/spaces/SPACEKEY/pages/...`
3. The space key is the all-caps code after "spaces/" in the URL

## Troubleshooting

- **Rate Limiting**: If you encounter errors related to rate limiting, the application will automatically retry with delays. For large spaces, indexing may take some time.
- **API Token Issues**: Ensure your API token has the necessary permissions and hasn't expired.
- **Vector Store**: The vector store is in-memory only and will reset when the application is restarted. Re-index your content if needed.

## Architecture

The application uses a dual-assistant approach:
1. A **Retrieval Assistant** extracts relevant information from the knowledge base
2. An **Analysis Assistant** formats and synthesizes the final answer

This separation improves both retrieval quality and final answer formatting.
