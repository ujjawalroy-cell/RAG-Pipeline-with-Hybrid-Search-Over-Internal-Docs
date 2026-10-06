from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Anthropic
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient







 