import os
from typing import Optional, List, Dict, Iterable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re
import glob
import string
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from langchain_experimental.open_clip import OpenCLIPEmbeddings

try:
    from pinecone.grpc import PineconeGRPC as Pinecone
except ImportError:
    from pinecone import Pinecone

from pinecone_text.hybrid import hybrid_convex_scale

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .vectordbinterface import DocumentManager
from langchain_core.documents import Document


########################################################################
# PineconeDocumentManager class (Based on the DocumentManager interface)
########################################################################
class PineconeDocumentManager(DocumentManager):
    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        """
        Initializes a PineconeDB object.
        :param api_key: API key (default: the 'PINECONE_API_KEY' environment variable).
        """
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required. Provide it as an argument or set it in the environment variable 'PINECONE_API_KEY'."
            )
        # Initialize Pinecone
        self.pc_db = Pinecone(api_key=self.api_key)

    def check_indexes(self):
        """
        Prints all indexes present in Pinecone.
        """
        try:
            all_indexes = self.pc_db.list_indexes()
            print(f"Existing Indexes: {all_indexes}")
        except Exception as e:
            print(f"Error listing indexes: {e}")
            return []

    def create_index(
        self,
        index_name: str,
        dimension: int,
        metric: str,
        spec: Optional[object] = None,
    ):
        """
        Creates an index or reuses it if it already exists.
        :param index_name: Name of the index to create.
        :param dimension: Number of vector dimensions.
        :param metric: Distance metric to use (e.g., "cosine", "dotproduct").
        :param spec: A ServerlessSpec or PodSpec object.
        """
        try:
            # Check existing indexes
            all_indexes = self.pc_db.list_indexes()
            existing_indexes = [index.name for index in all_indexes]
            if index_name in existing_indexes:
                print(f"Using existing index: {index_name}")
                return self.pc_db.Index(index_name)

            # Create index
            print(f"Creating index '{index_name}'...")
            self.pc_db.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=spec,
            )
            return self.pc_db.Index(index_name)
        except Exception as e:
            print(f"Error creating index: {e}")
            raise

    def describe_index(self, index_name: str):
        """
        Returns the status of the specified index.
        :param index_name: The name of the index for which to retrieve the status.
        """
        try:
            return self.pc_db.describe_index(index_name)
        except Exception as e:
            print(f"Error describing index '{index_name}': {e}")
            raise

    def get_index(self, index_name: str):
        """
        Returns the specified index object.
        :param index_name: The name of the index to return.
        """
        return self.pc_db.Index(index_name)

    def delete_index(self, index_name: str):
        """
        Deletes the specified index.
        :param index_name: The name of the index to delete.
        """
        try:
            self.pc_db.delete_index(index_name)
            print(f"Index '{index_name}' deleted.")
        except Exception as e:
            print(f"Error deleting index '{index_name}': {e}")
            raise

    def list_indexes(self):
        """
        Returns all indexes that exist in Pinecone.
        """
        try:
            return self.pc_db.list_indexes()
        except Exception as e:
            print(f"Error listing indexes: {e}")
            return []

    def upsert_documents(
        self,
        index,
        contents: List[str],
        metadatas: dict,
        embedder,
        sparse_encoder,
        namespace: str,
        batch_size: int = 32,
    ):
        """
        Converts documents to vectors and upserts them into Pinecone.
        :param index: Pinecone Index object.
        :param contents: List of documents.
        :param metadatas: Dictionary of metadata.
        :param embedder: Dense vector embedding object.
        :param sparse_encoder: Sparse vector embedding object.
        :param namespace: Pinecone namespace.
        :param batch_size: Batch size for processing.
        """
        total_batches = (len(contents) + batch_size - 1) // batch_size

        for batch_start in tqdm(
            range(0, len(contents), batch_size),
            desc="Processing Batches",
            total=total_batches,
        ):
            batch_end = min(batch_start + batch_size, len(contents))

            # Extract current batch data
            content_batch = contents[batch_start:batch_end]
            metadata_batch = {
                key: metadatas[key][batch_start:batch_end] for key in metadatas
            }

            # Dense vector creation (batch)
            dense_vectors = embedder.embed_documents(content_batch)

            # Sparse vector creation (batch)
            sparse_vectors = sparse_encoder.encode_documents(content_batch)

            # Configuring data to upsert into Pinecone
            vectors = [
                {
                    "id": f"doc-{batch_start + i}",
                    "values": dense_vectors[i],
                    "sparse_values": {
                        "indices": sparse_vectors[i]["indices"],
                        "values": sparse_vectors[i]["values"],
                    },
                    "metadata": {
                        **{key: metadata_batch[key][i] for key in metadata_batch},
                        "context": context,
                    },
                }
                for i, context in enumerate(content_batch)
            ]

            # Upsert to Pinecone
            index.upsert(vectors=vectors, namespace=namespace)

        # Print index stats
        print(index.describe_index_stats())

    def process_batch(
        self,
        index,
        content_batch: List[str],
        metadata_batch: Dict[str, List],
        embedder,
        sparse_encoder,
        namespace: str,
        batch_start: int,
    ):
        """
        Processes a single batch and upserts it into Pinecone.
        """
        # Dense vectors creation
        dense_vectors = embedder.embed_documents(content_batch)

        # Sparse vectors creation
        sparse_vectors = sparse_encoder.encode_documents(content_batch)

        # Configuring data to upsert into Pinecone
        vectors = [
            {
                "id": f"doc-{batch_start + i}",
                "values": dense_vectors[i],
                "sparse_values": {
                    "indices": sparse_vectors[i]["indices"],
                    "values": sparse_vectors[i]["values"],
                },
                "metadata": {
                    **{key: metadata_batch[key][i] for key in metadata_batch},
                    "context": content,
                },
            }
            for i, content in enumerate(content_batch)
        ]

        # Upsert to Pinecone
        index.upsert(vectors=vectors, namespace=namespace)

    def upsert_documents_parallel(
        self,
        index,
        contents: List[str],
        metadatas: Dict[str, List],
        embedder,
        sparse_encoder,
        namespace: str,
        batch_size: int = 32,
        max_workers: int = 8,
    ):
        """
        Upserts documents into Pinecone in parallel.
        :param index: Pinecone Index object.
        :param contents: List of documents.
        :param metadatas: Metadata dictionary.
        :param embedder: Dense vector generator object.
        :param sparse_encoder: Sparse vector generator object.
        :param namespace: Pinecone namespace.
        :param batch_size: Batch size for processing.
        :param max_workers: Number of parallel workers.
        """
        # Prepare batches
        batches = [
            (
                contents[batch_start : batch_start + batch_size],
                {
                    key: metadatas[key][batch_start : batch_start + batch_size]
                    for key in metadatas
                },
                batch_start,
            )
            for batch_start in range(0, len(contents), batch_size)
        ]

        # Parallel processing using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self.process_batch,
                    index,
                    batch[0],
                    batch[1],
                    embedder,
                    sparse_encoder,
                    namespace,
                    batch[2],
                )
                for batch in batches
            ]

            # Display parallel job status with tqdm
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing Batches in Parallel",
            ):
                future.result()
        # Print index stats
        print(index.describe_index_stats())

    def create_hybrid_search_retriever(
        self,
        index_name: str,
        embeddings,
        sparse_encoder,
        namespace: str,
        top_k: int = 4,
        alpha: float = 0.5,
    ):
        """
        Initializes a hybrid search retriever.
        :param index_name: Pinecone index name.
        :param embeddings: Dense vector generator object (e.g., OpenAIEmbeddings).
        :param sparse_encoder: Sparse vector generator object (e.g., BM25Encoder).
        :param namespace: Pinecone namespace.
        :param top_k: Number of search results to return.
        :param alpha: Weight ratio between dense and sparse vectors.
        :return: A method to execute the hybrid search retriever.
        """
        # Checks the existence of the specified index.
        all_indexes = self.pc_db.list_indexes()
        existing_indexes = [index.name for index in all_indexes]
        if index_name not in existing_indexes:
            raise ValueError(
                f"[ERROR] Index '{index_name}' does not exist. Please create it first."
            )

        # Creates an Index object.
        try:
            index = self.pc_db.Index(index_name)
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to access index '{index_name}': {e}")

        def retriever_invoke(query: str, **kwargs) -> List[Dict]:
            """
            Dynamically processes search parameters and executes the query.
            :param query: The search query.
            :param kwargs: Search parameters (e.g., top_k, alpha).
            :return: A list of search results.
            """
            nonlocal top_k, alpha
            if "top_k" in kwargs:
                top_k = kwargs.pop("top_k")
            if "alpha" in kwargs:
                alpha = kwargs.pop("alpha")

            try:
                sparse_vec = sparse_encoder.encode_queries(query)
                dense_vec = embeddings.embed_query(query)
            except Exception as e:
                raise RuntimeError(f"[ERROR] Failed to encode query: {e}")

            dense_vec, sparse_vec = hybrid_convex_scale(dense_vec, sparse_vec, alpha)

            try:
                result = index.query(
                    vector=dense_vec,
                    sparse_vector=sparse_vec,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=namespace,
                    **kwargs,
                )
                return result.get("matches", [])
            except Exception as e:
                raise RuntimeError(f"[ERROR] Query execution failed: {e}")

        print(f"[INFO] Hybrid Search Retriever initialized for index '{index_name}'.")
        return retriever_invoke

    def upsert_images_parallel(
        self,
        index,
        image_paths: list,
        prompts: list,
        categories: list,
        image_embedding,
        namespace: str,
        batch_size: int = 32,
        max_workers: int = 8,
    ):
        """
        Upserts images to Pinecone in parallel.

        :param index: Pinecone Index object
        :param image_paths: List of image file paths
        :param prompts: List of prompts
        :param categories: List of categories
        :param image_embedding: OpenCLIPEmbeddings object
        :param namespace: Pinecone namespace
        :param batch_size: Batch size
        :param max_workers: Number of parallel worker threads
        """
        if not (len(image_paths) == len(prompts) == len(categories)):
            raise ValueError(
                "[ERROR] image_paths, prompts, and categories must have the same length"
            )

        def process_batch(batch):
            vectors = []
            for img_path, prompt, category in batch:
                image_vector = image_embedding.embed_image([img_path])[0]

                vectors.append(
                    {
                        "id": os.path.basename(img_path),
                        "values": image_vector,
                        "metadata": {
                            "prompt": prompt,
                            "category": category,
                            "file_name": os.path.basename(img_path),
                        },
                    }
                )

            index.upsert(vectors=vectors, namespace=namespace)
            return len(vectors)

        data = list(zip(image_paths, prompts, categories))
        batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

        total_uploaded = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_batch, batch): batch for batch in batches
            }

            for future in tqdm(
                as_completed(futures),
                total=len(batches),
                desc="Uploading image batches",
            ):
                try:
                    uploaded = future.result()
                    total_uploaded += uploaded
                except Exception as e:
                    print(f"[ERROR] Batch upload failed: {e}")

        print(f"Uploaded {total_uploaded} images to Pinecone.")

    def search_by_text(
        self, index, query, clip_embedder, namespace, top_k=5, local_image_paths=None
    ):
        """
        Searches for similar images in Pinecone based on a text query.

        :param index: Pinecone Index object
        :param query: Text query
        :param clip_embedder: OpenCLIPEmbeddings object
        :param namespace: Pinecone namespace
        :param top_k: Number of top results to return
        :param local_image_paths: List of local image paths (matched with retrieved files)
        """
        print(f"Text Query: {query}")
        query_vector = clip_embedder.embed_query([query])

        results = index.query(
            vector=query_vector, top_k=top_k, namespace=namespace, include_metadata=True
        )

        fig, axes = plt.subplots(1, len(results["matches"]), figsize=(15, 5))
        for ax, result in zip(axes, results["matches"]):
            print(
                f"Category: {result['metadata']['category']}, "
                f"Prompt: {result['metadata']['prompt']}, Score: {result['score']}"
            )
            img_file = result["metadata"]["file_name"]
            img_full_path = next(
                (
                    path
                    for path in local_image_paths
                    if os.path.basename(path) == img_file
                ),
                None,
            )
            if img_full_path and os.path.exists(img_full_path):
                img = Image.open(img_full_path)
                ax.imshow(img)
                ax.set_title(f"Score: {result['score']:.2f}")
                ax.axis("off")
            else:
                print(f"[WARNING] Image not found for: {img_file}")
                ax.axis("off")
                ax.set_title("Image Not Found")
        plt.tight_layout()
        plt.show()

    def search_by_image(
        self, index, img_path, clip_embedder, namespace, top_k=5, local_image_paths=None
    ):
        """
        Searches for similar images in Pinecone based on a given image.

        :param index: Pinecone Index object
        :param img_path: Path to the query image file
        :param clip_embedder: OpenCLIPEmbeddings object
        :param namespace: Pinecone namespace
        :param top_k: Number of top results to return
        :param local_image_paths: List of local image paths (matched with retrieved files)
        """
        print(f"Image Query: {img_path}")
        query_vector = clip_embedder.embed_image([img_path])

        # Check if the vector is nested and extract
        if isinstance(query_vector, list) and isinstance(query_vector[0], list):
            query_vector = query_vector[0]

        results = index.query(
            vector=query_vector, top_k=top_k, namespace=namespace, include_metadata=True
        )

        fig, axes = plt.subplots(1, len(results["matches"]), figsize=(15, 5))
        for ax, result in zip(axes, results["matches"]):
            print(
                f"Category: {result['metadata']['category']}, "
                f"Prompt: {result['metadata']['prompt']}, Score: {result['score']}"
            )
            img_file = result["metadata"]["file_name"]
            img_full_path = next(
                (
                    path
                    for path in local_image_paths
                    if os.path.basename(path) == img_file
                ),
                None,
            )
            if img_full_path:
                img = Image.open(img_full_path)
                ax.imshow(img)
                ax.set_title(f"Score: {result['score']:.2f}")
                ax.axis("off")
        plt.show()

    def _initialize_openclip(self, model_name: str, checkpoint: str):
        embedding_instance = OpenCLIPEmbeddings(
            model_name=model_name, checkpoint=checkpoint
        )
        print("[INFO] OpenCLIP model initialized.")
        return embedding_instance

    @staticmethod
    def save_temp_image(image: Image) -> str:
        """
        Saves an image to a temporary file and returns its file path.
        """
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        image.save(temp_file, format="PNG")
        temp_file.close()
        return temp_file.name

    def upload_images(
        self,
        index: str,
        image_paths: List[str],
        prompts: List[str],
        categories: List[str],
        image_embedding: str,
        namespace: str,
    ):
        """
        Uploads image embeddings to the Pinecone index.

        :param image_paths: List of image file paths
        :param prompts: List of prompts associated with the images
        :param categories: List of categories associated with the images
        """
        vectors = []
        for img_path, prompt, category in tqdm(
            zip(image_paths, prompts, categories),
            total=len(image_paths),
            desc="Processing Images",
        ):
            # Generate image embeddings
            image_vector = image_embedding.embed_image([img_path])[0]

            # Prepare vector for Pinecone
            vectors.append(
                {
                    "id": os.path.basename(img_path),
                    "values": image_vector,
                    "metadata": {
                        "prompt": prompt,
                        "category": category,
                        "file_name": os.path.basename(img_path),
                    },
                }
            )

        # Upload vectors to Pinecone
        index.upsert(vectors=vectors, namespace=namespace)
        print(f"Uploaded {len(vectors)} images to Pinecone.")

    def upsert(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Implements the interface method to upsert documents.
        Expects kwargs to include: index, embedder, sparse_encoder, namespace, batch_size.
        """
        index = kwargs.get("index")
        embedder = kwargs.get("embedder")
        sparse_encoder = kwargs.get("sparse_encoder")
        namespace = kwargs.get("namespace")
        batch_size = kwargs.get("batch_size", 32)
        self.upsert_documents(
            index,
            list(texts),
            metadatas,
            embedder,
            sparse_encoder,
            namespace,
            batch_size,
        )

    def upsert_parallel(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        workers: int = 10,
        **kwargs: Any,
    ) -> None:
        """
        Implements the interface method to upsert documents in parallel.
        Expects kwargs to include: index, embedder, sparse_encoder, namespace.
        """
        index = kwargs.get("index")
        embedder = kwargs.get("embedder")
        sparse_encoder = kwargs.get("sparse_encoder")
        namespace = kwargs.get("namespace")
        self.upsert_documents_parallel(
            index,
            list(texts),
            metadatas,
            embedder,
            sparse_encoder,
            namespace,
            batch_size,
            workers,
        )

    def search(self, query: str = None, k: int = 10, **kwargs: Any) -> dict:
        index = kwargs.get("index")
        namespace = kwargs.get("namespace")
        sparse_vector = kwargs.get("sparse_vector")
        k = kwargs.get("top_k")
        include_metadata = kwargs.get("include_metadata", True)

        results = index.query(
            namespace=namespace,
            vector=query,
            sparse_vector=sparse_vector,
            top_k=k,
            include_metadata=include_metadata,
        )
        return results

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """
        Implements the interface method to delete documents.
        This wrapper calls the delete method on the index.
        """
        if self.index:
            self.index.delete(ids=ids, filter=filters, namespace=self.namespace)
            print(f"Deleted documents from index '{self.index_name}'.")


########################################################################
# DocumentProcessor class (For preprocessing PDF files, etc.)
########################################################################


class DocumentProcessor:
    def __init__(
        self,
        directory_path: str,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        use_basename: bool = False,
    ):
        """
        Initializes the document processing class.

        Parameters:
            - directory_path: The directory path where documents are located
            - chunk_size: Text chunk size
            - chunk_overlap: Chunk overlap length
            - use_basename: Whether to use only the file name for the 'source' metadata
        """
        self.directory_path = directory_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.use_basename = use_basename

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans the text.

        - Removes non-ASCII characters
        - Removes extra spaces and trims the text
        - Removes patterns where special characters and numbers repeat three or more times
        """
        text = re.sub(r"[^\x00-\x7F]+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[0-9#%$&()*+,\-./:;<=>?@\[\]^_`{|}~]{3,}", "", text)
        return text

    def process_pdf_files(self, directory_path: str) -> List[Document]:
        """
        Loads, preprocesses, and splits PDF files.
        """
        split_docs = []
        files = sorted(glob.glob(directory_path))
        if not files:
            print(f"[WARNING] No PDF files found in directory: {directory_path}")
            return split_docs
        for file in files:
            loader = PyMuPDFLoader(file)
            raw_docs = loader.load_and_split(self.text_splitter)
            for doc in raw_docs:
                doc.page_content = self.clean_text(doc.page_content)
                if self.use_basename and "source" in doc.metadata:
                    doc.metadata["source"] = os.path.basename(doc.metadata["source"])
                split_docs.append(doc)
        print(f"[INFO] Processed {len(split_docs)} documents from {len(files)} files.")
        return split_docs

    def preprocess_documents(self, docs, min_length=5):
        """
        Cleans and filters document data.
        :param docs: List of raw documents.
        :param min_length: Minimum text length to save.
        :return: Cleaned content and metadata.
        """
        contents = []
        metadatas = {key: [] for key in ["source", "page", "author"]}

        for doc in tqdm(docs, desc="Preprocessing documents"):
            content = self.clean_text(doc.page_content.strip())
            if content and len(content) >= min_length:
                contents.append(content)
                for k in metadatas.keys():
                    value = doc.metadata.get(k)
                    if k == "source" and self.use_basename:
                        value = os.path.basename(value)
                    try:
                        metadatas[k].append(int(value))
                    except (ValueError, TypeError):
                        metadatas[k].append(value)

        return contents, metadatas


########################################################################
# NLTKBM25Tokenizer class (NLTK-based BM25 tokenizer)
########################################################################


class NLTKBM25Tokenizer:
    def __init__(self, stop_words: Optional[List[str]] = None):
        """
        Initialize NLTK-based BM25 tokenizer.
        :param stop_words: List of custom stop words (default: None).
        """

        self._initialize_nltk()

        # Set stop words and punctuation
        self._stop_words = (
            set(stop_words)
            if stop_words
            else set(nltk.corpus.stopwords.words("english"))
        )
        self._punctuation = set(string.punctuation)

    @staticmethod
    def _initialize_nltk():
        """
        Initialize NLTK settings and download necessary data.
        """
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        try:
            print("[INFO] Downloading NLTK stopwords and punkt tokenizer...")
            nltk.download("stopwords")
            nltk.download("punkt")
            print("[INFO] NLTK setup completed.")
        except Exception as e:
            print(f"[ERROR] Failed to download NLTK resources: {e}")

    def add_stop_words(self, words: List[str]):
        """
        Add custom stop words.
        :param words: List of stop words to add.
        """
        self._stop_words.update(words)

    def remove_stop_words(self, words: List[str]):
        """
        Remove specific words from the existing stop words.
        :param words: List of stop words to remove.
        """
        for word in words:
            self._stop_words.discard(word)

    def __call__(self, text: str) -> List[str]:
        """
        Tokenize the text and remove stop words and punctuation.
        :param text: Input text.
        :return: List of cleaned tokens.
        """
        tokens = nltk.word_tokenize(text)
        return [
            word.lower()
            for word in tokens
            if word not in self._punctuation and word.lower() not in self._stop_words
        ]
