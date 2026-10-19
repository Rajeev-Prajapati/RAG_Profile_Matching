from utils.metadata_extractor import MetadataExtractor
from pathlib import Path
from milestone1.file_tools import list_files, read_file
from chunking.resume_chunker import ResumeChunker
from embeddings.embedding_service import EmbeddingService
from vector_store.chroma_store import ChromaStore

class ResumeRAG:

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    def __init__(self, resume_folder="resumes"):

        # Absolute path to the project directory
        self.project_root = Path(__file__).resolve().parent
        self.metadata_extractor = MetadataExtractor()
        self.embedding_service = EmbeddingService()
        self.embeddings = None
        self.chunker = ResumeChunker(
            chunk_size=600,
            chunk_overlap=100
        )
        self.chunks = []
        self.vector_store = ChromaStore(
        persist_directory="chroma_db",
        collection_name="synthetic_resume_chunks"
    )
        # Resolve the resume directory
        folder = Path(resume_folder)

        if folder.is_absolute():
            self.resume_folder = folder
        else:
            self.resume_folder = self.project_root / folder

        self.documents = []

    def get_resume_files(self):
        """
        Get resume file paths.

        First try the Milestone 1 list_files() tool.
        If its result cannot be interpreted, use pathlib.
        """

        if not self.resume_folder.is_dir():
            raise FileNotFoundError(
                f"Resume directory not found: {self.resume_folder}"
            )

        print(f"\nScanning folder: {self.resume_folder}")

        # Try Milestone 1 list_files()
        try:
            result = list_files(str(self.resume_folder))

            print("Milestone 1 list_files result:", result)

            files = self.normalize_file_list(result)

            if files is not None:
                return files

            print(
                "Milestone 1 result could not be interpreted. "
                "Using pathlib for file discovery."
            )

        except Exception as error:
            print(f"Milestone 1 list_files error: {error}")

        # Fallback for discovering files only
        return [
            path
            for path in self.resume_folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

    def normalize_file_list(self, result):
        """
        Convert supported list_files() return formats
        into a list of Path objects.

        Returns None if the result format is unknown.
        """

        if isinstance(result, dict):
            if result.get("success") is False:
                print("File tool error:", result.get("error"))
                return None

            result = result.get("files")

        if not isinstance(result, list):
            return None

        paths = []

        for item in result:

            if isinstance(item, dict):

                if item.get("success") is False:
                    print("File tool error:", item.get("error"))
                    return None

                file_path = (
                    item.get("path")
                    or item.get("file_path")
                    or item.get("name")
                )

                if not file_path:
                    return None

            elif isinstance(item, (str, Path)):
                file_path = item

            else:
                return None

            path = Path(file_path)

            if not path.is_absolute():

                # Support both:
                # "resumes/john.pdf" and "john.pdf"
                project_relative = self.project_root / path
                folder_relative = self.resume_folder / path

                if project_relative.is_file():
                    path = project_relative

                elif folder_relative.is_file():
                    path = folder_relative

                else:
                    print(f"File not found: {file_path}")
                    continue

            if (
                path.is_file()
                and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ):
                paths.append(path)

        return paths

    def extract_text(self, file_path):
        """
        Extract text using the Milestone 1 read_file() tool.
        """

        result = read_file(str(file_path))

        # Some tools return plain text
        if isinstance(result, str):
            return result

        # Some tools return a dictionary
        if isinstance(result, dict):

            if result.get("success") is False:
                raise ValueError(
                    result.get("error", "Unable to read file")
                )

            text = (
                result.get("content")
                or result.get("text")
            )

            if isinstance(text, str):
                return text

        raise TypeError(
            f"Unsupported read_file() result: {type(result).__name__}"
        )

    def load_resumes(self):
        """
        Load all supported resumes and return
        structured document dictionaries.
        """

        self.documents = []

        resume_files = self.get_resume_files()

        print(f"\nFound {len(resume_files)} resume file(s)")

        for path in resume_files:

            try:
                text = self.extract_text(path)

                if not text.strip():
                    print(f"Skipped empty resume: {path.name}")
                    continue

                metadata = self.metadata_extractor.extract_metadata(text)

                document = {
                    "resume_path": str(path),
                    "file_name": path.name,
                    "file_type": path.suffix.lower(),
                    "text": text,
                    "metadata": metadata
                }

                self.documents.append(document)

                print(f"Successfully loaded: {path.name}")

            except Exception as error:
                print(f"Failed to load {path.name}: {error}")

        return self.documents

    def display_resumes(self):
        """
        Display summary of loaded resumes.
        """

        print("\n" + "=" * 55)
        print("RESUME LOADING SUMMARY")
        print("=" * 55)

        print(f"Total resumes loaded: {len(self.documents)}")

        for index, document in enumerate(self.documents, start=1):

            print("\n" + "-" * 55)
            print(f"Resume Number: {index}")
            print(f"File Name: {document['file_name']}")
            print(f"File Type: {document['file_type']}")
            print(f"File Path: {document['resume_path']}")
            print(f"Characters: {len(document['text'])}")

            print("\nContent Preview:")
            print(document["text"][:300])

            print("-" * 55)

        metadata = document["metadata"]

        print("\nCandidate Metadata:")
        print("Name:", metadata["candidate_name"])
        print("Skills:", ", ".join(metadata["skills"]))
        print("Experience:", metadata["experience_years"])
        print("Education:", metadata["education"])


    
    def create_chunks(self):
        """
        Convert loaded resumes into section-aware chunks.
        """

        self.chunks = []

        for document in self.documents:

            resume_chunks = self.chunker.chunk_resume(document)

            self.chunks.extend(resume_chunks)

            print(
                f"Created {len(resume_chunks)} chunks "
                f"from {document['file_name']}"
            )

        return self.chunks

    
    def generate_chunk_embeddings(self):
        """
        Generate embeddings for all resume chunks.
        """

        if not self.chunks:
            print("No chunks found. Run create_chunks() first.")
            return None

        texts = [
            f"Section: {chunk['section']}\n{chunk['text']}"
            for chunk in self.chunks
        ]

        print(f"\nGenerating embeddings for {len(texts)} chunks...")

        self.embeddings = (
            self.embedding_service.generate_embeddings(texts)
        )

        print("Embedding generation completed!")

        print("Embedding shape:", self.embeddings.shape)

        return self.embeddings

    
    
    def store_embeddings(self):

        if not self.chunks:
            print("No chunks available.")
            return 0

        if self.embeddings is None:
            raise ValueError(
                "Generate embeddings before storing chunks"
            )

        if len(self.chunks) != len(self.embeddings):
            raise ValueError(
                "Chunk and embedding counts do not match"
            )

        from collections import defaultdict

        grouped = defaultdict(
            lambda: {"chunks": [], "embeddings": []}
        )

        for chunk, embedding in zip(
            self.chunks,
            self.embeddings
        ):
            resume_path = chunk["resume_path"]

            grouped[resume_path]["chunks"].append(chunk)
            grouped[resume_path]["embeddings"].append(
                embedding
            )

        total_stored = 0

        for resume_path, data in grouped.items():

            count = self.vector_store.replace_resume_chunks(
                chunks=data["chunks"],
                embeddings=data["embeddings"]
            )

            total_stored += count

            print(
                f"Reindexed {count} chunks: {resume_path}"
            )

        print(
            "Total indexed chunks:",
            self.vector_store.count_chunks()
        )

        return total_stored

  

def main():

    print("=" * 55)
    print("RAG BASED RESUME MATCHING SYSTEM")
    print("=" * 55)

    rag = ResumeRAG(
    resume_folder="dataset/resumes"
)

    try:
        # Step 2: Load resumes
        documents = rag.load_resumes()

        if not documents:
            print("No resumes loaded.")
            return

        # Step 3: Metadata extracted during loading
        rag.display_resumes()

        # Step 4: Create chunks
        chunks = rag.create_chunks()

        if not chunks:
            print("No chunks created.")
            return

        # Step 5: Generate embeddings
        embeddings = rag.generate_chunk_embeddings()

        if embeddings is None:
            return

        # Step 6: Store embeddings in ChromaDB
        stored_count = rag.store_embeddings()

        print("\n" + "=" * 55)
        print("VECTOR DATABASE SUMMARY")
        print("=" * 55)

        print("Total resumes:", len(documents))
        print("Total generated chunks:", len(chunks))
        print("Total embeddings:", len(embeddings))
        print("Stored/updated chunks:", stored_count)

        print(
            "Total records in ChromaDB:",
            rag.vector_store.count_chunks()
        )

        print("\nStep 6 completed successfully!")

    except Exception as error:
        print("Application error:", error)


if __name__ == "__main__":
    main()

