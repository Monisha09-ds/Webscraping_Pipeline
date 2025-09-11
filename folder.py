import os

# Project structure for Web Scraper + Embedding Pipeline
structure = {
    "": ["README.md", ".env", "requirements.txt"],

    "src": ["__init__.py", "main.py"],

    # Scraper modules
    "src/scraper": [
        "__init__.py",
        "scraper.py",
        "pagination.py",
        "markdown_saver.py",
    ],

    # Storage for scraped content
    "sitecontent": [],

    # Vector DB modules
    "src/vectorstore": [
        "__init__.py",
        "embeddings.py",
        "store.py",
        "search.py",
    ],

    # Utilities
    "src/utils": [
        "__init__.py",
        "logger.py",
        "config.py",
    ],

    
}


def create_structure(base_path="."):
    """
    Creates project folders and files from the structure dict.
    """
    for folder, files in structure.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)

        for file in files:
            file_path = os.path.join(folder_path, file)
            if not os.path.exists(file_path):
                with open(file_path, "w", encoding="utf-8") as f:
                    if file.endswith(".py"):
                        f.write("# " + file + "\n")
                    elif file == "README.md":
                        f.write("# Web Scraper & VectorStore Pipeline\n")
                    elif file == "requirements.txt":
                        f.write("requests\nbeautifulsoup4\nmarkdownify\nfaiss-cpu\nchromadb\n")
                    elif file == ".env":
                        f.write("# Environment variables\n")
    print("✅ Project structure created successfully!")


if __name__ == "__main__":
    create_structure()
