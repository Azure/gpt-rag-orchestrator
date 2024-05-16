import re
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import logging

class LineListOutputParser(StrOutputParser):
    def parse(self, text: str):
        # Split by newlines
        lines = text.strip().split("\n")
        # Remove special characters
        lines = [(re.sub("\W+", " ", k)).strip() for k in lines]
        # Remove the number at the beginning of the line (if exists)
        lines = [re.sub("^\d+\s*", "", k) for k in lines]
        return lines


def retrieval_transform(docs):
    sources = [x.metadata.get("filepath", "") for x in docs]
    docs = [f"Source {i}: {x.metadata.get('filepath', '')} \n{x.page_content}" for i, x in enumerate(docs, start=1)]
    source_knowledge = "\n---\n".join(docs)
    # logging.info(f"SOURCES {sources}")
    return source_knowledge, sources
