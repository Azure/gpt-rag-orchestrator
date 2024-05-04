import re
from langchain_core.output_parsers import StrOutputParser


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
    docs = [x.page_content for x in docs]
    source_knowledge = "\n---\n".join(docs)
    return source_knowledge
