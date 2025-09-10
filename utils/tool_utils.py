from abc import abstractmethod, ABC
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from datetime import datetime, timedelta
import json 

class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


class GenericTool(ABC):
    """
    An abstraction over all tools.
    """

    @classmethod
    def call_tool(cls, **kwargs) -> Optional[List[Dict[str, Any]]]:
        try:
            return cls._tool_wrapper(**kwargs)
        except Exception as e:
            print(f"ERROR WITH TOOL CALL: {e}")

    @classmethod
    @abstractmethod
    def _tool_wrapper(cls, **kwargs) -> Optional[List[Dict[str, Any]]]:
        ...


def load_emails():
    with open("data/real_word_problem/emails.jsonl", "r") as f:
        emails = [json.loads(line) for line in f.readlines()]
    return emails

class EmailSearchTool(GenericTool):
    """
    A tool that searches emails from a mock email database.
    """

    tool_spec = ToolSpec(
        name="email_search",
        description="Performs a lexical search over the user's emails, returning the 10 most relevant email results.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Query to search the user's emails with."},
            },
            "required": ["query"]
        }
    )
    # In practice, this wouldn't exist, and the tool would call an external API service.
    _EMAILS = load_emails()


    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        """
        Parses a date in YYYY/MM/DD format into a datetime object.
        Returns None if parsing fails.
        """
        if not date_str:
            return None
            
        try:
            return datetime.strptime(date_str, "%Y/%m/%d")
        except ValueError:
            print(f"Invalid date format: {date_str}. Expected YYYY/MM/DD")
            return None

    @staticmethod
    def _parse_relative_time(relative_str: str) -> Optional[timedelta]:
        """
        Parses strings like '3d', '2m', '1y' into a timedelta.
        
        Note: This is a simplified implementation:
        - 'm' means 30 days (not calendar months)
        - 'y' means 365 days (not accounting for leap years)
        
        For production use, consider using dateutil.relativedelta instead.
        
        Returns None if parsing fails.
        """
        if not relative_str:
            return timedelta(0)
            
        try:
            unit = relative_str[-1].lower()
            amount = int(relative_str[:-1])

            if unit == 'd':
                return timedelta(days=amount)
            elif unit == 'm':
                return timedelta(days=30 * amount)
            elif unit == 'y':
                return timedelta(days=365 * amount)
            else:
                print(f"Unknown relative time suffix: {unit}")
                return None
        except (ValueError, IndexError):
            print(f"Invalid relative time format: {relative_str}")
            return None


    @classmethod
    def _tool_wrapper(cls, 
            query: Optional[str] = None,
            after: Optional[str] = None,
            before: Optional[str] = None,
        ) -> Optional[List[Dict[str, Any]]]:

        # Absolute date filters
        after_dt = cls._parse_date(after)
        before_dt = cls._parse_date(before)

        filtered_emails = []
        for email in cls._EMAILS:
            email_date = cls._parse_date(email['received_date'])
            if email_date is None:
                print(f"Skipping email with invalid date: {email['subject']}")
                continue

            # after filter => received_date > after_dt
            if after_dt and not (email_date > after_dt):
                continue
            # before filter => received_date < before_dt
            if before_dt and not (email_date < before_dt):
                continue
     
            filtered_emails.append(email)

        # Convert each email into a LangChain Document
        # Use a clear separator between subject and body
        docs = []
        for email in filtered_emails:
            doc = Document(
                page_content=f"Subject: {email['subject']}\nBody: {email['body']}",
                metadata={
                    "subject": email["subject"],
                    "body": email["body"],
                    "sender": email["sender"],
                    "received_date": email["received_date"],
                }
            )
            docs.append(doc)

        # If no documents match filters, return empty list
        if not docs:
            return []

        # Create a local BM25Retriever on these docs
        retriever = BM25Retriever.from_documents(docs)
        retriever.k = min(len(docs), 10)  # Limit results to avoid returning irrelevant matches

        # Query the retriever
        results_docs = retriever.invoke(query)

        # ----- Convert results to the final return structure. -----
        results = []
        for doc in results_docs:
            md = doc.metadata
            results.append({
                "subject": md["subject"],
                "body": md["body"],
                "sender": md["sender"],
                "received_date": md["received_date"],
            })

        return results
    

if __name__ == "__main__":
    print(EmailSearchTool.call_tool(query="whats up", before="2025/09/19"))