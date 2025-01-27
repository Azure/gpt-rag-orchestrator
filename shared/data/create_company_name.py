import json 
import os 
from datetime import datetime, timezone 
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path

class Company(BaseModel): 
    """Company with basic information """

    model_config = ConfigDict(
        frozen = True, 
        validate_assignment = True, 
        json_encoders = {datetime: lambda v: v.isoformat()},
        json_schema_extra={
            "examples": [
                {
                    "name": "Lowes",
                    "id": "1",
                    "ticker": "LOW"
                }
            ]
        }
    )

    name: str = Field(..., min_length = 1, description = "the name of the comapny")
    id: Optional[str] = Field(None, description = "unique id of the company")
    ticker: Optional[str] = Field(None, description = "ticker of the company")
    is_active: bool = Field(default = True, description = "whether the we want to generate company analysis report for this company")
    created_at: datetime = Field(
        default_factory = lambda: datetime.now(timezone.utc),
        description = " UTC timesptamp of whe the company was created"
    )
    
    def to_dict(self) -> dict:
        """Convert company object to dictionary representation """
        data = self.model_dump()
        # Convert datetime to ISO format string
        data['created_at'] = data['created_at'].isoformat()
        return data
    
def create_company_data(companies: List[Company]) -> List[dict]:
    """Convert a list of company objects to a list of dictionaries 
    
    Args:  
        companies: List of company objects to convert 

    Returns: 
        List of dictionaries containing company information 
    """
    return [company.to_dict() for company in companies]

def save_to_json(data: List[dict], file_path: str) -> None: 
    """ Save data to json file """
    try: 
        with open(file_path, "w", encoding = "utf-8") as f: 
            json.dump(data, f, indent = 4, ensure_ascii= False)
    except IOError as e: 
        raise IOError(f"Failed to save data to {file_path}: {str(e)}") from e

def main() -> None:
    """Main function to demonstrate company data creation and storage."""
    try:
        companies = [
            Company(name="Lowes", ticker="LOW"),
            Company(name="Home Depot", ticker="HD")
        ]
        
        company_data = create_company_data(companies)
        print(company_data)

        file_path = Path(__file__).parent / "company_name.json"
        save_to_json(company_data, file_path)
        print(f"Successfully saved company data to {file_path}")
    
    except (ValueError, IOError) as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()

# class Company:
#     def __init__(self, name: str, id: Optional[str] = None, ticker: Optional[str] = None, is_active: bool = True):
#         self.id: Optional[str] = id
#         self.name: str = name
#         self.ticker: Optional[str] = ticker
#         self.created_at: datetime = datetime.now(timezone.utc)
#         self.is_active: bool = is_active


#     def to_dict(self) -> dict:
#         return {
#             "id": self.id, 
#             "name": self.name, 
#             "ticker": self.ticker, 
#             "created_at": self.created_at.isoformat(), 
#             "is_active": self.is_active
#         }

# def create_company_name(companies: list[Company]) -> list[dict]:
#     return [company.to_dict() for company in companies]

# def save_to_json(data: list[dict], file_path: str) -> None:
#     with open(file_path, 'w') as f:
#         json.dump(data, f, indent = 4)

# if __name__ == "__main__":
#     companies = [Company(id="1", name="Lowes", ticker="LOW"), 
#                  Company(id="2", name="Home Depot", ticker="HD")]
#     company_data = create_company_name(companies)
#     print(company_data)

#     # save to json in the same directory as this file 
#     file_path = os.path.join(os.path.dirname(__file__), "company_name.json")
#     save_to_json(company_data, file_path)



