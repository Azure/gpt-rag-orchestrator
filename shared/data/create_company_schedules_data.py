import json 
import os 
def create_company_schedules(companies: list[str], report_types: list[str]) -> list[dict]:
    """
    Create a list of company schedules for a given list of companies and report types 
    """
    schedules = []
    for company in companies:
        for report_type in report_types:
            schedules.append({"companyId": company, 
                              "reportType": report_type, 
                              "isActive": True,
                              "frequency": "twice_a_day"})
    return schedules

def save_to_json(data: list[dict], file_path: str) -> None:
    with open(file_path, 'w') as f:
        json.dump(data, f, indent = 4)

if __name__ == "__main__":
    companies = ['HD', 'LOW', 'WMT', 'AMZN', 'COST', 'TGT']
    report_types = ["10-K", "10-Q", "8-K", "DEF 14A"]
    schedules = create_company_schedules(companies, report_types)
    
    # save to json in the same directory as this file 
    file_path = os.path.join(os.path.dirname(__file__), "companyID_schedules.json")
    save_to_json(schedules, file_path)



