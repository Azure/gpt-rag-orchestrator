import json 
import os 
import uuid


def create_subscription_email(emails: list[str], 
                              names: list[str], 
                              is_active: bool = True) -> list[dict]:
    """
    Create a list of company schedules for a given list of companies and report types 
    """
    email_table = []
    for email, name in zip(emails, names):
        email_table.append({
            "id": str(uuid.uuid4()),
            "email": email, 
            "name": name, 
            "isActive": is_active})
    return email_table

def save_to_json(data: list[dict], file_path: str) -> None:
    with open(file_path, 'w') as f:
        json.dump(data, f, indent = 4)

if __name__ == "__main__":
    emails = ['test@test.com', 'test2@test.com', 'test3@test.com']
    names = ['test', 'test2', 'test3']
    email_table = create_subscription_email(emails, names)
    
    # save to json in the same directory as this file 
    file_path = os.path.join(os.path.dirname(__file__), "subscription_emails.json")
    save_to_json(email_table, file_path)



