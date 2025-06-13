import json
import logging
import os
from datetime import datetime, timedelta, timezone
from azure.storage.blob import (
    BlobServiceClient, 
    ContentSettings, 
    generate_blob_sas,
    BlobSasPermissions
)
from azure.identity import DefaultAzureCredential
from shared.cosmos_db import get_conversation_data
from shared.util import get_conversation

def format_conversation_as_html(conversation_data):
    """
    Convert conversation data to a readable HTML format.
    
    Args:
        conversation_data: The conversation data from Cosmos DB
        
    Returns:
        str: HTML formatted conversation
    """
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Shared Conversation</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }}
            .conversation-header {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .message {{
                background: white;
                margin: 10px 0;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .user-message {{
                background: #e3f2fd;
                margin-left: 20px;
            }}
            .freddaid-message {{
                background: #f3e5f5;
                margin-right: 20px;
            }}
            .role {{
                font-weight: bold;
                color: #1976d2;
                margin-bottom: 8px;
                text-transform: uppercase;
                font-size: 12px;
            }}
            .content {{
                line-height: 1.6;
                white-space: pre-wrap;
            }}
            .timestamp {{
                color: #666;
                font-size: 12px;
                margin-top: 10px;
            }}
            
            .export-info {{
                background: #fff3cd;
                padding: 10px;
                border-radius: 4px;
                margin-bottom: 20px;
                font-size: 12px;
                color: #856404;
            }}
        </style>
    </head>
    <body>
        <div class="export-info">
            📋 This conversation was exported on {export_date}
        </div>
        
        <div class="conversation-header">
            <h1>Conversation Export</h1>
            <p><strong>Started:</strong> {start_date}</p>
            <p><strong>Conversation ID:</strong> {conversation_id}</p>
            <p><strong>Total Messages:</strong> {message_count}</p>
        </div>
        
        <div class="messages">
            {messages_html}
        </div>
    </body>
    </html>
    """
    
    messages_html = ""
    messages = conversation_data.get('messages', [])
    for message in messages:
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        
        css_class = 'user-message' if role == 'user' else 'freddaid-message'
        role_display = '👤 User' if role == 'user' else '🤖 Freddaid'
        
        messages_html += f"""
        <div class="message {css_class}">
            <div class="role">{role_display}</div>
            <div class="content">{content}</div>
        </div>
        """
    
    return html_template.format(
        export_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        start_date=conversation_data.get('start_date', 'Unknown'),
        conversation_id=conversation_data.get('id', 'Unknown'),
        message_count=len(messages),
        messages_html=messages_html
    )

def format_conversation_as_json(conversation_data):
    """
    Convert conversation data to formatted JSON.
    
    Args:
        conversation_data: The conversation data from Cosmos DB
        
    Returns:
        str: JSON formatted conversation
    """
    messages = conversation_data.get('messages', [])
    export_data = {
        "conversation_id": conversation_data.get('id'),
        "export_date": datetime.now().isoformat(),
        "start_date": conversation_data.get('start_date'),
        "message_count": len(messages),
        "messages": []
    }
    
    # Add messages without sensitive user data
    for message in messages:
        export_message = {
            "role": message.get('role'),
            "content": message.get('content')
        }
        export_data["messages"].append(export_message)
    
    return json.dumps(export_data, indent=2, ensure_ascii=False)

def upload_to_blob_storage(content, filename, user_id, content_type="text/html"):
    """
    Upload content to Azure Blob Storage and return shareable URL.
    
    Args:
        content: File content to upload
        filename: Name of the file
        content_type: MIME type of the content
        user_id: path to the user's folder in the blob storage
    Returns:
        str: Shareable URL to the uploaded file
    """
    try:
        # Get Azure Storage connection
        storage_account_url = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")
        if not storage_account_url:
            raise ValueError("AZURE_STORAGE_ACCOUNT_URL environment variable not set")
        
        # Use managed identity for authentication
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(
            account_url=storage_account_url, 
            credential=credential
        )
        
        # Container for shared conversations
        container_name = "shared-conversations"
        
        # Create container if it doesn't exist
        try:
            blob_service_client.create_container(container_name)
        except Exception:
            pass  # Container might already exist
        
        # Upload the file
        blob_client = blob_service_client.get_blob_client(
            container=container_name, 
            blob=f"{user_id}/{filename}"
        )
        
        blob_client.upload_blob(
            content, 
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type)
        )
        
        # Get a user delegation key to sign the SAS token, valid for 7 days
        delegation_key_start_time = datetime.now(timezone.utc)
        delegation_key_expiry_time = delegation_key_start_time + timedelta(days=7)
        
        user_delegation_key = blob_service_client.get_user_delegation_key(
            key_start_time=delegation_key_start_time,
            key_expiry_time=delegation_key_expiry_time
        )
        
        # Generate a user-delegation SAS token for the blob
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=container_name,
            blob_name=f"{user_id}/{filename}",
            user_delegation_key=user_delegation_key,
            permission=BlobSasPermissions(read=True),
            expiry=delegation_key_expiry_time
        )
        
        # Construct the full URL with SAS token
        blob_url_with_sas = f"{blob_client.url}?{sas_token}"
        
        return blob_url_with_sas
        
    except Exception as e:
        logging.error(f"Error uploading to blob storage: {str(e)}")
        raise

def export_conversation(conversation_id, user_id, export_format="html"):
    """
    Export a conversation to a file and upload to blob storage.
    
    Args:
        conversation_id: ID of the conversation to export
        user_id: ID of the user requesting the export (for security)
        export_format: Format to export ("html", "json")
        
    Returns:
        dict: Export result with URL and metadata
    """
    try:
        # Get conversation data
        conversation_data = get_conversation(conversation_id, user_id)
        
        if not conversation_data:
            raise ValueError("Conversation not found or access denied")
        
        # Add the conversation ID to the data for formatting
        conversation_data['id'] = conversation_id
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_conv_id = conversation_id[:8]  # First 8 chars of conversation ID
        filename = f"conversation_{safe_conv_id}_{timestamp}.{export_format}"
        
        # Format content based on requested format
        if export_format.lower() == "html":
            content = format_conversation_as_html(conversation_data)
            content_type = "text/html"
        elif export_format.lower() == "json":
            content = format_conversation_as_json(conversation_data)
            content_type = "application/json"
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        # Upload to blob storage
        share_url = upload_to_blob_storage(content, filename, user_id, content_type)
        
        return {
            "success": True,
            "share_url": share_url,
            "filename": filename,
            "format": export_format,
            "message_count": len(conversation_data.get('messages', [])),
            "export_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error exporting conversation {conversation_id}: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        } 