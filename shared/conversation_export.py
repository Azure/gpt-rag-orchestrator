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
            * {{
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 900px;
                margin: 0 auto;
                padding: 32px;
                background-color: #fafbfc;
                color: #1f2937;
                line-height: 1.6;
                min-height: 100vh;
            }}
            
            .conversation-header {{
                background: white;
                padding: 32px;
                border-radius: 16px;
                margin-bottom: 32px;
                border: 1px solid #e5e7eb;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            }}
            
            .conversation-header h1 {{
                margin: 0 0 20px 0;
                font-size: 28px;
                font-weight: 600;
                color: #111827;
                letter-spacing: -0.025em;
            }}
            
            .conversation-header p {{
                margin: 8px 0;
                color: #6b7280;
                font-size: 15px;
                font-weight: 500;
            }}
            
            .conversation-header strong {{
                color: #374151;
                font-weight: 600;
            }}
            
            .message {{
                background: white;
                margin: 20px 0;
                padding: 24px;
                border-radius: 12px;
                border: 1px solid #e5e7eb;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
                transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            }}
            
            .message:hover {{
                border-color: #d1d5db;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
                transform: translateY(-1px);
            }}
            
            .user-message {{
                background: #f8fafc;
                border-left: 3px solid #3b82f6;
                margin-left: 32px;
            }}
            
            .freddaid-message {{
                background: #f9fafb;
                border-left: 3px solid #6b7280;
                margin-right: 32px;
            }}
            
            .role {{
                font-weight: 600;
                font-size: 13px;
                margin-bottom: 12px;
                display: flex;
                align-items: center;
                gap: 8px;
                letter-spacing: 0.025em;
                color: #374151;
                text-transform: uppercase;
            }}
            
            .role::before {{
                content: '';
                width: 6px;
                height: 6px;
                border-radius: 50%;
                background: #9ca3af;
            }}
            
            .user-message .role::before {{
                background: #3b82f6;
            }}
            
            .freddaid-message .role::before {{
                background: #6b7280;
            }}
            
            .content {{
                line-height: 1.7;
                white-space: pre-wrap;
                color: #1f2937;
                font-size: 15px;
                font-weight: 400;
            }}
            
            .timestamp {{
                color: #9ca3af;
                font-size: 12px;
                margin-top: 16px;
                font-weight: 500;
            }}
            
            .export-info {{
                background: #f3f4f6;
                padding: 16px 20px;
                border-radius: 12px;
                margin-bottom: 32px;
                font-size: 14px;
                color: #4b5563;
                border: 1px solid #e5e7eb;
                display: flex;
                align-items: center;
                gap: 8px;
                font-weight: 500;
            }}
            
            .messages {{
                animation: fadeIn 0.6s ease-out;
            }}
            
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            
            /* Clean scrollbar */
            ::-webkit-scrollbar {{
                width: 6px;
            }}
            
            ::-webkit-scrollbar-track {{
                background: #f1f5f9;
            }}
            
            ::-webkit-scrollbar-thumb {{
                background: #cbd5e1;
                border-radius: 3px;
            }}
            
            ::-webkit-scrollbar-thumb:hover {{
                background: #94a3b8;
            }}
            
            @media (max-width: 768px) {{
                body {{
                    padding: 20px;
                }}
                
                .conversation-header {{
                    padding: 24px;
                }}
                
                .message {{
                    padding: 20px;
                }}
                
                .user-message {{
                    margin-left: 16px;
                }}
                
                .freddaid-message {{
                    margin-right: 16px;
                }}
            }}
            
            @media (max-width: 480px) {{
                .user-message {{
                    margin-left: 8px;
                }}
                
                .freddaid-message {{
                    margin-right: 8px;
                }}
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
        role_display = 'User' if role == 'user' else 'Freddaid'
        
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