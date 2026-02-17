import io
import os
import time
import random
from typing import Optional, BinaryIO
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from googleapiclient.errors import HttpError
import config


class GoogleDriveService:
    def __init__(self):
        self.creds = None
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate using refresh token"""
        self.creds = Credentials(
            token=None,
            refresh_token=config.REFRESH_TOKEN,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=config.CLIENT_ID,
            client_secret=config.CLIENT_SECRET,
        )
        
        # Refresh the token
        if self.creds and self.creds.expired and self.creds.refresh_token:
            self.creds.refresh(Request())
        
        self.service = build('drive', 'v3', credentials=self.creds)
    
    def extract_file_id(self, drive_url: str) -> str:
        """Extract file ID from Google Drive URL"""
        if "drive.google.com" in drive_url:
            if "/file/d/" in drive_url:
                file_id = drive_url.split("/file/d/")[1].split("/")[0]
            elif "id=" in drive_url:
                file_id = drive_url.split("id=")[1].split("&")[0]
            else:
                raise ValueError("Invalid Drive URL format")
            return file_id
        else:
            # Assume it's already a file ID
            return drive_url
    
    def download_file(self, file_id: str, destination_path: str) -> str:
        """Download file from Google Drive"""
        try:
            request = self.service.files().get_media(
                fileId=file_id,
                supportsAllDrives=True
            )
            
            with open(destination_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        print(f"Download progress: {int(status.progress() * 100)}%")
            
            return destination_path
        except HttpError as error:
            raise Exception(f"Failed to download file: {error}")
    
    def get_file_metadata(self, file_id: str) -> dict:
        """Get file metadata from Google Drive"""
        try:
            file = self.service.files().get(
                fileId=file_id,
                fields="id, name, size, mimeType, createdTime, modifiedTime"
            ).execute()
            return file
        except HttpError as error:
            # If 404, try with supportsAllDrives for shared files
            if error.resp.status == 404:
                try:
                    file = self.service.files().get(
                        fileId=file_id,
                        fields="id, name, size, mimeType, createdTime, modifiedTime",
                        supportsAllDrives=True
                    ).execute()
                    return file
                except HttpError:
                    pass
            raise Exception(f"Failed to get file metadata: {error}")
    
    def create_folder(self, folder_name: str, parent_folder_id: Optional[str] = None) -> str:
        """Create a folder in Google Drive"""
        try:
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            if parent_folder_id:
                file_metadata['parents'] = [parent_folder_id]
            
            folder = self.service.files().create(
                body=file_metadata,
                fields='id'
            ).execute()
            
            return folder.get('id')
        except HttpError as error:
            raise Exception(f"Failed to create folder: {error}")
    
    def upload_file(
        self, 
        file_path: str, 
        folder_id: Optional[str] = None,
        file_name: Optional[str] = None
    ) -> dict:
        """Upload file to Google Drive with retry logic"""
        # Basic rate limiting
        time.sleep(0.5)
        
        if file_name is None:
            file_name = os.path.basename(file_path)
            
        file_metadata = {'name': file_name}
        if folder_id:
            file_metadata['parents'] = [folder_id]
            
        max_retries = 5
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                media = MediaFileUpload(file_path, resumable=True)
                file = self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id, name, webViewLink, webContentLink'
                ).execute()
                
                # Make file accessible
                self._set_file_permission(file.get('id', ''))
                
                return file
                
            except Exception as e:
                # Catch both HttpError and network errors (like WinError 10054)
                if attempt == max_retries - 1:
                    print(f"Failed to upload {file_name} after {max_retries} attempts. Last error: {e}")
                    raise Exception(f"Failed to upload file after retries: {e}")
                
                delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                print(f"Upload error for {file_name}: {str(e)[:100]}... Retrying in {delay:.2f}s (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                
                # Re-authenticate on token expiry error
                if "invalid_grant" in str(e) or "unauthorized" in str(e).lower():
                    print("Refreshing credentials...")
                    try:
                        self._authenticate()
                    except:
                        pass
    
    def _set_file_permission(self, file_id: str):
        """Set file permission to allow access"""
        try:
            permission = {
                'type': 'anyone',
                'role': 'reader'
            }
            self.service.permissions().create(
                fileId=file_id,
                body=permission
            ).execute()
        except Exception as error:
            print(f"Warning: Failed to set permission: {error}")
    
    def get_file_url(self, file_id: str) -> str:
        """Get shareable URL for file"""
        return f"https://drive.google.com/file/d/{file_id}/view"
    
    def delete_file(self, file_id: str):
        """Delete file from Google Drive"""
        try:
            self.service.files().delete(fileId=file_id).execute()
        except HttpError as error:
            print(f"Warning: Failed to delete file: {error}")


# Singleton instance
drive_service = GoogleDriveService()
