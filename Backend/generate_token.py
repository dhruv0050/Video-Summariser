"""
Script to generate a new Google Drive OAuth refresh token
Run this once to get a new REFRESH_TOKEN for your .env file
"""

from google_auth_oauthlib.flow import InstalledAppFlow
import os
from dotenv import load_dotenv

load_dotenv()

# Define the scopes needed
SCOPES = [
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/drive.metadata.readonly'
]

def generate_refresh_token():
    """Generate a new refresh token"""
    
    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    
    if not client_id or not client_secret:
        print("❌ CLIENT_ID or CLIENT_SECRET not found in .env")
        return
    
    # Create OAuth client config
    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uris": ["http://localhost:5173/", "urn:ietf:wg:oauth:2.0:oob"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token"
        }
    }
    
    print("\n🔐 Starting OAuth flow...\n")
    print("📋 Requesting these scopes:")
    for scope in SCOPES:
        print(f"   - {scope}")
    
    # Run the OAuth flow
    flow = InstalledAppFlow.from_client_config(
        client_config,
        scopes=SCOPES
    )
    
    print("\n🌐 Opening browser for authentication...")
    print("   Please sign in with the Google account that owns your Drive videos")
    print("   Or the account you want to grant access to\n")
    
    creds = flow.run_local_server(port=8080, open_browser=True)
    
    if creds and creds.refresh_token:
        print("\n✅ Success! Here's your new REFRESH_TOKEN:\n")
        print("="*70)
        print(creds.refresh_token)
        print("="*70)
        print("\n📝 Update your .env file:")
        print(f"\n   REFRESH_TOKEN={creds.refresh_token}\n")
        print("💾 Then restart the backend server.")
    else:
        print("\n❌ Failed to get refresh token")
        print("   The token might have been generated before.")
        print("   Try revoking access at: https://myaccount.google.com/permissions")
        print("   Then run this script again.")

if __name__ == "__main__":
    try:
        generate_refresh_token()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure CLIENT_ID and CLIENT_SECRET are correct in .env")
        print("2. Check authorized redirect URIs include http://localhost:8080/")
        print("3. Enable Google Drive API in Google Cloud Console")
