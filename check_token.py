import requests
import os

def check_hf_token(token):
    """Check if HuggingFace token is valid and get user info"""
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        # Check token validity
        response = requests.get("https://huggingface.co/api/whoami", headers=headers)
        
        if response.status_code == 200:
            user_info = response.json()
            print("✅ Token is VALID!")
            print(f"Username: {user_info.get('name', 'N/A')}")
            print(f"Email: {user_info.get('email', 'N/A')}")
            print(f"Account Type: {user_info.get('type', 'N/A')}")
            return True
        elif response.status_code == 401:
            print("❌ Token is INVALID or EXPIRED!")
            print("Please generate a new token at: https://huggingface.co/settings/tokens")
            return False
        else:
            print(f"❌ Error checking token: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    # Get token from environment variable
    HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    if not HF_TOKEN:
        print("❌ No token found! Please set the HUGGINGFACE_HUB_TOKEN environment variable.")
        print("Example: export HUGGINGFACE_HUB_TOKEN='your_token_here'")
        exit(1)
    
    print("Checking HuggingFace token...")
    check_hf_token(HF_TOKEN)