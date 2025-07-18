import requests

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
    # Your new token - replace with your actual token
    HF_TOKEN = "hf_your_new_token_here"  # Replace this with your actual token from HuggingFace
    
    print("Checking HuggingFace token...")
    check_hf_token(HF_TOKEN)