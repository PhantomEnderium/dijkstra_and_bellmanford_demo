import os
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

def test_discord_webhook():
    """Test sending a simple message to a Discord webhook."""
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("✗ No webhook URL provided in .env file")
        return False

    try:
        payload = {'content': 'Test message from webhook tester.'}
        response = requests.post(webhook_url, json=payload)

        if response.status_code in [200, 204]:
            print("✓ Test message sent successfully!")
            return True
        else:
            print(f"✗ Failed to send test message: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error sending test message: {e}")
        return False

if __name__ == "__main__":
    test_discord_webhook()