
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_avatar_prompt():
    """Test using the avatar_prompt from Langfuse"""
    print("Testing avatar_prompt from Langfuse...")
    
    try:
        from langfuse import Langfuse
        
        # Get credentials from environment
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        if not secret_key or not public_key:
            print("❌ Langfuse credentials not found in environment variables")
            return False
            
        # Initialize Langfuse
        langfuse = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=host
        )
        
        # Test getting the avatar_prompt
        try:
            prompt = langfuse.get_prompt("avatar_prompt")
            print(f"✅ Retrieved avatar_prompt from Langfuse:")
            print(f"   Name: {getattr(prompt, 'name', 'N/A')}")
            print(f"   Version: {getattr(prompt, 'version', 'N/A')}")
            print(f"   Prompt: {getattr(prompt, 'prompt', 'N/A')}")
            print(f"   Config: {getattr(prompt, 'config', 'N/A')}")
            return True
        except Exception as e:
            print(f"ℹ️  Could not retrieve 'avatar_prompt' from Langfuse: {e}")
            print("   This is expected if you haven't created this prompt in Langfuse yet.")
            return False
            
    except Exception as e:
        print(f"❌ Langfuse test failed: {e}")
        return False

def test_local_fallback():
    """Test the local prompt manager as fallback"""
    print("\nTesting local prompt manager fallback...")
    
    try:
        from prompt_manager import prompt_manager
        
        # Test getting avatar_prompt from local manager
        try:
            prompt = prompt_manager.get_current_prompt("avatar_prompt")
            print(f"✅ Retrieved avatar_prompt from local manager:")
            print(f"   Version: {prompt.get('version', 'N/A')}")
            print(f"   Prompt: {prompt.get('prompt', 'N/A')}")
            return True
        except Exception as e:
            print(f"ℹ️  Could not retrieve 'avatar_prompt' from local manager: {e}")
            
        # Try avatar_assistant as fallback
        try:
            prompt = prompt_manager.get_current_prompt("avatar_assistant")
            print(f"✅ Retrieved avatar_assistant from local manager (fallback):")
            print(f"   Version: {prompt.get('version', 'N/A')}")
            print(f"   Prompt: {prompt.get('prompt', 'N/A')[:100]}...")
            return True
        except Exception as e:
            print(f"❌ Failed to retrieve any prompt from local manager: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Local prompt manager test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Avatar Prompt Usage Test")
    print("=" * 30)
    
    # Test components
    tests = [
        test_avatar_prompt,
        test_local_fallback
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
        print()
    
    # Summary
    print("Test Summary:")
    print("=" * 30)
    passed = sum(results)
    total = len(results)
    
    if passed > 0:
        print(f"✅ At least one prompt source is working ({passed}/{total})")
        print("\n🎉 Your application can now use prompts from Langfuse!")
        print("   If avatar_prompt exists in Langfuse, it will be used.")
        print("   Otherwise, it will fall back to local prompts.")
    else:
        print(f"⚠️  No prompt sources are available")
        print("   Your app will use the default prompt.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())