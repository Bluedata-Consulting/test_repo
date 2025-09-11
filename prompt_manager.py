import json
import os
from typing import Dict, Any
from datetime import datetime

# Langfuse tracing
try:
    from langfuse import Langfuse
    from langfuse import observe
except ImportError:
    class Langfuse:
        pass
    
    def observe():
        def decorator(func):
            return func
        return decorator

class PromptManager:
    def __init__(self, langfuse_client=None):
        self.langfuse = langfuse_client
        self.prompts_file = "prompts.json"
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from local file"""
        if os.path.exists(self.prompts_file):
            try:
                with open(self.prompts_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_prompts(self):
        """Save prompts to local file"""
        with open(self.prompts_file, 'w') as f:
            json.dump(self.prompts, f, indent=2)
    
    @observe()
    def create_prompt(self, name: str, prompt: str, tags: list = None, metadata: dict = None, config: dict = None):
        """Create a new prompt version"""
        if name not in self.prompts:
            self.prompts[name] = {
                "name": name,
                "versions": [],
                "current_version": None
            }
        
        version_info = {
            "version": len(self.prompts[name]["versions"]) + 1,
            "prompt": prompt,
            "created_at": datetime.now().isoformat(),
            "tags": tags or [],
            "metadata": metadata or {},
            "config": config or {"temperature": 0.7}
        }
        
        self.prompts[name]["versions"].append(version_info)
        self.prompts[name]["current_version"] = version_info["version"]
        
        # Save to local file
        self._save_prompts()
        
        # If Langfuse is available, also save there
        if self.langfuse:
            try:
                self.langfuse.create_prompt(
                    name=name,
                    prompt=prompt,
                    config=version_info["config"],
                    tags=tags or [],
                    metadata=metadata or {}
                )
            except Exception as e:
                print(f"Warning: Could not save prompt to Langfuse: {e}")
        
        return version_info
    
    @observe()
    def get_prompt(self, name: str, version: int = None):
        """Get a prompt by name and optional version"""
        # First try to get from Langfuse if available
        if self.langfuse:
            try:
                langfuse_prompt = self.langfuse.get_prompt(name, version=version)
                return {
                    "name": name,
                    "version": getattr(langfuse_prompt, 'version', None),
                    "prompt": getattr(langfuse_prompt, 'prompt', ''),
                    "config": getattr(langfuse_prompt, 'config', {}),
                    "tags": getattr(langfuse_prompt, 'tags', []),
                    "metadata": {}
                }
            except Exception as e:
                print(f"Could not fetch prompt from Langfuse: {e}")
        
        # Fall back to local prompts
        if name not in self.prompts:
            raise ValueError(f"Prompt '{name}' not found")
        
        prompt_data = self.prompts[name]
        
        if version is None:
            version = prompt_data["current_version"]
        
        for ver in prompt_data["versions"]:
            if ver["version"] == version:
                return ver
        
        raise ValueError(f"Version {version} not found for prompt '{name}'")
    
    @observe()
    def get_current_prompt(self, name: str):
        """Get the current version of a prompt"""
        return self.get_prompt(name)
    
    @observe()
    def set_current_version(self, name: str, version: int):
        """Set the current version for a prompt (local only)"""
        if name not in self.prompts:
            raise ValueError(f"Prompt '{name}' not found")
        
        # Verify version exists
        version_exists = any(v["version"] == version for v in self.prompts[name]["versions"])
        if not version_exists:
            raise ValueError(f"Version {version} not found for prompt '{name}'")
        
        self.prompts[name]["current_version"] = version
        self._save_prompts()
    
    def list_prompts(self):
        """List all available prompts"""
        # Get prompts from Langfuse if available
        langfuse_prompts = []
        if self.langfuse:
            try:
                # Langfuse doesn't have a direct list prompts method, so we'll just use local ones
                pass
            except Exception as e:
                print(f"Could not fetch prompts from Langfuse: {e}")
        
        # Combine with local prompts
        local_prompts = list(self.prompts.keys())
        return list(set(local_prompts + langfuse_prompts))
    
    @observe()
    def list_versions(self, name: str):
        """List all versions of a prompt"""
        # Try Langfuse first
        if self.langfuse:
            try:
                # Langfuse handles versions internally, we'll just return info about the prompt
                langfuse_prompt = self.langfuse.get_prompt(name)
                return [{
                    "version": getattr(langfuse_prompt, 'version', 'latest'),
                    "prompt": getattr(langfuse_prompt, 'prompt', ''),
                    "config": getattr(langfuse_prompt, 'config', {}),
                    "tags": getattr(langfuse_prompt, 'tags', [])
                }]
            except Exception as e:
                print(f"Could not fetch prompt versions from Langfuse: {e}")
        
        # Fall back to local prompts
        if name not in self.prompts:
            raise ValueError(f"Prompt '{name}' not found")
        return self.prompts[name]["versions"]
    
    @observe()
    def get_langfuse_prompt(self, name: str, version: int = None):
        """Get a prompt directly from Langfuse"""
        if not self.langfuse:
            raise ValueError("Langfuse client not initialized")
        
        try:
            return self.langfuse.get_prompt(name, version=version)
        except Exception as e:
            raise ValueError(f"Could not fetch prompt from Langfuse: {e}")

# Initialize the prompt manager with Langfuse client
langfuse_client = None
try:
    from dotenv import load_dotenv
    load_dotenv()
    
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    if LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY:
        langfuse_client = Langfuse(
            secret_key=LANGFUSE_SECRET_KEY,
            public_key=LANGFUSE_PUBLIC_KEY,
            host=LANGFUSE_HOST
        )
except Exception as e:
    print(f"Could not initialize Langfuse client: {e}")

prompt_manager = PromptManager(langfuse_client)

# Create default prompts if they don't exist
if "avatar_assistant" not in prompt_manager.list_prompts():
    prompt_manager.create_prompt(
        name="avatar_assistant",
        prompt="You are a helpful assistant, your name is BlueAssistant. Keep your responses to a maximum of three sentences.",
        tags=["avatar", "assistant"],
        metadata={"created_by": "system", "purpose": "main assistant prompt"}
    )
