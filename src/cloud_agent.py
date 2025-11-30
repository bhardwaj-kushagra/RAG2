#!/usr/bin/env python3
"""
Cloud Agent module for delegating LLM tasks to cloud-based services.

This module provides an interface to delegate generation tasks from
local llama.cpp to cloud-based LLM providers (e.g., OpenAI API).

Supported providers:
- OpenAI (GPT-3.5, GPT-4)
- Custom API endpoints (OpenAI-compatible)

Usage:
  from cloud_agent import CloudAgent, CloudAgentConfig
  
  config = CloudAgentConfig(
      provider="openai",
      api_key="your-api-key",
      model="gpt-3.5-turbo"
  )
  agent = CloudAgent(config)
  response = agent.generate(prompt, max_tokens=512)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Optional: httpx for API calls (lighter than requests)
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# Optional: openai library
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class CloudAgentConfig:
    """Configuration for cloud agent delegation."""
    
    provider: str = "openai"  # "openai" or "custom"
    api_key: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    base_url: Optional[str] = None  # For custom API endpoints
    temperature: float = 0.2
    top_p: float = 0.9
    timeout: float = 60.0
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Try to get API key from environment if not provided
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")


class CloudAgentError(Exception):
    """Base exception for cloud agent errors."""
    pass


class CloudAgentConfigError(CloudAgentError):
    """Configuration error for cloud agent."""
    pass


class CloudAgentAPIError(CloudAgentError):
    """API error from cloud provider."""
    pass


class CloudAgent:
    """
    Cloud Agent for delegating LLM generation to cloud-based services.
    
    This agent provides a fallback mechanism when local LLM generation
    is not available or when cloud-based generation is preferred.
    """
    
    def __init__(self, config: CloudAgentConfig):
        """
        Initialize the cloud agent with configuration.
        
        Args:
            config: CloudAgentConfig instance with provider settings
        """
        self.config = config
        self._validate_config()
        self._client = None
        
    def _validate_config(self) -> None:
        """Validate configuration settings."""
        if self.config.provider not in ("openai", "custom"):
            raise CloudAgentConfigError(
                f"Unsupported provider: {self.config.provider}. "
                "Supported providers: openai, custom"
            )
        
        if self.config.provider == "custom" and not self.config.base_url:
            raise CloudAgentConfigError(
                "base_url is required for custom provider"
            )
            
    def _ensure_client(self) -> None:
        """Ensure API client is initialized."""
        if self._client is not None:
            return
            
        if self.config.provider == "openai":
            if OPENAI_AVAILABLE:
                self._client = openai.OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout
                )
            elif HTTPX_AVAILABLE:
                self._client = "httpx"
            else:
                raise CloudAgentConfigError(
                    "Neither 'openai' nor 'httpx' package is installed. "
                    "Install with: pip install openai  or  pip install httpx"
                )
        elif self.config.provider == "custom":
            if not HTTPX_AVAILABLE:
                raise CloudAgentConfigError(
                    "httpx package is required for custom provider. "
                    "Install with: pip install httpx"
                )
            self._client = "httpx"
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        """
        Generate text using the cloud LLM provider.
        
        Args:
            prompt: The input prompt for generation
            max_tokens: Maximum tokens to generate
            stop: List of stop sequences
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated text response
            
        Raises:
            CloudAgentError: If generation fails
        """
        self._ensure_client()
        
        if self.config.api_key is None:
            raise CloudAgentConfigError(
                "API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key in CloudAgentConfig."
            )
        
        try:
            if OPENAI_AVAILABLE and isinstance(self._client, openai.OpenAI):
                return self._generate_openai(prompt, max_tokens, stop, **kwargs)
            else:
                return self._generate_httpx(prompt, max_tokens, stop, **kwargs)
        except Exception as e:
            raise CloudAgentAPIError(f"Cloud generation failed: {e}") from e
    
    def _generate_openai(
        self,
        prompt: str,
        max_tokens: int,
        stop: Optional[List[str]],
        **kwargs: Any
    ) -> str:
        """Generate using the openai library."""
        messages = [{"role": "user", "content": prompt}]
        
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stop=stop,
            **{**self.config.extra_params, **kwargs}
        )
        
        if not response.choices:
            raise CloudAgentAPIError("No choices returned from API response")
        
        return response.choices[0].message.content or ""
    
    def _generate_httpx(
        self,
        prompt: str,
        max_tokens: int,
        stop: Optional[List[str]],
        **kwargs: Any
    ) -> str:
        """Generate using httpx for API calls."""
        base_url = self.config.base_url or "https://api.openai.com/v1"
        url = f"{base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            **self.config.extra_params,
            **kwargs
        }
        
        if stop:
            payload["stop"] = stop
        
        with httpx.Client(timeout=self.config.timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                error_detail = ""
                try:
                    error_detail = response.text
                except Exception:
                    pass
                raise CloudAgentAPIError(
                    f"API request failed with status {response.status_code}: {error_detail}"
                ) from e
            data = response.json()
        
        # Validate response structure
        if not isinstance(data, dict) or "choices" not in data:
            raise CloudAgentAPIError("Invalid API response format: missing 'choices'")
        if not data["choices"]:
            raise CloudAgentAPIError("No choices returned from API response")
        
        choice = data["choices"][0]
        if not isinstance(choice, dict) or "message" not in choice:
            raise CloudAgentAPIError("Invalid API response format: missing 'message' in choice")
        
        return choice["message"].get("content", "") or ""
    
    def is_available(self) -> bool:
        """
        Check if cloud agent is available and properly configured.
        
        Returns:
            True if the agent can be used for generation
        """
        try:
            self._validate_config()
            if self.config.api_key is None:
                return False
            return OPENAI_AVAILABLE or HTTPX_AVAILABLE
        except CloudAgentConfigError:
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status information about the cloud agent.
        
        Returns:
            Dictionary with status information
        """
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "api_key_set": self.config.api_key is not None,
            "openai_available": OPENAI_AVAILABLE,
            "httpx_available": HTTPX_AVAILABLE,
            "is_available": self.is_available()
        }


def create_cloud_agent_from_env() -> Optional[CloudAgent]:
    """
    Create a CloudAgent from environment variables.
    
    Environment variables:
        OPENAI_API_KEY: API key for OpenAI
        CLOUD_AGENT_MODEL: Model to use (default: gpt-3.5-turbo)
        CLOUD_AGENT_BASE_URL: Custom API base URL (optional)
        
    Returns:
        CloudAgent instance if configured, None otherwise
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    
    model = os.environ.get("CLOUD_AGENT_MODEL", "gpt-3.5-turbo")
    base_url = os.environ.get("CLOUD_AGENT_BASE_URL")
    
    config = CloudAgentConfig(
        provider="custom" if base_url else "openai",
        api_key=api_key,
        model=model,
        base_url=base_url
    )
    
    return CloudAgent(config)


if __name__ == "__main__":
    # Simple test/demo
    print("Cloud Agent Module")
    print("=" * 40)
    
    config = CloudAgentConfig()
    agent = CloudAgent(config)
    status = agent.get_status()
    
    print(f"Provider: {status['provider']}")
    print(f"Model: {status['model']}")
    print(f"API Key Set: {status['api_key_set']}")
    print(f"OpenAI Library: {status['openai_available']}")
    print(f"HTTPX Library: {status['httpx_available']}")
    print(f"Available: {status['is_available']}")
    
    if agent.is_available():
        print("\nCloud agent is ready for use!")
    else:
        print("\nCloud agent is not available.")
        print("To enable, set OPENAI_API_KEY environment variable.")
