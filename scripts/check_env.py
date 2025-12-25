#!/usr/bin/env python3
"""
Environment sanity check for HALO-Agent.

Checks API key availability and optionally pings OpenAI to verify auth.

Usage:
    python scripts/check_env.py
    python scripts/check_env.py --ping
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def check_api_keys() -> dict:
    """Check if API keys are set (without printing values).
    
    Returns:
        Dict with key names and boolean availability
    """
    keys = {
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }
    return keys


def ping_openai() -> tuple:
    """Perform a minimal OpenAI request to verify auth.
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return False, "OPENAI_API_KEY not set"
    
    try:
        from openai import OpenAI, APIError, AuthenticationError, RateLimitError
        
        client = OpenAI()
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'ok'"}],
            max_tokens=1,
            temperature=0
        )
        
        return True, f"OpenAI ping successful (model: gpt-4o-mini)"
        
    except AuthenticationError as e:
        return False, f"Authentication failed (401): Invalid API key. Check your OPENAI_API_KEY."
    except RateLimitError as e:
        return False, f"Rate limit (429): You've exceeded your quota. Check billing at platform.openai.com"
    except APIError as e:
        status = getattr(e, 'status_code', 'unknown')
        return False, f"API error ({status}): {str(e)[:100]}"
    except Exception as e:
        return False, f"Unexpected error: {type(e).__name__}: {str(e)[:100]}"


def main():
    parser = argparse.ArgumentParser(
        description="Check HALO-Agent environment and API keys",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check if keys are set
    python scripts/check_env.py
    
    # Check keys and ping OpenAI
    python scripts/check_env.py --ping
"""
    )
    parser.add_argument(
        "--ping",
        action="store_true",
        help="Perform a minimal OpenAI request to verify authentication"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("HALO-Agent Environment Check")
    print("=" * 50)
    
    keys = check_api_keys()
    
    print("\nAPI Keys:")
    for key_name, is_set in keys.items():
        status = "✓ SET" if is_set else "✗ NOT SET"
        print(f"  {key_name}: {status}")
    
    if not keys["OPENAI_API_KEY"]:
        print("\n⚠ WARNING: OPENAI_API_KEY is required for HALO-Agent to function.")
        print("  Set it in your .env file or environment variables.")
    
    if args.ping:
        print("\nPinging OpenAI...")
        success, message = ping_openai()
        
        if success:
            print(f"  ✓ {message}")
        else:
            print(f"  ✗ {message}")
            sys.exit(1)
    
    print("\n" + "=" * 50)
    
    if keys["OPENAI_API_KEY"]:
        print("Environment OK - ready to run HALO-Agent")
        sys.exit(0)
    else:
        print("Environment NOT ready - missing required API keys")
        sys.exit(1)


if __name__ == "__main__":
    main()
