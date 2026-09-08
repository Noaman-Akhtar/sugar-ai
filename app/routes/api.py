"""
API routes for Sugar-AI.
"""
from fastapi import APIRouter, Depends, HTTPException, Header, Query, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
import time
import logging
import os
import json
import uuid
from datetime import datetime
from typing import Dict, Optional, List

from app.database import get_db, APIKey
from app.ai import RAGAgent
from app.multimodal import ImagePart, ResponsesRequest, normalize_messages
from app.providers.base import (
    GenerationParams,
    UnsupportedModalityError,
    UnsupportedResponseFormatError,
)
from app.config import settings

# Pydantic models for chat completions
class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant" 
    content: str

class PromptedLLMRequest(BaseModel):
    """Request model for ask-llm-prompted endpoint"""
    chat: bool = Field(False, description="Enable chat mode (uses messages instead of question)")
    question: Optional[str] = Field(None, description="The question to ask (required if chat=False)")
    custom_prompt: Optional[str] = Field(None, description="Custom prompt to replace system prompt (required if chat=False)")
    messages: Optional[List[ChatMessage]] = Field(None, description="List of chat messages (required if chat=True)")
    
    # Boundary validation added below:
    max_length: int = Field(1024, gt=0, le=8192, description="Maximum length of generated text")
    truncation: bool = Field(True, description="Whether to truncate input if too long")
    repetition_penalty: float = Field(1.1, gt=0.0, le=2.0, description="Repetition penalty")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature for sampling")
    top_p: float = Field(0.9, gt=0.0, le=1.0, description="Top-p (nucleus) sampling parameter")
    top_k: int = Field(50, ge=0, description="Top-k sampling parameter")

router = APIRouter(tags=["api"])

# setup logging
logger = logging.getLogger("sugar-ai")

# Initialize the agent
agent = None

# user quotas tracking
user_quotas: Dict[str, Dict] = {}

# An image costs more than plain text because vision requests are heavier
# for the provider. The formula is one unit per request plus two per image.
IMAGE_QUOTA_UNITS = 2

def _quota_state(api_key: str) -> Dict:
    """Return today's quota record for a key, resetting it daily."""
    today = datetime.now().date()
    state = user_quotas.get(api_key)
    if state is None or state["date"] != today:
        state = {"count": 0, "date": today}
        user_quotas[api_key] = state
    return state

def consume_quota_units(api_key: str, units: int) -> bool:
    """Charge units against today's quota, all or nothing."""
    state = _quota_state(api_key)
    if state["count"] + units > settings.MAX_DAILY_REQUESTS:
        return False
    state["count"] += units
    return True

def remaining_quota_units(api_key: str) -> int:
    """Return how many units the key can still spend today."""
    return max(settings.MAX_DAILY_REQUESTS - _quota_state(api_key)["count"], 0)

def request_quota_units(request_data: ResponsesRequest) -> int:
    """Return the unit cost of a validated multimodal request."""
    images = sum(
        1
        for message in request_data.messages
        for part in message.content
        if isinstance(part, ImagePart)
    )
    return 1 + IMAGE_QUOTA_UNITS * images

def check_quota(api_key: str) -> bool:
    """Check if a user has exceeded their daily quota"""
    return consume_quota_units(api_key, 1)

def authenticate_api_key(api_key: Optional[str] = Header(None, alias="X-API-Key"), request: Request = None) -> str:
    """Verify the API key without consuming quota; return the key itself."""
    if not api_key:
        logger.warning(f"API key missing: {request.client.host if request else 'unknown'}")
        raise HTTPException(status_code=401, detail="API key is missing")

    if api_key not in settings.API_KEYS:
        logger.warning(f"Invalid API key used: {api_key[:5]}... from {request.client.host if request else 'unknown'}")
        raise HTTPException(status_code=401, detail="Invalid API key")

    return api_key

def verify_api_key(api_key: Optional[str] = Header(None, alias="X-API-Key"), request: Request = None):
    """Verify API key and check quota"""
    api_key = authenticate_api_key(api_key, request)

    if not check_quota(api_key):
        logger.warning(f"Quota exceeded for user: {settings.API_KEYS[api_key]['name']}")
        raise HTTPException(status_code=429, detail="Daily request quota exceeded")

    return settings.API_KEYS[api_key]

@router.post("/ask")
async def ask_question(
    question: str, 
    user_info: dict = Depends(verify_api_key), 
    request: Request = None
):
    """Process a question using RAG pipeline"""
    start_time = time.time()
    
    client_ip = request.client.host if request else "unknown"
    logger.info(f"REQUEST - /ask - User: {user_info['name']} - IP: {client_ip} - Question: {question[:50]}...")
    
    try:
        answer = agent.run(question)
        
        # log completion
        process_time = time.time() - start_time
        logger.info(f"RESPONSE - User: {user_info['name']} - Success - Time: {process_time:.2f}s")
        
        # check quota
        api_key = next(
            key for key, value in settings.API_KEYS.items()
            if value['name'] == user_info['name']
        )
        remaining = (
            settings.MAX_DAILY_REQUESTS
            - user_quotas.get(api_key, {}).get("count", 0)
        )
        
        return {
            "answer": answer, 
            "user": user_info["name"],
            "quota": {"remaining": remaining, "total": settings.MAX_DAILY_REQUESTS}
        }
    except Exception as e:
        logger.error(f"ERROR - User: {user_info['name']} - Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@router.post("/ask-llm")
async def ask_llm(
    question: str, 
    user_info: dict = Depends(verify_api_key), 
    request: Request = None
):
    """Process a question with direct LLM call (no retrieval)"""
    start_time = time.time()
    
    client_ip = request.client.host if request else "unknown"
    logger.info(f"REQUEST - /ask-llm - User: {user_info['name']} - IP: {client_ip} - Question: {question[:50]}...")
    
    try:
        answer = agent.provider.generate(question)
        
        process_time = time.time() - start_time
        logger.info(f"RESPONSE - User: {user_info['name']} - Success - Time: {process_time:.2f}s")
        
        # check quota
        api_key = next(key for key, value in settings.API_KEYS.items() if value['name'] == user_info['name'])
        remaining = settings.MAX_DAILY_REQUESTS - user_quotas.get(api_key, {}).get("count", 0)
        
        return {
            "answer": answer, 
            "user": user_info["name"],
            "quota": {"remaining": remaining, "total": settings.MAX_DAILY_REQUESTS}
        }
    except Exception as e:
        logger.error(f"ERROR - User: {user_info['name']} - Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@router.post("/ask-llm-prompted")
async def ask_llm_prompted(
    request_data: PromptedLLMRequest,
    user_info: dict = Depends(verify_api_key), 
    request: Request = None
):
    """This endpoint lets you ask a question to the model running on Sugar-AI using custom prompts and also provides options to change model parameters to tune the output.
    RAG is disabled for this endpoint. Set chat=True for chat completions mode.
    """
    start_time = time.time()
    client_ip = request.client.host if request else "unknown"
    
    # Check quota first
    api_key = next(key for key, value in settings.API_KEYS.items() if value['name'] == user_info['name'])
    remaining = settings.MAX_DAILY_REQUESTS - user_quotas.get(api_key, {}).get("count", 0)
    
    try:
        if request_data.chat:
            # Chat completions mode
            if not request_data.messages:
                raise HTTPException(status_code=400, detail="messages field is required when chat=True")
            
            # Log the last user message for tracking
            user_messages = [msg for msg in request_data.messages if msg.role == "user"]
            last_user_msg = user_messages[-1].content if user_messages else "No user message"
            logger.info(f"REQUEST - /ask-llm-prompted (chat=True) - User: {user_info['name']} - IP: {client_ip} - Last message: {last_user_msg[:200]}...")
            
            # Log system message if present
            system_messages = [msg for msg in request_data.messages if msg.role == "system"]
            if system_messages:
                logger.info(f"SYSTEM PROMPT - User: {user_info['name']} - Prompt: {system_messages[0].content[:100]}...")
            
            # Convert Pydantic messages to dict format for the agent function
            messages_dict = [{"role": msg.role, "content": msg.content} for msg in request_data.messages]
            
            # Build generation params from request
            params = GenerationParams(
                max_new_tokens=request_data.max_length,
                temperature=request_data.temperature,
                top_p=request_data.top_p,
                top_k=request_data.top_k,
                repetition_penalty=request_data.repetition_penalty,
                truncation=request_data.truncation,
            )

            answer = agent.run_chat_completion(
                messages=messages_dict,
                params=params,
            )
            
            process_time = time.time() - start_time
            logger.info(f"RESPONSE - User: {user_info['name']} - Success - Time: {process_time:.2f}s - Last message: {last_user_msg[:200]}...")
            
            # Return chat format response
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": answer
                    },
                    "index": 0,
                    "finish_reason": "stop"
                }],
                "user": user_info["name"],
                "quota": {"remaining": remaining, "total": settings.MAX_DAILY_REQUESTS},
                "generation_params": {
                    "max_length": request_data.max_length,
                    "truncation": request_data.truncation,
                    "repetition_penalty": request_data.repetition_penalty,
                    "temperature": request_data.temperature,
                    "top_p": request_data.top_p,
                    "top_k": request_data.top_k
                }
            }
        else:
            # Prompted mode
            if not request_data.question or not request_data.custom_prompt:
                raise HTTPException(status_code=400, detail="question and custom_prompt fields are required when chat=False")
            
            logger.info(f"REQUEST - /ask-llm-prompted - User: {user_info['name']} - IP: {client_ip} - Question: {request_data.question[:200]}...")
            logger.info(f"CUSTOM PROMPT - User: {user_info['name']} - Prompt: {request_data.custom_prompt[:100]}...")
            
            params = GenerationParams(
                max_new_tokens=request_data.max_length,
                temperature=request_data.temperature,
                top_p=request_data.top_p,
                top_k=request_data.top_k,
                repetition_penalty=request_data.repetition_penalty,
                truncation=request_data.truncation,
            )

            answer = agent.run_with_custom_prompt(
                question=request_data.question,
                custom_prompt=request_data.custom_prompt,
                params=params,
            )
            
            process_time = time.time() - start_time
            logger.info(f"RESPONSE - User: {user_info['name']} - Success - Time: {process_time:.2f}s")
            
            return {
                "answer": answer, 
                "user": user_info["name"],
                "quota": {"remaining": remaining, "total": settings.MAX_DAILY_REQUESTS},
                "generation_params": {
                    "max_length": request_data.max_length,
                    "truncation": request_data.truncation,
                    "repetition_penalty": request_data.repetition_penalty,
                    "temperature": request_data.temperature,
                    "top_p": request_data.top_p,
                    "top_k": request_data.top_k
                }
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ERROR - User: {user_info['name']} - Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
        
@router.post("/debug")
async def debug(
    code: str, 
    context: bool,
    user_info: dict = Depends(verify_api_key), 
    request: Request = None
):
    """Process python code for debugging"""
    start_time = time.time()
    
    client_ip = request.client.host if request else "unknown"
    logger.info(f"REQUEST - /debug - User: {user_info['name']} - IP: {client_ip} - code: {code[:50]}...")
    
    try:
        response = agent.debug(code, context)
        answer = response
        
        process_time = time.time() - start_time
        logger.info(f"RESPONSE - User: {user_info['name']} - Success - Time: {process_time:.2f}s")
        
        # check quota
        api_key = next(key for key, value in settings.API_KEYS.items() if value['name'] == user_info['name'])
        remaining = settings.MAX_DAILY_REQUESTS - user_quotas.get(api_key, {}).get("count", 0)
        
        return {
            "answer": answer, 
            "user": user_info["name"],
            "quota": {"remaining": remaining, "total": settings.MAX_DAILY_REQUESTS}
        }
        
    except Exception as e:
        logger.error(f"ERROR - User: {user_info['name']} - Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@router.post("/change-model")
async def change_model(
    model: str, 
    api_key: str = Query(...), 
    password: str = Query(...), 
    request: Request = None
):
    """Change the model used by the RAG agent (admin only)"""
    client_ip = request.client.host if request else "unknown"
    logger.info(f"REQUEST - /change-model - API Key: {api_key[:5]}... - IP: {client_ip} - Model: {model}")
    
    if api_key not in settings.API_KEYS:
        logger.warning(f"Invalid API key used for model change: {api_key[:5]}... from {client_ip}")
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    user_info = settings.API_KEYS[api_key]
    if not user_info.get("can_change_model", False):
        logger.warning(f"Unauthorized model change attempt by: {user_info['name']} from {client_ip}")
        raise HTTPException(status_code=403, detail="User doesn't have permission to change model")
    
    if password != settings.MODEL_CHANGE_PASSWORD:
        logger.warning(f"Invalid password for model change by: {user_info['name']} from {client_ip}")
        raise HTTPException(status_code=403, detail="Invalid model change password")
    
    try:
        from app.providers import create_provider
        from app.config import settings
        new_provider = create_provider(
            provider_name=settings.AI_PROVIDER,
            model_name=model,
            quantize=True,
            dev_mode=False,
            base_url=settings.OLLAMA_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
            openai_base_url=settings.OPENAI_BASE_URL,
            gemini_api_key=settings.GEMINI_API_KEY,
            gemini_base_url=settings.GEMINI_BASE_URL,
        )
        agent.set_model(new_provider)
        logger.info(f"Model changed to {model} by {user_info['name']}")
        return {"message": f"Model changed to {model}", "user": user_info["name"]}
    except Exception as e:
        logger.error(f"Error changing model to {model} by {user_info['name']}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error changing model: {str(e)}")


@router.post("/v1/responses")
async def create_response(
    request_data: ResponsesRequest,
    api_key: str = Depends(authenticate_api_key),
    request: Request = None,
):
    """Generate a response from typed text and image content parts.

    The versioned contract for activities: validation happens before any
    quota is spent, capability failures are clear client errors, and media
    bytes never reach the logs.
    """
    client_ip = request.client.host if request else "unknown"
    user_name = settings.API_KEYS[api_key]["name"]
    image_count = sum(
        1
        for message in request_data.messages
        for part in message.content
        if isinstance(part, ImagePart)
    )
    logger.info(
        f"REQUEST - /v1/responses - User: {user_name} - IP: {client_ip} - "
        f"Messages: {len(request_data.messages)} - Images: {image_count} - "
        f"Format: {request_data.response_format}"
    )

    provider = agent.provider
    required_modalities = {"text"}
    if image_count:
        required_modalities.add("image")

    # Refuse impossible requests before spending any quota.
    if not provider.supports_input_modalities(required_modalities):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unsupported_modality",
                "message": "The configured provider does not accept image input",
            },
        )
    if not provider.supports_response_format(request_data.response_format):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unsupported_response_format",
                "message": (
                    "The configured provider does not support "
                    f"{request_data.response_format} responses"
                ),
            },
        )

    units = request_quota_units(request_data)
    if not consume_quota_units(api_key, units):
        raise HTTPException(
            status_code=429,
            detail={
                "code": "insufficient_quota",
                "message": "Daily request quota exceeded",
            },
        )

    generation = request_data.generation
    params_kwargs = {
        "max_new_tokens": generation.max_new_tokens,
        "top_p": generation.top_p,
        "top_k": generation.top_k,
        "repetition_penalty": generation.repetition_penalty,
        "truncation": generation.truncation,
    }
    if generation.temperature is not None:
        params_kwargs["temperature"] = generation.temperature
    params = GenerationParams(**params_kwargs)

    start_time = time.time()
    try:
        result = agent.run_multimodal(
            normalize_messages(request_data),
            params=params,
            response_format=request_data.response_format,
            retrieval=request_data.retrieval,
        )
    except UnsupportedModalityError as error:
        raise HTTPException(
            status_code=422,
            detail={"code": "unsupported_modality", "message": str(error)},
        )
    except UnsupportedResponseFormatError as error:
        raise HTTPException(
            status_code=422,
            detail={"code": "unsupported_response_format", "message": str(error)},
        )
    except Exception as error:
        logger.error(f"ERROR - /v1/responses - User: {user_name} - Error: {str(error)}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "provider_error",
                "message": "Error generating response",
            },
        )

    process_time = time.time() - start_time
    logger.info(
        f"RESPONSE - /v1/responses - User: {user_name} - Status: {result.status} - "
        f"Time: {process_time:.2f}s"
    )

    if request_data.response_format == "json_object":
        try:
            parsed = json.loads(result.text)
        except json.JSONDecodeError:
            parsed = None
        if not isinstance(parsed, dict):
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "invalid_provider_output",
                    "message": "The provider did not return a valid JSON object",
                },
            )
        output = [{"type": "json", "json": parsed}]
    else:
        output = [{"type": "text", "text": result.text}]

    body = {
        "id": f"resp_{uuid.uuid4().hex}",
        "status": result.status,
        "output": output,
        "quota": {
            "used_units": units,
            "remaining_units": remaining_quota_units(api_key),
            "daily_limit_units": settings.MAX_DAILY_REQUESTS,
        },
    }
    if result.status == "incomplete":
        body["incomplete_reason"] = "output_limit"
    return body


@router.get("/health")
async def health_check():
    """Check if the AI backend is alive and responsive."""
    if agent is None:
        return {"status": "unavailable", "detail": "Agent not initialized"}

    try:
        model_name = agent.provider.get_model_name()
        is_healthy = agent.provider.health_check()

        if is_healthy:
            return {
                "status": "healthy",
                "provider": type(agent.provider).__name__,
                "model": model_name,
            }
        else:
            return {
                "status": "unhealthy",
                "provider": type(agent.provider).__name__,
                "model": model_name,
                "detail": "Health check failed",
            }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "error",
            "detail": str(e),
        }
