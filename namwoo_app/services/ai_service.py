# NAMWOO/services/ai_service.py
# -*- coding: utf-8 -*-
import logging
import json
from typing import Dict, Any, Optional

from ..config import Config
from . import support_board_service

# --- START OF MODIFICATION ---
# CORRECTED IMPORT PATHS:
# We are in the 'services' package. 'utils' is a sibling.
# So we must go up one level ('..') to the parent 'namwoo_app'
# and then down into the 'utils' package.
from ..utils import conversation_details
# --- END OF MODIFICATION ---

# Import the provider modules
from .providers import openai_chat_provider
from .providers import openai_assistant_provider
from .providers import google_gemini_provider

logger = logging.getLogger(__name__)

# --- Provider Factory ---

def get_ai_provider():
    """
    Factory function to read the config and instantiate the correct AI provider.
    This is the core of the dynamic switching mechanism.
    """
    # NOTE: The code expects the environment variable to be named 'AI_PROVIDER'.
    # Please ensure your .env file uses this key, not 'LLM_PROVIDER'.
    provider_name = getattr(Config, "AI_PROVIDER", "openai_chat").lower()
    logger.info(f"AI Provider selected: '{provider_name}'")

    # This function uses Config values, which are loaded correctly. No changes needed here.
    if provider_name == "openai_assistant":
        if not Config.OPENAI_ASSISTANT_ID:
            raise ValueError("AI_PROVIDER is 'openai_assistant', but OPENAI_ASSISTANT_ID is not set.")
        return openai_assistant_provider.OpenAIAssistantProvider(
            api_key=Config.OPENAI_API_KEY,
            assistant_id=Config.OPENAI_ASSISTANT_ID
        )
    elif provider_name == "google_gemini":
        return google_gemini_provider.GoogleGeminiProvider(
            api_key=Config.GOOGLE_API_KEY
        )
    elif provider_name == "openai_chat":
        return openai_chat_provider.OpenAIChatProvider(
            api_key=Config.OPENAI_API_KEY
        )
    else:
        raise ValueError(f"Unsupported AI_PROVIDER configured: '{provider_name}'")


# --- Main Application Entry Point ---

def process_new_message(
    sb_conversation_id: str,
    new_user_message: Optional[str],
    conversation_source: Optional[str],
    sender_user_id: str,
    customer_user_id: str,
    triggering_message_id: Optional[str],
) -> None:
    """
    This is the single, unified entry point for the main application.
    It determines the correct AI provider and delegates the message processing.
    """
    try:
        provider = get_ai_provider()
    except Exception as e:
        logger.exception("Failed to initialize an AI provider. Check configuration.")
        support_board_service.send_reply_to_channel(
            conversation_id=sb_conversation_id,
            message_text=f"Error de configuración del servidor de IA: {e}",
            source=conversation_source,
            target_user_id=customer_user_id,
            triggering_message_id=triggering_message_id,
        )
        return

    # Dynamic context is common to all providers and is fetched here
    reservation_context = conversation_details.get_reservation_details(sb_conversation_id)
    conversation_data = support_board_service.get_sb_conversation_data(sb_conversation_id)

    # --- START OF FIX: Automatic Department Routing ---
    # This block checks if the conversation is unassigned and routes it to the
    # default department specified in the .env file.
    if conversation_data:
        # In Support Board, an unassigned conversation has a `department` value of `null` or `0`.
        # We check for any falsy value (None, 0, etc.) to be safe.
        current_department = conversation_data.get("details", {}).get("department")
        if not current_department:
            default_dept_id = getattr(Config, "SUPPORT_BOARD_ATENCION_AL_CLIENTE_ID", None)
            if default_dept_id:
                try:
                    dept_id_int = int(default_dept_id)
                    logger.info(f"Conversation {sb_conversation_id} is unassigned. Routing to default 'Atención al Cliente' department (ID: {dept_id_int}).")
                    support_board_service.assign_conversation_to_department(
                        conversation_id=sb_conversation_id,
                        department_id=dept_id_int
                    )
                except (ValueError, TypeError):
                    logger.error(f"Cannot auto-route conversation {sb_conversation_id}: Configured SUPPORT_BOARD_ATENCION_AL_CLIENTE_ID ('{default_dept_id}') is not a valid integer.")
            else:
                logger.warning(f"Cannot auto-route conversation {sb_conversation_id}: SUPPORT_BOARD_ATENCION_AL_CLIENTE_ID is not configured in the environment.")
    # --- END OF FIX ---

    # Delegate the entire processing task to the selected provider
    final_assistant_response = provider.process_message(
        sb_conversation_id=sb_conversation_id,
        new_user_message=new_user_message,
        conversation_data=conversation_data,
        reservation_context=reservation_context
    )
    
    # Sending the final reply back to the user is also a common task
    if final_assistant_response:
        support_board_service.send_reply_to_channel(
            conversation_id=sb_conversation_id,
            message_text=str(final_assistant_response),
            source=conversation_source,
            target_user_id=customer_user_id,
            conversation_details=conversation_data,
            triggering_message_id=triggering_message_id,
        )
    else:
        logger.error(f"Provider '{Config.AI_PROVIDER}' returned no response for Conv {sb_conversation_id}.")
        # Optionally, send a generic fallback message
        support_board_service.send_reply_to_channel(
            conversation_id=sb_conversation_id,
            message_text="Lo siento, no pude generar una respuesta en este momento.",
            source=conversation_source,
            target_user_id=customer_user_id,
            conversation_details=conversation_data,
            triggering_message_id=triggering_message_id,
        )

# This utility function should ideally live in its own file (e.g., services/llm_utils.py)
# but for now, we leave it here and correct its dependency.
def extract_customer_info_via_llm(message_text: str) -> Optional[Dict[str, Any]]:
    # This function depends on the OpenAI client, which is now inside the provider.
    # For now, let's create a temporary client instance.
    from openai import OpenAI
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    system_prompt = ("Extrae la siguiente información del mensaje del cliente. "
        "Devuelve solo JSON válido con las claves: full_name, cedula, telefono, "
        "correo, direccion, productos y total. Si falta algún campo, usa null. "
        "No incluyas explicaciones ni comentarios.")
    user_prompt = f"Mensaje del cliente:\n\"\"\"{message_text}\"\"\""
    try:
        response = client.chat.completions.create(model=Config.OPENAI_CHAT_MODEL, messages=[
                {"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"}, temperature=0, max_tokens=256)
        content = response.choices[0].message.content if response.choices else None
        return json.loads(content) if content else None
    except Exception as e:
        logger.exception(f"Error extracting customer info via OpenAI: {e}")
        return None