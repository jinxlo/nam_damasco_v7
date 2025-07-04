# NAMWOO/services/openai_service.py
# -*- coding: utf-8 -*-
import logging
import json
import time
from typing import List, Dict, Optional, Tuple, Union, Any
from openai import OpenAI, APIError, RateLimitError, APITimeoutError, BadRequestError
from flask import current_app

# Import local services and utils
from . import product_service
from . import support_board_service
from . import geolocation_service
from ..config import Config
from ..utils import embedding_utils
from ..utils import conversation_location
from ..utils import conversation_details
from ..utils import message_parser
from ..utils.conversation_details import (
    KEY_ITEM_CODE, KEY_ITEM_NAME, KEY_FULL_NAME, KEY_CEDULA, KEY_PHONE,
    KEY_EMAIL, KEY_DELIVERY_METHOD, KEY_BRANCH_NAME, KEY_DELIVERY_ADDRESS,
    KEY_PAYMENT_METHOD, KEY_ACCESSORIES, KEY_PRICE_USD, KEY_PRICE_VES
)


logger = logging.getLogger(__name__)

_chat_client: Optional[OpenAI] = None
try:
    openai_api_key = Config.OPENAI_API_KEY
    if openai_api_key:
        timeout_seconds = getattr(Config, 'OPENAI_REQUEST_TIMEOUT', 60.0)
        _chat_client = OpenAI(api_key=openai_api_key, timeout=timeout_seconds)
        logger.info(f"OpenAI client initialized for Chat Completions service with timeout: {timeout_seconds}s.")
    else:
        _chat_client = None
        logger.error("OpenAI API key not configured. Chat functionality will fail.")
except Exception as e:
    logger.exception(f"Failed to initialize OpenAI client for chat: {e}")
    _chat_client = None

MAX_HISTORY_MESSAGES = Config.MAX_HISTORY_MESSAGES
TOOL_CALL_RETRY_LIMIT = 2
DEFAULT_OPENAI_MODEL = getattr(Config, "OPENAI_CHAT_MODEL", "gpt-4o-mini")
DEFAULT_MAX_TOKENS = getattr(Config, "OPENAI_MAX_TOKENS", 1024)
DEFAULT_OPENAI_TEMPERATURE = getattr(Config, "OPENAI_TEMPERATURE", 0.7)


def generate_product_embedding(text_to_embed: str) -> Optional[List[float]]:
    if not text_to_embed or not isinstance(text_to_embed, str):
        return None
    embedding_model_name = Config.OPENAI_EMBEDDING_MODEL
    if not embedding_model_name:
        logger.error("OPENAI_EMBEDDING_MODEL not configured.")
        return None
    return embedding_utils.get_embedding(text=text_to_embed, model=embedding_model_name)


def get_openai_product_summary(plain_text_description: str, item_name: Optional[str] = None) -> Optional[str]:
    global _chat_client
    if not _chat_client or not plain_text_description: return None
    prompt_context = f"Nombre del Producto: {item_name}\nDescripción Original:\n{plain_text_description}"
    system_prompt = "Eres un redactor experto. Resume la siguiente descripción de producto de forma concisa (50-75 palabras), factual, y sin jerga de marketing. Salida en texto plano."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_context}]
    try:
        completion = _chat_client.chat.completions.create(model=getattr(Config, "OPENAI_SUMMARY_MODEL", DEFAULT_OPENAI_MODEL),
            messages=messages, temperature=0.2, max_tokens=150, n=1)
        return completion.choices[0].message.content.strip() if completion.choices and completion.choices[0].message.content else None
    except Exception as e:
        logger.error(f"OpenAI summarization error for '{item_name}': {e}", exc_info=True)
        return None


def extract_customer_info_via_llm(message_text: str) -> Optional[Dict[str, Any]]:
    """Extract structured customer info from a plain text message using OpenAI."""
    global _chat_client
    if not _chat_client:
        logger.error("OpenAI client for chat not initialized.")
        return None
    system_prompt = ("Extrae la siguiente información del mensaje del cliente. "
        "Devuelve solo JSON válido con las claves: full_name, cedula, telefono, "
        "correo, direccion, productos y total. Si falta algún campo, usa null. "
        "No incluyas explicaciones ni comentarios.")
    user_prompt = f"Mensaje del cliente:\n\"\"\"{message_text}\"\"\""
    try:
        response = _chat_client.chat.completions.create(model=DEFAULT_OPENAI_MODEL, messages=[
                {"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"}, temperature=0, max_tokens=256)
        content = response.choices[0].message.content if response.choices else None
        return json.loads(content) if content else None
    except Exception as e:
        logger.exception(f"Error extracting customer info via OpenAI: {e}")
        return None


# ===========================================================================
# Tool Definitions and Handling for the Main Agent
# ===========================================================================

tools_schema = [
    {
        "type": "function", "function": {
            "name": "find_products",
            "description": "Searches the product catalog by keyword, SKU, or brand. If 'city' is provided, it filters for stock in that city's stores. Otherwise, it performs a nationwide search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "User's search term (e.g., 'iPhone 15', 'Honor', 'celular 16gb ram')."},
                    "city": {"type": "string", "description": "Optional. City to check for in-store stock."}
                }, "required": ["query"]
            }
        }
    },
    {
        "type": "function", "function": {
            "name": "get_available_brands",
            "description": "Gets a list of available brands for a given product category, like 'CELULAR'. Use this when the user asks for 'marcas'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The category to get brands for. Defaults to 'CELULAR'.",
                        "default": "CELULAR"
                    }
                }, "required": []
            }
        }
    },
    { 
        "type": "function", "function": {
            "name": "get_branch_address",
            "description": "Obtiene la dirección completa de una sucursal específica de Damasco basada en su nombre y ciudad.",
            "parameters": {"type": "object", "properties": {
                    "branchName": {"type": "string", "description": "Nombre de la sucursal."},
                    "city": {"type": "string", "description": "Ciudad donde se encuentra la sucursal."}
                },"required": ["branchName", "city"] 
            }
        }
    },
    {
        "type": "function", "function": {
            "name": "query_accessories",
            "description": "Busca accesorios relevantes (ej. forros, cargadores) para un producto principal, basado en su código de item (SKU).",
            "parameters": {"type": "object", "properties": {
                    "itemCode": {"type": "string", "description": "El itemCode (SKU) del producto principal."}
                },"required": ["itemCode"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_location_details_from_address",
            "description": "Busca tiendas cercanas basado en una dirección de texto del usuario (ej. calle, sector, ciudad). Usar cuando el usuario describe su ubicación con palabras en lugar de compartir un pin GPS.",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "La dirección textual del usuario (ej. 'Petare, Caracas', 'Avenida Libertador')."}
                },
                "required": ["address"]
            }
        }
    },
    {
        "type": "function", "function": {
            "name": "save_customer_reservation_details",
            "description": "Guarda o actualiza la información del cliente para su reserva actual. Debe llamarse cada vez que el usuario proporciona un dato personal.",
            "parameters": {"type": "object", "properties": {
                    KEY_FULL_NAME: {"type": "string"}, KEY_CEDULA: {"type": "string"},
                    KEY_PHONE: {"type": "string"}, KEY_EMAIL: {"type": "string"},
                    "city": {"type": "string", "description": "La ciudad del cliente para la entrega o retiro."},
                    KEY_DELIVERY_METHOD: {"type": "string", "enum": ["retiro_en_tienda", "envio_a_domicilio_caracas", "envio_nacional"]},
                    KEY_BRANCH_NAME: {"type": "string"}, KEY_DELIVERY_ADDRESS: {"type": "string"},
                    KEY_PAYMENT_METHOD: {"type": "string"}, KEY_ITEM_CODE: {"type": "string"},
                    KEY_ITEM_NAME: {"type": "string"}, KEY_PRICE_USD: {"type": "number"},
                    KEY_PRICE_VES: {"type": "number"}, KEY_ACCESSORIES: {"type": "array", "items": {"type": "string"}}
                }, "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_whatsapp_order_summary_template",
            "description": "Envía la plantilla de resumen de pedido por WhatsApp al cliente.",
            "parameters": { "type": "object", "properties": {
                    "customer_platform_user_id": {"type": "string"}, "conversation_id": {"type": "string"},
                    "template_variables": {"type": "array", "items": {"type": "string"}, "minItems": 8, "maxItems": 8}
                }, "required": ["customer_platform_user_id", "conversation_id", "template_variables"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_location_details_from_user",
            "description": "Information automatically extracted from a GPS location shared by the user. Contains the formatted address and a sorted list of nearby stores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "formatted_address": {
                        "type": "string",
                        "description": "The full, human-readable address of the user's location."
                    },
                    "nearby_stores": {
                        "type": "array",
                        "description": "A list of nearby stores, sorted from closest to farthest.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "branch_name": {"type": "string"},
                                "distance_km": {"type": "number"}
                            }
                        }
                    }
                },
                "required": ["nearby_stores"]
            }
        }
    }
]

def _format_brands_for_llm(brands: Optional[List[str]]) -> str:
    """Formats the list of brand strings into a JSON for the LLM."""
    if not brands:
        return json.dumps({"status": "not_found", "message": "No se encontraron marcas para esa categoría."}, ensure_ascii=False)
    
    return json.dumps({"status": "success", "brands": brands}, ensure_ascii=False)

def _format_result_for_llm(result: Optional[Dict[str, Any]], tool_name: str) -> str:
    """Generic formatter for tool results to be sent back to the LLM as a JSON string."""
    if result is None:
        return json.dumps({"status": "error", "message": f"Error interno al ejecutar la herramienta {tool_name}."}, ensure_ascii=False)
    try:
        return json.dumps(result, indent=2, ensure_ascii=False)
    except (TypeError, ValueError) as err:
        logger.error(f"JSON serialisation error for {tool_name} results: {err}", exc_info=True)
        return json.dumps({"status": "error", "message": "Error al formatear los resultados."}, ensure_ascii=False)

def _format_sb_history_for_openai(sb_messages: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not sb_messages: return []
    openai_messages: List[Dict[str, Any]] = []
    bot_user_id_str = str(Config.SUPPORT_BOARD_DM_BOT_USER_ID)
    
    for msg in sb_messages:
        role = "assistant" if str(msg.get("user_id")) == bot_user_id_str else "user"
        
        # Attempt to reconstruct tool calls and their responses from the payload
        try:
            # Check for assistant message with tool_calls
            if role == 'assistant' and 'payload' in msg and msg['payload']:
                payload_json = json.loads(msg['payload'])
                if 'tool_calls' in payload_json:
                    openai_messages.append({
                        "role": "assistant", 
                        "content": None, 
                        "tool_calls": payload_json['tool_calls']
                    })
                    continue # Skip adding this message as simple text

            # Check for tool response message
            if role == 'tool' and 'payload' in msg and msg['payload']:
                payload_json = json.loads(msg['payload'])
                if 'tool_call_id' in payload_json and 'name' in payload_json:
                     openai_messages.append({
                        "role": "tool",
                        "tool_call_id": payload_json['tool_call_id'],
                        "name": payload_json['name'],
                        "content": payload_json['content']
                     })
                     continue
        except (json.JSONDecodeError, TypeError):
            # If payload is not valid JSON or not a tool call, fall through to process as text.
            pass

        # Process as a regular text message if it's not a tool message or if payload parsing fails
        text_content = msg.get("message", "").strip()
        if text_content:
            openai_messages.append({"role": role, "content": text_content})
            
    return openai_messages


def process_new_message(
    sb_conversation_id: str, new_user_message: Optional[str], conversation_source: Optional[str],
    sender_user_id: str, customer_user_id: str, triggering_message_id: Optional[str],
) -> None:
    if not _chat_client:
        logger.error("OpenAI client not initialized.")
        return

    conversation_data = support_board_service.get_sb_conversation_data(sb_conversation_id)
    sb_history_list = (conversation_data.get("messages", []) if conversation_data else [])
    openai_history = _format_sb_history_for_openai(sb_history_list)
    
    # --- START OF GEOLOCATION INTEGRATION ---
    if new_user_message:
        location_data = message_parser.extract_location_from_text(new_user_message)
        if location_data:
            logger.info(f"Location URL detected in message for Conv {sb_conversation_id}. Processing with geolocation service.")
            geo_details = geolocation_service.get_location_details(
                latitude=location_data['latitude'],
                longitude=location_data['longitude']
            )
            
            if geo_details and not geo_details.get("error"):
                tool_call_id = "user_location_tool_call"
                
                # Append the original user message (the URL) to the history
                openai_history.append({"role": "user", "content": new_user_message})

                # Append a virtual assistant tool call to the history.
                assistant_tool_call = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{ "id": tool_call_id, "type": "function", "function": {"name": "get_location_details_from_user", "arguments": '{}' }}]
                }
                openai_history.append(assistant_tool_call)
                # Also log this to support board for complete history
                support_board_service.add_message_to_sb_conversation(sb_conversation_id, {"payload": json.dumps({"tool_calls": assistant_tool_call["tool_calls"]})})
                
                # Append the actual result from our service as a tool response.
                tool_result_message = {
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "name": "get_location_details_from_user",
                    "content": json.dumps(geo_details, ensure_ascii=False)
                }
                openai_history.append(tool_result_message)
                # Also log this to support board
                support_board_service.add_message_to_sb_conversation(sb_conversation_id, {"payload": json.dumps(tool_result_message)})

                logger.info(f"Injected location context into history for Conv {sb_conversation_id}")
    # --- END OF GEOLOCATION INTEGRATION ---

    if not openai_history and new_user_message:
        openai_history.append({"role": "user", "content": new_user_message})
    if not openai_history:
        return

    system_prompt_content = Config.SYSTEM_PROMPT
    current_details = conversation_details.get_reservation_details(sb_conversation_id)
    if current_details:
        context_header = "\n\n--- CONTEXTO DE RESERVA ---\n"
        context_footer = "\n---------------------------\n\n"
        context_str = "Estado actual de la reserva del cliente (no preguntar de nuevo):\n" + "\n".join([f"- {key}: {value}" for key, value in current_details.items()])
        system_prompt_content = context_header + context_str + context_footer + system_prompt_content

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt_content}] + openai_history
    if len(messages) > (MAX_HISTORY_MESSAGES + 1):
        messages = [messages[0]] + messages[-(MAX_HISTORY_MESSAGES):]

    final_assistant_response: Optional[str] = None
    try:
        tool_call_count = 0
        while tool_call_count <= TOOL_CALL_RETRY_LIMIT:
            call_params = {"model": DEFAULT_OPENAI_MODEL, "messages": messages, "max_tokens": DEFAULT_MAX_TOKENS, "temperature": DEFAULT_OPENAI_TEMPERATURE, "tools": tools_schema, "tool_choice": "auto"}
            response = _chat_client.chat.completions.create(**call_params)
            response_message = response.choices[0].message
            messages.append(response_message.model_dump(exclude_none=True))

            if not response_message.tool_calls:
                final_assistant_response = response_message.content
                break

            tool_outputs_for_llm: List[Dict[str, str]] = []
            for tc in response_message.tool_calls:
                fn_name = tc.function.name
                tool_call_id = tc.id
                output_txt = ""
                try:
                    args = json.loads(tc.function.arguments)
                    logger.info(f"OpenAI requested tool call: {fn_name} with args: {args} for Conv {sb_conversation_id}")

                    if fn_name == "find_products":
                        query = args.get("query")
                        city_arg = args.get("city")

                        if city_arg:
                            logger.info(f"City '{city_arg}' provided. Checking for served warehouses in conversation {sb_conversation_id}.")
                            conversation_location.set_conversation_city(sb_conversation_id, city_arg)
                            warehouse_names_arg = conversation_location.get_city_warehouses(sb_conversation_id)

                            if not warehouse_names_arg:
                                logger.warning(f"City '{city_arg}' is not a served location. No warehouses found.")
                                output_txt = json.dumps({"status": "city_not_served", "city": city_arg}, ensure_ascii=False)
                            else:
                                search_res = product_service.find_products(query=query, warehouse_names=warehouse_names_arg)
                                
                                if not search_res or not (search_res.get("products_grouped") or search_res.get("product_details")):
                                    logger.info(f"Product '{query}' not found in city '{city_arg}'.")
                                    output_txt = json.dumps({"status": "not_found_in_city", "city": city_arg}, ensure_ascii=False)
                                else:
                                    output_txt = _format_result_for_llm(search_res, fn_name)
                        else:
                            search_res = product_service.find_products(query=query, warehouse_names=None)
                            output_txt = _format_result_for_llm(search_res, fn_name)

                    elif fn_name == "get_available_brands":
                        category_arg = args.get("category", "CELULAR")
                        brands_list = product_service.get_available_brands_by_category(category=category_arg)
                        output_txt = _format_brands_for_llm(brands_list)
                    
                    elif fn_name == "get_branch_address":
                        branch_details_result = product_service.get_branch_address(branch_name=args.get("branchName"), city=args.get("city"))
                        output_txt = _format_result_for_llm(branch_details_result, fn_name)

                    elif fn_name == "query_accessories":
                        city_warehouses = conversation_location.get_city_warehouses(sb_conversation_id)
                        accessories_result = product_service.query_accessories(main_product_item_code=args.get("itemCode"), city_warehouses=city_warehouses)
                        output_txt = json.dumps({"status": "success", "accessories_list": ", ".join(accessories_result)} if accessories_result else {"status": "not_found"}, ensure_ascii=False)

                    elif fn_name == "get_location_details_from_address":
                        address_arg = args.get("address")
                        if not address_arg:
                            output_txt = json.dumps({"status": "error", "message": "El parámetro 'address' es requerido."}, ensure_ascii=False)
                        else:
                            location_details_result = geolocation_service.get_location_details_from_address(address=address_arg)
                            output_txt = _format_result_for_llm(location_details_result, fn_name)

                    elif fn_name == "save_customer_reservation_details":
                        saved_keys = []
                        for key, value in args.items():
                            if key == 'city' and isinstance(value, str):
                                conversation_location.set_conversation_city(sb_conversation_id, value)
                            conversation_details.store_reservation_detail(sb_conversation_id, key, value)
                            saved_keys.append(key)
                        output_txt = json.dumps({"status": "success", "message": f"OK. Detalles guardados: {', '.join(saved_keys)}."} if saved_keys else {"status": "no_action"}, ensure_ascii=False)

                    elif fn_name == "send_whatsapp_order_summary_template":
                         output_txt = json.dumps({"status": "success", "message": "OK_TEMPLATE_SENT"}, ensure_ascii=False)

                    else:
                         output_txt = json.dumps({"status": "error", "message": f"Herramienta desconocida '{fn_name}'."}, ensure_ascii=False)

                except Exception as tool_exec_err:
                    logger.exception(f"Tool execution error for {fn_name}: {tool_exec_err}")
                    output_txt = json.dumps({"status": "error", "message": f"Error interno al ejecutar {fn_name}."}, ensure_ascii=False)

                tool_outputs_for_llm.append({"tool_call_id": tool_call_id, "role": "tool", "name": fn_name, "content": output_txt})

            messages.extend(tool_outputs_for_llm)
            tool_call_count += 1
            if tool_call_count > TOOL_CALL_RETRY_LIMIT and not final_assistant_response: break

    except Exception as e:
        logger.exception(f"Unexpected OpenAI interaction error for Conv {sb_conversation_id}: {e}")
        final_assistant_response = "Ocurrió un error inesperado. Por favor, intenta de nuevo."

    if final_assistant_response:
        support_board_service.send_reply_to_channel(sb_conversation_id, str(final_assistant_response), conversation_source, customer_user_id, conversation_data, triggering_message_id)
    else:
        logger.error("No final response generated for Conv %s; sending fallback.", sb_conversation_id)
        support_board_service.send_reply_to_channel(sb_conversation_id, "No pude generar una respuesta. Por favor, intenta de nuevo.", conversation_source, customer_user_id, conversation_data, triggering_message_id)