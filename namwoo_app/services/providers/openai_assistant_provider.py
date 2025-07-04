# NAMWOO/services/providers/openai_assistant_provider.py
# -*- coding: utf-8 -*-
import logging
import json
import time
from typing import List, Dict, Optional, Any
from openai import OpenAI
from openai.types.beta.threads import Run

# Import local services and utils
from .. import product_service
from .. import support_board_service
from .. import geolocation_service
from .. import thread_mapping_service 
from ...config import Config
from ...utils import conversation_location
from ...utils import conversation_details
from ...utils import message_parser

logger = logging.getLogger(__name__)

# In the Assistants API, the tool schema is defined ONCE on the Assistant object itself.
# It is not needed here, but we need the tool execution logic.

class OpenAIAssistantProvider:
    def __init__(self, api_key: str, assistant_id: str):
        if not api_key or not assistant_id:
            raise ValueError("API key and Assistant ID are required for OpenAIAssistantProvider.")
        
        self.client = OpenAI(api_key=api_key)
        self.assistant_id = assistant_id
        # Polling configuration
        self.polling_interval_seconds = 1
        self.run_timeout_seconds = 120
        logger.info(f"OpenAIAssistantProvider initialized for Assistant ID '{self.assistant_id}'.")

    def _get_or_create_thread_id(self, sb_conversation_id: str) -> str:
        """
        Retrieves a thread_id for a given conversation_id.
        If none exists, creates a new thread and stores the mapping.
        """
        thread_id = thread_mapping_service.get_thread_id(sb_conversation_id)
        if not thread_id:
            logger.info(f"No existing thread for Conv {sb_conversation_id}. Creating a new one.")
            thread = self.client.beta.threads.create()
            thread_id = thread.id
            thread_mapping_service.store_thread_id(sb_conversation_id, thread_id)
        return thread_id

    def _handle_geolocation_injection(self, thread_id: str, new_user_message: str) -> bool:
        """
        Handles the special case where a user sends a GPS link.
        This logic adds the location details as a new user message in the thread.
        """
        location_data = message_parser.extract_location_from_text(new_user_message)
        if not location_data:
            return False

        logger.info(f"Location URL detected. Injecting geo-context into thread {thread_id}.")
        geo_details = geolocation_service.get_location_details(
            latitude=location_data['latitude'],
            longitude=location_data['longitude']
        )
        
        if geo_details and not geo_details.get("error"):
            # First, add the user's original message (the URL) to the thread.
            self.client.beta.threads.messages.create(
                thread_id=thread_id, 
                role="user", 
                content=new_user_message
            )
            
            # Next, create a formatted string with the context for the AI.
            nearby_stores_text = "\n".join([
                f"- {store['branch_name']} (a {store['distance_km']} km)" 
                for store in geo_details.get("nearby_stores", [])
            ])
            context_message = (
                "[CONTEXTO DE UBICACIÓN PROPORCIONADO POR EL USUARIO]\n"
                f"Dirección Detectada: {geo_details.get('formatted_address', 'No disponible')}\n"
                f"Tiendas Cercanas:\n{nearby_stores_text}\n"
                "[FIN DEL CONTEXTO DE UBICACIÓN]"
            )
            
            # Add this new context block as another user message.
            # The system prompt will instruct the AI on how to interpret this.
            self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=context_message
            )
            logger.info(f"Successfully injected location context into thread {thread_id}.")
            return True
        return False

    def process_message(
        self,
        sb_conversation_id: str,
        new_user_message: Optional[str],
        conversation_data: Dict[str, Any],
        reservation_context: Dict[str, Any]
    ) -> Optional[str]:
        try:
            thread_id = self._get_or_create_thread_id(sb_conversation_id)
            
            # If the user sent a GPS link, handle it by injecting context and then start the run.
            # Otherwise, just add the new message.
            if new_user_message and self._handle_geolocation_injection(thread_id, new_user_message):
                pass # Geolocation was handled, message is already in thread.
            elif new_user_message:
                self.client.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=new_user_message
                )

            # --- Create the Run with dynamic instructions ---
            # The main instructions are on the Assistant object itself.
            # This 'instructions' parameter overrides/appends to it for this specific run.
            additional_instructions = Config.SYSTEM_PROMPT
            if reservation_context:
                context_header = "\n\n--- CONTEXTO DE RESERVA ---\n"
                context_footer = "\n---------------------------\n"
                context_str = "Estado actual de la reserva del cliente (no preguntar de nuevo):\n" + "\n".join([f"- {key}: {value}" for key, value in reservation_context.items()])
                additional_instructions += context_header + context_str + context_footer
            
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=self.assistant_id,
                instructions=additional_instructions
            )
            logger.info(f"Created Run {run.id} for Thread {thread_id}.")

            # --- Poll for Run completion ---
            start_time = time.time()
            while time.time() - start_time < self.run_timeout_seconds:
                run = self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)

                if run.status == 'completed':
                    logger.info(f"Run {run.id} completed.")
                    messages = self.client.beta.threads.messages.list(thread_id=thread_id, limit=1)
                    # The latest message is at index 0
                    return messages.data[0].content[0].text.value

                if run.status == 'requires_action':
                    logger.info(f"Run {run.id} requires tool action.")
                    tool_outputs = self._execute_tool_calls(run.required_action.submit_tool_outputs.tool_calls, sb_conversation_id)
                    self.client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread_id,
                        run_id=run.id,
                        tool_outputs=tool_outputs
                    )
                    logger.info(f"Submitted {len(tool_outputs)} tool outputs for Run {run.id}.")

                if run.status in ['failed', 'cancelled', 'expired']:
                    logger.error(f"Run {run.id} terminated with status: {run.status}. Error: {run.last_error}")
                    return f"Lo siento, la operación ha fallado con el estado: {run.status}. Por favor, intente de nuevo."

                time.sleep(self.polling_interval_seconds)
            
            logger.error(f"Run {run.id} timed out after {self.run_timeout_seconds} seconds.")
            return "Lo siento, la operación ha tardado demasiado en completarse."

        except Exception as e:
            logger.exception(f"OpenAIAssistantProvider error for Conv {sb_conversation_id}: {e}")
            return "Ocurrió un error inesperado con nuestro asistente (Asistente API). Por favor, intenta de nuevo."

    def _execute_tool_calls(self, tool_calls: List[Any], sb_conversation_id: str) -> List[Dict[str, str]]:
        tool_outputs = []
        for tc in tool_calls:
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON arguments for tool {fn_name}: {tc.function.arguments}")
                args = {} # Default to empty dict if args are invalid
            
            logger.info(f"Assistants API requested tool: {fn_name} with args: {args} for Conv {sb_conversation_id}")
            
            output = {}
            try:
                if fn_name == "find_products":
                    query, city_arg = args.get("query"), args.get("city")
                    if city_arg:
                        conversation_location.set_conversation_city(sb_conversation_id, city_arg)
                        warehouses = conversation_location.get_city_warehouses(sb_conversation_id)
                        if not warehouses:
                            output = {"status": "city_not_served", "city": city_arg}
                        else:
                            search_res = product_service.find_products(query=query, warehouse_names=warehouses)
                            if not search_res or not (search_res.get("products_grouped") or search_res.get("product_details")):
                                output = {"status": "not_found_in_city", "city": city_arg}
                            else:
                                output = search_res
                    else:
                        output = product_service.find_products(query=query, warehouse_names=None)
                elif fn_name == "get_available_brands":
                    brands = product_service.get_available_brands_by_category(category=args.get("category", "CELULAR"))
                    output = {"status": "success", "brands": brands} if brands else {"status": "not_found"}
                elif fn_name == "get_branch_address":
                    output = product_service.get_branch_address(
                        branch_name=args.get("branchName"), 
                        city=args.get("city")
                    )
                elif fn_name == "query_accessories":
                    warehouses = conversation_location.get_city_warehouses(sb_conversation_id)
                    result = product_service.query_accessories(
                        main_product_item_code=args.get("itemCode"), 
                        city_warehouses=warehouses
                    )
                    output = {"status": "success", "accessories_list": result} if result else {"status": "not_found"}
                elif fn_name == "get_location_details_from_address":
                    output = geolocation_service.get_location_details_from_address(address=args.get("address"))
                elif fn_name == "save_customer_reservation_details":
                    saved_keys = [key for key, value in args.items() if conversation_details.store_reservation_detail(sb_conversation_id, key, value)]
                    if 'city' in args:
                        conversation_location.set_conversation_city(sb_conversation_id, args['city'])
                    output = {"status": "success", "message": f"OK. Detalles guardados: {', '.join(saved_keys)}."} if saved_keys else {"status": "no_action"}
                elif fn_name == "send_whatsapp_order_summary_template":
                     # This tool is a placeholder for a future implementation.
                     output = {"status": "success", "message": "OK_TEMPLATE_SENT"}
                else:
                    output = {"status": "error", "message": f"Herramienta desconocida '{fn_name}'."}

            except Exception as e:
                logger.exception(f"Tool execution error for {fn_name}: {e}")
                output = {"status": "error", "message": f"Error interno al ejecutar {fn_name}: {str(e)}"}
            
            tool_outputs.append({
                "tool_call_id": tc.id,
                "output": json.dumps(output, ensure_ascii=False) # Output MUST be a string
            })
        return tool_outputs