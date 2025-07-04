import pytest
import json
from unittest.mock import patch, MagicMock

# Import the specific model needed for creating test data
from namwoo_app.models import Product

# A helper function to create a standard webhook payload for our tests
def create_webhook_payload(conversation_id, message, customer_id):
    """Helper function to create a mock Support Board webhook payload."""
    return {
        "conversation_id": conversation_id,
        "message": message,
        "sender_user_id": str(customer_id),
        "user": {"id": str(customer_id)},
        "conversation_source": "whatsapp",
        "data": {  # Nesting data to match your logs
            "conversation_id": conversation_id,
            "message": message,
            "sender_user_id": str(customer_id),
            "user_id": str(customer_id),
            "conversation_user_id": str(customer_id),
            "conversation_source": "whatsapp",
            "triggering_message_id": "msg-12345"
        }
    }

# This is the test for our specific "get_branch_address" scenario.
# We patch the two external network calls: the OpenAI API and the final reply service.
@patch('namwoo_app.services.support_board_service.send_reply_to_channel')
@patch('openai.resources.chat.completions.Completions.create')
def test_get_branch_address_tool_call_flow(mock_openai_create, mock_send_reply, client, db_session):
    """
    Tests the full integration flow for the 'get_branch_address' tool call.
    1. ARRANGE: Sets up the database with a known product and address.
    2. ARRANGE: Mocks the OpenAI API to return a specific "tool_call" response.
    3. ACT: Sends a webhook to the API simulating the user's question.
    4. ASSERT: Verifies that the final reply sent to the user contains the correct address from the database.
    """
    # 1. --- ARRANGE ---

    # A. Create a product in the test database with a known branch and address.
    # This is the "ground truth" our test will check against.
    test_address = "Avenida Principal de Chuao, Nivel C1, Caracas 1061, Miranda"
    test_product = Product(
        id="IPHONE15_CCCT",
        item_code="IPHONE15",
        item_name="iPhone 15 Pro",
        branch_name="CCCT",
        warehouse_name="ALMACEN_CCCT",
        warehouse_name_canonical="almacen_ccct",
        store_address=test_address,
        stock=10,
        price=1200.00
    )
    db_session.add(test_product)
    db_session.commit()

    # B. Configure the mock for the OpenAI API call.
    # We want it to behave as if the AI decided to call our tool.
    mock_tool_call_response = MagicMock()
    mock_tool_call_response.choices = [MagicMock()]
    mock_tool_call_response.choices[0].message = MagicMock(
        tool_calls=[
            MagicMock(
                id='call_123',
                type='function',
                function=MagicMock(
                    name='get_branch_address',
                    arguments=json.dumps({'branchName': 'CCCT', 'city': 'Caracas'})
                )
            )
        ]
    )
    
    # Second mock response after the tool result is sent back to OpenAI
    mock_final_text_response = MagicMock()
    mock_final_text_response.choices = [MagicMock()]
    mock_final_text_response.choices[0].message = MagicMock(
        content=f"Claro, la dirección de nuestra sucursal CCCT es: {test_address}. ¿Te puedo ayudar en algo más?",
        tool_calls=None # No more tool calls
    )

    # Make the mock return our desired responses in order.
    mock_openai_create.side_effect = [
        mock_tool_call_response, 
        mock_final_text_response
    ]

    # C. Prepare the incoming webhook from the user.
    webhook_payload = create_webhook_payload(
        conversation_id="conv-test-address-1",
        message="dime la direccion del ccct",
        customer_id="user-xyz"
    )

    # 2. --- ACT ---
    # Make the HTTP request to our application's webhook endpoint.
    response = client.post('/api/sb-webhook', json=webhook_payload)

    # 3. --- ASSERT ---
    # A. Check that our API responded successfully.
    assert response.status_code == 200

    # B. Verify that the OpenAI API was called (we expect it to be called twice).
    assert mock_openai_create.call_count == 2
    
    # C. This is the most important check:
    # Verify that our `send_reply_to_channel` function was called to send the final answer.
    mock_send_reply.assert_called_once()

    # D. Inspect the arguments of the call to `send_reply_to_channel`
    # to ensure it contains the correct address from our test database.
    call_args, call_kwargs = mock_send_reply.call_args
    final_message_sent = call_kwargs.get("message_text", "")
    
    print(f"Final message sent by bot: {final_message_sent}") # Helpful for debugging
    assert test_address in final_message_sent
    assert "CCCT" in final_message_sent