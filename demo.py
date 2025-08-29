from agent.customer_support_agent import CustomerSupportAgent

if __name__ == "__main__":
    # Example input payload
    input_payload = {
        "customer_name": "Alice",
        "email": "alice@example.com",
        "query": "I can't log into my account",
        "priority": "high",
        "ticket_id": "T12345"
    }

    agent = CustomerSupportAgent()
    final_state = agent.process_customer_request(input_payload)
    print(final_state)
