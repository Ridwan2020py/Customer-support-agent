# LangGraph Customer Support Agent

A structured, stage-based customer support agent built with LangGraph, implementing 11-stage workflow with MCP client orchestration.

## 🎯 Overview

**Langie** is a LangGraph Agent that models customer support workflows as graph-based stages. Each stage represents a step in the workflow, with state persistence and dynamic ability orchestration through MCP (Model Context Protocol) clients.

### Key Features

- **11-Stage Workflow**: Complete customer support process from intake to completion
- **State Persistence**: Maintains context and data across all stages
- **MCP Integration**: Routes abilities to Common and Atlas servers
- **Non-Deterministic Logic**: Dynamic decision-making in the DECIDE stage
- **Comprehensive Logging**: Tracks execution flow and decisions

## 🏗️ Architecture

### Stage Flow

```
INTAKE → UNDERSTAND → PREPARE → ASK → WAIT → RETRIEVE → DECIDE → UPDATE → CREATE → DO → COMPLETE
```

### Stage Details

| Stage | Mode | Abilities | MCP Server | Description |
|-------|------|-----------|------------|-------------|
| INTAKE | Entry | accept_payload | Internal | Accept incoming request |
| UNDERSTAND | Deterministic | parse_request_text, extract_entities | Common, Atlas | Parse and extract entities |
| PREPARE | Deterministic | normalize_fields, enrich_records, add_flags_calculations | Common, Atlas | Data normalization and enrichment |
| ASK | Human | clarify_question | Atlas | Request missing information |
| WAIT | Deterministic | extract_answer, store_answer | Atlas, Internal | Capture customer response |
| RETRIEVE | Deterministic | knowledge_base_search, store_data | Atlas, Internal | Knowledge base lookup |
| DECIDE | Non-Deterministic | solution_evaluation, escalation_decision, update_payload | Common, Atlas | Score solutions and decide escalation |
| UPDATE | Deterministic | update_ticket, close_ticket | Atlas | Update ticket status |
| CREATE | Deterministic | response_generation | Common | Generate customer response |
| DO | Deterministic | execute_api_calls, trigger_notifications | Atlas | Execute actions and notifications |
| COMPLETE | Output | output_payload | Internal | Output final payload |

### MCP Server Distribution

- **Common Server**: Internal abilities with no external data requirements
- **Atlas Server**: External system interactions and data retrieval

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <github.com:Ridwan2020py/Customer-support-agent.git>
cd langgraph-customer-support-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Demo

```bash
python customer_support_agent.py
```

### Sample Input

```json
{
  "customer_name": "John Smith",
  "email": "john.smith@email.com", 
  "query": "I was charged twice for my premium subscription this month",
  "priority": "high",
  "ticket_id": "12345"
}
```

### Expected Output

```json
{
  "customer_name": "John Smith",
  "email": "john.smith@email.com",
  "query": "I was charged twice for my premium subscription this month",
  "priority": "high",
  "ticket_id": "12345",
  "escalation_decision": false,
  "ticket_updates": {"status": "resolved", "resolution": "auto_resolved"},
  "generated_response": "Dear John Smith, we have reviewed your I was charged twice for my premium subscription this month and...",
  "notifications_sent": ["email_sent", "sms_sent"],
  "completed_at": "2024-08-28T10:30:00"
}
```

## 🔧 Implementation Details

### State Management

The agent maintains comprehensive state across all stages:

```python
class CustomerSupportState(TypedDict):
    # Input fields
    customer_name: str
    email: str
    query: str
    priority: str
    ticket_id: str
    
    # Processing fields
    parsed_request: Optional[Dict[str, Any]]
    entities: Optional[Dict[str, Any]]
    # ... additional fields
    
    # Metadata
    stage_history: List[str]
    execution_log: List[Dict[str, Any]]
    current_stage: str
```

### Non-Deterministic Decision Logic

The DECIDE stage implements dynamic logic:

```python
# Evaluate solutions and escalate if best score < 90
best_score = max(solution["score"] for solution in state["solution_scores"])

if best_score < 90:
    state["escalation_decision"] = True
    # Route to human agent
else:
    state["escalation_decision"] = False  
    # Auto-resolve
```

### MCP Client Integration

```python
class MCPClient:
    def execute_ability(self, ability_name: str, payload: Dict[str, Any]):
        # Routes abilities to appropriate server (Common/Atlas)
        # Returns structured results for state updates
```

## 📊 Demo Execution Log

```
🤖 LANGIE - Customer Support Agent Demo
========================================

📥 Input Payload:
{
  "customer_name": "John Smith",
  "email": "john.smith@email.com",
  "query": "I was charged twice for my premium subscription this month",
  "priority": "high",
  "ticket_id": "12345"
}

🔄 Processing through 11 stages...

📊 Stage Execution Summary:
  ✅ INTAKE - accept_payload
  ✅ UNDERSTAND - parse_request_text, extract_entities
  ✅ PREPARE - normalize_fields, enrich_records, add_flags_calculations
  ✅ ASK - clarify_question
  ✅ WAIT - extract_answer, store_answer
  ✅ RETRIEVE - knowledge_base_search, store_data
  ✅ DECIDE - solution_evaluation, escalation_decision, update_payload
  ✅ UPDATE - update_ticket, close_ticket
  ✅ CREATE - response_generation
  ✅ DO - execute_api_calls, trigger_notifications
  ✅ COMPLETE - output_payload

🏁 Final Status: resolved
📧 Generated Response: Dear John Smith, we have reviewed your I was charged twice for my premium subscription this month and...
🔔 Notifications: email_sent, sms_sent
```

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/

# Run specific test
python -m unittest tests.test_agent.TestCustomerSupportAgent.test_complete_workflow
```

## 📁 Project Structure

```
langgraph-customer-support-agent/
├── customer_support_agent.py      # Main implementation
├── agent_config.yaml             # Configuration
├── requirements.txt               # Dependencies  
├── setup.py                      # Package setup
├── README.md                     # This file
├── tests/                        # Test suite
│   ├── test_agent.py
│   └── test_stages.py
├── examples/                     # Usage examples
│   └── sample_inputs.json
└── docs/                        # Documentation
    └── architecture.md
```
