"""
LangGraph Customer Support Agent Implementation
Author: Generated for task submission
"""

from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, START, END
import json
import logging
from datetime import datetime
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MCPServer(Enum):
    COMMON = "common"
    ATLAS = "atlas"

class StageMode(Enum):
    DETERMINISTIC = "deterministic"
    NON_DETERMINISTIC = "non_deterministic"
    HUMAN = "human"

# State Schema
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
    normalized_data: Optional[Dict[str, Any]]
    enriched_records: Optional[Dict[str, Any]]
    flags_calculations: Optional[Dict[str, Any]]
    clarification_needed: Optional[str]
    customer_response: Optional[str]
    knowledge_base_results: Optional[List[Dict]]
    solution_scores: Optional[List[Dict]]
    escalation_decision: Optional[bool]
    ticket_updates: Optional[Dict[str, Any]]
    generated_response: Optional[str]
    api_results: Optional[List[Dict]]
    notifications_sent: Optional[List[str]]
    
    # Metadata
    stage_history: List[str]
    execution_log: List[Dict[str, Any]]
    current_stage: str
    completed_at: Optional[str]

class MCPClient:
    """Mock MCP Client for demonstration"""
    
    def __init__(self, server_type: MCPServer):
        self.server_type = server_type
        
    def execute_ability(self, ability_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an ability on the specified MCP server"""
        logger.info(f"Executing {ability_name} on {self.server_type.value} server")
        
        # Mock implementations for demonstration
        if ability_name == "parse_request_text":
            return {
                "structured_data": {
                    "issue_type": "billing",
                    "urgency": "high",
                    "category": "payment_issue"
                }
            }
        elif ability_name == "extract_entities":
            return {
                "entities": {
                    "product": "Premium Plan",
                    "account_id": "ACC123",
                    "dates": ["2024-01-15"]
                }
            }
        elif ability_name == "normalize_fields":
            return {
                "normalized": {
                    "priority": payload.get("priority", "medium").upper(),
                    "ticket_id": f"TK-{payload.get('ticket_id', '000')}"
                }
            }
        elif ability_name == "enrich_records":
            return {
                "enriched": {
                    "sla_deadline": "2024-12-31T23:59:59Z",
                    "customer_tier": "premium",
                    "historical_tickets": 3
                }
            }
        elif ability_name == "solution_evaluation":
            return {
                "solutions": [
                    {"solution": "Auto-refund", "score": 95},
                    {"solution": "Manual review", "score": 75},
                    {"solution": "Escalate", "score": 60}
                ]
            }
        elif ability_name == "knowledge_base_search":
            return {
                "results": [
                    {"title": "Billing Issues FAQ", "relevance": 0.9, "content": "Common billing solutions..."},
                    {"title": "Premium Plan Guide", "relevance": 0.8, "content": "Premium features..."}
                ]
            }
        elif ability_name == "response_generation":
            return {
                "response": f"Dear {payload.get('customer_name', 'Customer')}, we have reviewed your {payload.get('query', 'inquiry')} and..."
            }
        else:
            return {"result": f"Executed {ability_name}", "status": "success"}

class CustomerSupportAgent:
    """LangGraph Customer Support Agent"""
    
    def __init__(self):
        self.common_client = MCPClient(MCPServer.COMMON)
        self.atlas_client = MCPClient(MCPServer.ATLAS)
        self.graph = self._create_graph()
        
    def _log_stage_execution(self, state: CustomerSupportState, stage: str, abilities: List[str]) -> CustomerSupportState:
        """Log stage execution details"""
        state["stage_history"].append(stage)
        state["current_stage"] = stage
        state["execution_log"].append({
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "abilities": abilities
        })
        logger.info(f"Executing stage: {stage} with abilities: {abilities}")
        return state
        
    def _get_mcp_client(self, server: MCPServer) -> MCPClient:
        """Get the appropriate MCP client"""
        return self.atlas_client if server == MCPServer.ATLAS else self.common_client
    
    # Stage 1: INTAKE
    def intake_stage(self, state: CustomerSupportState) -> CustomerSupportState:
        """Accept incoming payload"""
        state = self._log_stage_execution(state, "INTAKE", ["accept_payload"])
        logger.info(f"Processing ticket {state['ticket_id']} for customer {state['customer_name']}")
        return state
    
    # Stage 2: UNDERSTAND (Deterministic)
    def understand_stage(self, state: CustomerSupportState) -> CustomerSupportState:
        """Parse request and extract entities"""
        state = self._log_stage_execution(state, "UNDERSTAND", ["parse_request_text", "extract_entities"])
        
        # Execute parse_request_text on COMMON server
        parse_result = self.common_client.execute_ability("parse_request_text", state)
        state["parsed_request"] = parse_result
        
        # Execute extract_entities on ATLAS server
        entities_result = self.atlas_client.execute_ability("extract_entities", state)
        state["entities"] = entities_result
        
        return state
    
    # Stage 3: PREPARE (Deterministic)
    def prepare_stage(self, state: CustomerSupportState) -> CustomerSupportState:
        """Normalize, enrich, and add flags"""
        state = self._log_stage_execution(state, "PREPARE", ["normalize_fields", "enrich_records", "add_flags_calculations"])
        
        # Normalize fields (COMMON)
        normalize_result = self.common_client.execute_ability("normalize_fields", state)
        state["normalized_data"] = normalize_result
        
        # Enrich records (ATLAS)
        enrich_result = self.atlas_client.execute_ability("enrich_records", state)
        state["enriched_records"] = enrich_result
        
        # Add flags calculations (COMMON)
        flags_result = self.common_client.execute_ability("add_flags_calculations", state)
        state["flags_calculations"] = flags_result
        
        return state
    
    # Stage 4: ASK (Human)
    def ask_stage(self, state: CustomerSupportState) -> CustomerSupportState:
        """Request clarification if needed"""
        state = self._log_stage_execution(state, "ASK", ["clarify_question"])
        
        # Check if clarification is needed
        if state.get("parsed_request", {}).get("structured_data", {}).get("urgency") == "high":
            clarify_result = self.atlas_client.execute_ability("clarify_question", state)
            state["clarification_needed"] = "Please provide your account verification details"
        else:
            state["clarification_needed"] = None
            
        return state
    
    # Stage 5: WAIT (Deterministic)
    def wait_stage(self, state: CustomerSupportState) -> CustomerSupportState:
        """Extract and store customer response"""
        state = self._log_stage_execution(state, "WAIT", ["extract_answer", "store_answer"])
        
        if state.get("clarification_needed"):
            # Simulate customer response
            extract_result = self.atlas_client.execute_ability("extract_answer", state)
            state["customer_response"] = "Account verified: Premium customer since 2023"
        
        return state
    
    # Stage 6: RETRIEVE (Deterministic)
    def retrieve_stage(self, state: CustomerSupportState) -> CustomerSupportState:
        """Search knowledge base"""
        state = self._log_stage_execution(state, "RETRIEVE", ["knowledge_base_search", "store_data"])
        
        kb_result = self.atlas_client.execute_ability("knowledge_base_search", state)
        state["knowledge_base_results"] = kb_result["results"]
        
        return state
    
    # Stage 7: DECIDE (Non-deterministic)
    def decide_stage(self, state: CustomerSupportState) -> CustomerSupportState:
        """Evaluate solutions and make escalation decision"""
        state = self._log_stage_execution(state, "DECIDE", ["solution_evaluation", "escalation_decision", "update_payload"])
        
        # Evaluate solutions (COMMON)
        eval_result = self.common_client.execute_ability("solution_evaluation", state)
        state["solution_scores"] = eval_result["solutions"]
        
        # Non-deterministic decision: escalate if best solution score < 90
        best_score = max(solution["score"] for solution in state["solution_scores"])
        
        if best_score < 90:
            escalation_result = self.atlas_client.execute_ability("escalation_decision", state)
            state["escalation_decision"] = True
            logger.info(f"Escalating to human agent - best solution score: {best_score}")
        else:
            state["escalation_decision"] = False
            logger.info(f"Auto-resolving - best solution score: {best_score}")
        
        return state
    
    # Stage 8: UPDATE (Deterministic)
    def update_stage(self, state: CustomerSupportState) -> CustomerSupportState:
        """Update or close ticket"""
        state = self._log_stage_execution(state, "UPDATE", ["update_ticket", "close_ticket"])
        
        if state.get("escalation_decision"):
            update_result = self.atlas_client.execute_ability("update_ticket", state)
            state["ticket_updates"] = {"status": "escalated", "assigned_to": "human_agent"}
        else:
            close_result = self.atlas_client.execute_ability("close_ticket", state)
            state["ticket_updates"] = {"status": "resolved", "resolution": "auto_resolved"}
        
        return state
    
    # Stage 9: CREATE (Deterministic)
    def create_stage(self, state: CustomerSupportState) -> CustomerSupportState:
        """Generate customer response"""
        state = self._log_stage_execution(state, "CREATE", ["response_generation"])
        
        response_result = self.common_client.execute_ability("response_generation", state)
        state["generated_response"] = response_result["response"]
        
        return state
    
    # Stage 10: DO
    def do_stage(self, state: CustomerSupportState) -> CustomerSupportState:
        """Execute API calls and send notifications"""
        state = self._log_stage_execution(state, "DO", ["execute_api_calls", "trigger_notifications"])
        
        # Execute API calls (ATLAS)
        api_result = self.atlas_client.execute_ability("execute_api_calls", state)
        state["api_results"] = [{"api": "CRM", "status": "updated"}, {"api": "billing", "status": "processed"}]
        
        # Send notifications (ATLAS)
        notify_result = self.atlas_client.execute_ability("trigger_notifications", state)
        state["notifications_sent"] = ["email_sent", "sms_sent"]
        
        return state
    
    # Stage 11: COMPLETE
    def complete_stage(self, state: CustomerSupportState) -> CustomerSupportState:
        """Output final payload"""
        state = self._log_stage_execution(state, "COMPLETE", ["output_payload"])
        state["completed_at"] = datetime.now().isoformat()
        
        logger.info("Customer support workflow completed successfully")
        return state
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(CustomerSupportState)
        
        # Add nodes
        workflow.add_node("intake", self.intake_stage)
        workflow.add_node("understand", self.understand_stage)
        workflow.add_node("prepare", self.prepare_stage)
        workflow.add_node("ask", self.ask_stage)
        workflow.add_node("wait", self.wait_stage)
        workflow.add_node("retrieve", self.retrieve_stage)
        workflow.add_node("decide", self.decide_stage)
        workflow.add_node("update", self.update_stage)
        workflow.add_node("create", self.create_stage)
        workflow.add_node("do", self.do_stage)
        workflow.add_node("complete", self.complete_stage)
        
        # Add edges (sequential flow)
        workflow.add_edge(START, "intake")
        workflow.add_edge("intake", "understand")
        workflow.add_edge("understand", "prepare")
        workflow.add_edge("prepare", "ask")
        workflow.add_edge("ask", "wait")
        workflow.add_edge("wait", "retrieve")
        workflow.add_edge("retrieve", "decide")
        workflow.add_edge("decide", "update")
        workflow.add_edge("update", "create")
        workflow.add_edge("create", "do")
        workflow.add_edge("do", "complete")
        workflow.add_edge("complete", END)
        
        return workflow.compile()
    
    def process_customer_request(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a customer support request through the workflow"""
        # Initialize state
        initial_state = CustomerSupportState(
            customer_name=input_data.get("customer_name", ""),
            email=input_data.get("email", ""),
            query=input_data.get("query", ""),
            priority=input_data.get("priority", "medium"),
            ticket_id=input_data.get("ticket_id", ""),
            stage_history=[],
            execution_log=[],
            current_stage="",
            completed_at=None
        )
        
        # Execute workflow
        result = self.graph.invoke(initial_state)
        
        return result

# Demo function
def run_demo():
    """Run a demonstration of the customer support agent"""
    print("=" * 60)
    print("🤖 LANGIE - Customer Support Agent Demo")
    print("=" * 60)
    
    # Initialize agent
    agent = CustomerSupportAgent()
    
    # Sample input
    sample_input = {
        "customer_name": "John Smith",
        "email": "john.smith@email.com",
        "query": "I was charged twice for my premium subscription this month",
        "priority": "high",
        "ticket_id": "12345"
    }
    
    print("\n📥 Input Payload:")
    print(json.dumps(sample_input, indent=2))
    
    # Process request
    print("\n🔄 Processing through 11 stages...")
    result = agent.process_customer_request(sample_input)
    
    # Display results
    print("\n📊 Stage Execution Summary:")
    for log_entry in result["execution_log"]:
        print(f"  ✅ {log_entry['stage']} - {', '.join(log_entry['abilities'])}")
    
    print(f"\n🏁 Final Status: {result['ticket_updates']['status']}")
    print(f"📧 Generated Response: {result['generated_response'][:100]}...")
    print(f"🔔 Notifications: {', '.join(result['notifications_sent'])}")
    
    print("\n📋 Complete Output Payload:")
    # Clean output for display
    clean_result = {k: v for k, v in result.items() 
                   if k not in ['execution_log'] and v is not None}
    print(json.dumps(clean_result, indent=2, default=str))

if __name__ == "__main__":
    run_demo()