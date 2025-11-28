# 5-Day AI Agent Course - High-Level Summary Report

This report provides a high-level overview of learnings from the Kaggle 5-Day AI Agents course using Google's Agent Development Kit (ADK).

---

## Day 1 Learnings

### Part A: From Prompt to Action - Your First AI Agent

**Core Concept**: Transform a static LLM into a dynamic agent that can take actions.

**Key Difference**:
- **Traditional LLM**: `Prompt → Text Response`
- **AI Agent**: `Prompt → Thought → Action → Observation → Answer`

**What You Build**: A simple agent that uses Google Search to answer questions about current events.

**Main Components**:
```python
Agent(
    name="helpful_assistant",
    model=Gemini(model="gemini-2.5-flash-lite"),
    instruction="Use Google Search for current info",
    tools=[google_search],
)
```

**Key Learnings**:
1. Agents autonomously decide when to use tools based on their instructions
2. ADK provides built-in tools like `google_search` out of the box
3. Retry configuration handles rate limits and transient errors
4. `InMemoryRunner` manages agent execution and conversation flow

---

### Part B: Multi-Agent Systems & Workflow Patterns

**Core Concept**: Build teams of specialized agents instead of one monolithic agent.

**Why Multi-Agent?**
- ❌ **Monolithic Agent**: Complex instructions, hard to debug, unreliable
- ✅ **Multi-Agent Team**: Each agent has one job, easier to maintain, more reliable

**Four Workflow Patterns**:

| Pattern | How It Works | Best For |
|---------|-------------|----------|
| **LLM Orchestration** | LLM decides which agents to call | Dynamic, unpredictable workflows |
| **Sequential** | Fixed order execution (A→B→C) | Linear pipelines with dependencies |
| **Parallel** | Run multiple agents simultaneously | Independent tasks, speed optimization |
| **Loop** | Repeat until condition met | Iterative refinement, quality improvement |

**Example - Blog Writing Pipeline** (Sequential):
```
Outline Agent → Writer Agent → Editor Agent → Final Blog
```

**Example - Multi-Topic Research** (Parallel):
```
Tech Researcher ─┐
Health Researcher├─→ Aggregator → Executive Summary
Finance Researcher┘
```

**Example - Story Refinement** (Loop):
```
Writer → Critic → Refiner → Critic → ... → Approved Story
```

**State Management**:
- Agents share data via `output_key` and `{placeholder}` syntax
- Session state persists across agent calls

**Key Learnings**:
1. Specialized agents are easier to build and maintain than monoliths
2. Patterns can be nested (e.g., Sequential containing Parallel)
3. Choose pattern based on task dependencies and execution requirements
4. `AgentTool()` wraps agents so they can be used as tools by other agents

---

## Day 2 Learnings

### Part A: Agent Tools

**Core Concept**: Extend agent capabilities through custom tools - any Python function can become an agent tool.

**Why Tools Matter**:
- Without tools: Knowledge frozen at training time, no real-world actions
- With tools: Access live data, perform calculations, take actions, connect to systems

**Tool Categories**:

**Custom Tools** (You Build):
- **Function Tools**: Regular Python functions → Agent tools
- **Agent Tools**: Use other agents as tools
- **Long-Running Tools**: For async operations requiring approval
- **MCP Tools**: Connect to Model Context Protocol servers
- **OpenAPI Tools**: Auto-generated from API specifications

**Built-in Tools** (Pre-made):
- **Gemini Tools**: `google_search`, `BuiltInCodeExecutor`
- **Google Cloud Tools**: BigQuery, Spanner, API Hub
- **Third-party Tools**: Hugging Face, GitHub, Firecrawl

**What You Build**: Currency converter agent with custom business logic.

**Example - Custom Function Tool**:
```python
def get_fee_for_payment_method(method: str) -> dict:
    """Looks up transaction fee for a payment method."""
    fee_database = {
        "platinum credit card": 0.02,  # 2%
        "bank transfer": 0.01,         # 1%
    }
    return {"status": "success", "fee_percentage": fee_database.get(method)}

# Add to agent
agent = LlmAgent(
    tools=[get_fee_for_payment_method, get_exchange_rate]
)
```

**Best Practices**:
1. Clear docstrings (LLMs read them to understand tool usage)
2. Type hints (enables automatic schema generation)
3. Structured returns with status field
4. Consistent error handling

**Agent as Tool Pattern**:
```python
# Create specialist agent for calculations
calculation_agent = LlmAgent(
    instruction="Generate Python code to calculate",
    code_executor=BuiltInCodeExecutor()
)

# Use it as a tool in main agent
currency_agent = LlmAgent(
    tools=[
        get_fee_for_payment_method,
        AgentTool(agent=calculation_agent)  # Specialist as tool
    ]
)
```

**Why This Matters**: LLMs are unreliable at math; code execution ensures accuracy.

**Key Learnings**:
1. Any Python function can be a tool with proper docstring and type hints
2. `AgentTool()` lets you use specialist agents as tools
3. `BuiltInCodeExecutor` enables reliable calculations via code generation
4. Agent Tools ≠ Sub-Agents (tools return to caller; sub-agents take over control)

---

### Part B: Agent Tool Patterns and Best Practices

**Core Concept**: Advanced patterns for external integrations and human-in-the-loop workflows.

**Pattern 1: Model Context Protocol (MCP)**

**Problem**: Building custom integrations for every external system is time-consuming.

**Solution**: MCP is an open standard - connect to any MCP server with zero integration code.

**Example - Image Generation via MCP**:
```python
mcp_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-everything"],
            tool_filter=["getTinyImage"]
        )
    )
)

agent = LlmAgent(tools=[mcp_server])
```

**How It Works**:
1. ADK launches MCP server via command line
2. Server announces available tools
3. ADK filters and exposes selected tools
4. Agent uses tools like any other function
5. ADK handles all communication

**Other MCP Examples**:
- **Kaggle MCP**: Dataset search, notebook access
- **GitHub MCP**: PR/issue analysis, repository operations
- **Hundreds more** at modelcontextprotocol.io

**Pattern 2: Long-Running Operations (LRO)**

**Problem**: Some operations need to pause and wait for human approval or external events.

**Traditional Flow**: `User → Tool → Result → Agent`

**LRO Flow**: `User → Tool → PAUSE → Human Approves → Tool Resumes → Result → Agent`

**Use Cases**:
- Financial transactions requiring approval
- Bulk operations (delete 1000 records)
- High-cost actions (spin up 50 servers)
- Compliance checkpoints
- Irreversible operations

**Example - Shipping Agent with Approval**:
```python
def place_shipping_order(num_containers: int, destination: str,
                        tool_context: ToolContext) -> dict:
    # Small orders: auto-approve
    if num_containers <= 5:
        return {"status": "approved", "order_id": "AUTO-123"}

    # Large orders: request approval (PAUSE here)
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"Approve {num_containers} containers?"
        )
        return {"status": "pending"}

    # Resuming with approval decision
    if tool_context.tool_confirmation.confirmed:
        return {"status": "approved", "order_id": "HUMAN-456"}
    else:
        return {"status": "rejected"}
```

**Key Components**:

1. **ToolContext**: ADK-provided parameter with:
   - `request_confirmation()`: Pause and ask for approval
   - `tool_confirmation`: Contains approval decision when resuming

2. **Resumable App**: Required to save/restore state
```python
app = App(
    root_agent=agent,
    resumability_config=ResumabilityConfig(is_resumable=True)
)
```

3. **Workflow Detection**: Your code must detect pause and handle resume
```python
# Initial call
events = await runner.run_async(message)

# Detect if paused for approval
if check_for_approval(events):
    # Resume with approval decision
    await runner.run_async(
        approval_message,
        invocation_id=saved_invocation_id  # Critical: same ID to resume
    )
```

**Execution Flow**:
```
TIME 1: User: "Ship 10 containers"
TIME 2: Tool detects large order
TIME 3: Tool calls request_confirmation()
TIME 4: Tool returns {"status": "pending"}
TIME 5: ADK creates pause event
TIME 6: Workflow detects pause, saves invocation_id
TIME 7: Human approves
TIME 8: Workflow resumes with same invocation_id
TIME 9: Tool receives approval decision
TIME 10: Tool returns {"status": "approved"}
TIME 11: Agent responds to user
```

**Critical Concepts**:
- **invocation_id**: Unique ID for each execution; same ID = resume, different ID = new execution
- **Events**: ADK creates events during execution; workflow must inspect them
- **State Persistence**: App saves conversation state when paused, restores when resumed

**Key Learnings**:
1. MCP enables zero-code integration with external services
2. Same MCP pattern works for any MCP server (GitHub, Kaggle, databases, etc.)
3. Long-Running Operations enable human-in-the-loop workflows
4. `ToolContext` provides pause/resume capabilities
5. Resumable Apps maintain state across pause/resume cycles
6. Workflow code must detect pause events and handle resume with correct invocation_id

---

## Day 3 Learnings

### Part A: Agent Sessions

**Core Concept**: Manage conversation state and enable multi-turn interactions with session management.

**Why Sessions Matter**:
- Agents need to remember conversation history
- Multi-turn interactions require context preservation
- Different users need isolated conversations
- Production systems need session lifecycle management

**Session Components**:

1. **Session Service**: Manages session storage and retrieval
   - `InMemorySessionService`: For testing (data lost on restart)
   - `CloudSessionService`: For production (persistent storage)

2. **Session Identifiers**:
   - `app_name`: Application identifier
   - `user_id`: User identifier
   - `session_id`: Unique conversation identifier

**Basic Session Pattern**:
```python
session_service = InMemorySessionService()

# Create session
await session_service.create_session(
    app_name="my_app",
    user_id="user_123",
    session_id="session_456"
)

# Use with runner
runner = Runner(agent=agent, session_service=session_service)
```

**Key Learnings**:
1. Sessions enable stateful, multi-turn conversations
2. Session services handle storage and retrieval automatically
3. Use InMemorySessionService for development, CloudSessionService for production
4. Each session maintains isolated conversation history

---

### Part B: Agent Memory

**Core Concept**: Long-term memory across sessions enables personalized agent experiences.

**Memory Types**:

| Type | Scope | Lifespan | Use Case |
|------|-------|----------|----------|
| **Short-term** | Single session | Until session ends | "What did I just say?" |
| **Long-term** | All sessions | Permanent | "What's my favorite color?" |
| **Shared** | Multiple agents | Persistent | Team collaboration data |

**Short-Term Memory (Session State)**:
- Automatic conversation history within a session
- Managed by `output_key` and session service
- Cleared when session ends

**Long-Term Memory (Memory Bank)**:
- Powered by Vertex AI Memory Bank
- Persists facts across all sessions
- Automatically extracts and recalls user preferences

**Memory Tools**:

1. **PreloadMemoryTool**: Automatically loads relevant memories before agent responds
```python
from google.adk.tools import PreloadMemoryTool

agent = LlmAgent(
    tools=[PreloadMemoryTool(memory_service=memory_service)]
)
```

2. **LoadMemoryTool**: Agent explicitly searches memories when needed
```python
from google.adk.tools import LoadMemoryTool

agent = LlmAgent(
    tools=[LoadMemoryTool(memory_service=memory_service)]
)
```

**How Memory Bank Works**:
1. User talks to agent across multiple sessions
2. Agent Engine extracts key facts ("User prefers Celsius")
3. Facts stored in Memory Bank with embeddings
4. Next session: Memory tools retrieve relevant facts
5. Agent uses recalled information automatically

**Example Flow**:
```
Session 1: User: "I prefer metric units"
           → Memory Bank saves: "User prefers metric system"

Session 2: User: "What's the weather?"
(days later) → Memory Bank recalls preference
           → Agent responds in Celsius automatically
```

**Key Learnings**:
1. Short-term memory = session history (automatic)
2. Long-term memory = Memory Bank (requires setup)
3. Use PreloadMemoryTool for automatic recall
4. Use LoadMemoryTool for explicit memory search
5. Memory Bank enables truly personalized agents

---

## Day 4 Learnings

### Part A: Agent Observability

**Core Concept**: Visibility into agent decision-making for debugging and monitoring.

**The Problem**:
```
User: "Find quantum computing papers"
Agent: "I cannot help with that request."
You: 😭 WHY?? Is it the prompt? Missing tools? API error?
```

**The Solution**: Three pillars of observability

1. **Logs**: What happened at a specific moment
   - Individual events (tool calls, LLM requests, errors)
   - Detailed request/response data

2. **Traces**: Why the final result occurred
   - Complete sequence of steps
   - Timing information for each operation
   - Parent-child relationships between operations

3. **Metrics**: How well the agent performs overall
   - Success rates, latency, error counts
   - Aggregated statistics over time

**Debugging with ADK Web UI**:

```bash
adk web --log_level DEBUG
```

**Features**:
- **Events Tab**: Chronological list of all agent actions
- **Trace View**: Visual timeline with timing data
- **Function Inspection**: See exact parameters passed to tools
- **LLM Request/Response**: Full prompts and model outputs

**Debugging Pattern**:
```
Symptom → Logs → Root Cause → Fix
```

**Production Observability with Plugins**:

**What are Plugins?**: Custom code that runs at specific lifecycle points

**Plugin Structure**:
```python
class MyPlugin(BasePlugin):
    async def before_agent_callback(self, *, agent, callback_context):
        # Runs before agent starts
        logging.info(f"Agent {agent.name} starting")

    async def after_tool_callback(self, *, tool_result, callback_context):
        # Runs after tool completes
        logging.info(f"Tool returned: {tool_result}")
```

**Built-in LoggingPlugin**:
```python
from google.adk.plugins.logging_plugin import LoggingPlugin

runner = InMemoryRunner(
    agent=agent,
    plugins=[LoggingPlugin()]  # Automatic logging for everything
)
```

**When to Use What**:
- **Development debugging**: `adk web --log_level DEBUG`
- **Production observability**: `LoggingPlugin()`
- **Custom requirements**: Build custom plugins

**Key Learnings**:
1. Observability = Logs + Traces + Metrics
2. ADK Web UI shows complete execution flow
3. Plugins enable production-scale logging
4. LoggingPlugin handles standard observability automatically
5. Custom plugins for specialized monitoring needs

---

### Part B: Agent Evaluation

**Core Concept**: Systematic testing to ensure agent quality and catch regressions.

**The Problem**:
- Agents are non-deterministic (different responses each time)
- Users give unpredictable, ambiguous commands
- Small prompt changes cause dramatic behavior shifts
- Traditional testing doesn't work for AI agents

**The Solution**: Systematic evaluation of both response quality and decision-making process.

**Evaluation Metrics**:

1. **Response Match Score**: Text similarity between actual and expected response
   - 1.0 = Perfect match
   - 0.0 = Completely different
   - Threshold: Typically 0.8 (80% similarity)

2. **Tool Trajectory Score**: Correctness of tool usage
   - 1.0 = Perfect tool calls with correct parameters
   - 0.0 = Wrong tools or parameters
   - Checks entire sequence of tool calls

**Interactive Evaluation (ADK Web UI)**:

1. Have normal conversations with agent
2. Save successful interactions as test cases
3. Run evaluation to verify agent can replicate behavior
4. See pass/fail results with detailed diffs

**Systematic Evaluation (CLI)**:

**File Structure**:
```
agent/
├── agent.py
├── test_config.json           # Pass/fail thresholds
└── integration.evalset.json   # Test cases
```

**test_config.json**:
```json
{
  "criteria": {
    "tool_trajectory_avg_score": 1.0,  # Perfect tool usage
    "response_match_score": 0.8         # 80% text match
  }
}
```

**integration.evalset.json**:
```json
{
  "eval_set_id": "test_suite",
  "eval_cases": [
    {
      "eval_id": "test_case_1",
      "conversation": [
        {
          "user_content": {"parts": [{"text": "User query"}]},
          "final_response": {"parts": [{"text": "Expected response"}]},
          "intermediate_data": {
            "tool_uses": [
              {"name": "tool_name", "args": {"param": "value"}}
            ]
          }
        }
      ]
    }
  ]
}
```

**Run Evaluation**:
```bash
adk eval agent_dir/ integration.evalset.json --config_file_path=test_config.json
```

**Analyzing Results**:
```
Test Case: test_case_1
  ✅ tool_trajectory_avg_score: 1.0/1.0 (Perfect)
  ❌ response_match_score: 0.45/0.80 (Failed)

Insight: Tool usage correct, but communication style needs work
Fix: Update agent instructions for clearer language
```

**Advanced: User Simulation**:
- Uses LLM to generate dynamic, varied user prompts
- Tests agent with unpredictable conversation flows
- Uncovers edge cases static tests miss

**Key Learnings**:
1. Evaluation tests both final response AND decision-making process
2. Use ADK Web UI for interactive test creation
3. Use `adk eval` CLI for automated regression testing
4. Response Match = communication quality
5. Tool Trajectory = technical correctness
6. User Simulation for dynamic, realistic testing

---

## Day 5 Learnings

### Part A: Agent2Agent (A2A) Communication

**Core Concept**: Enable agents to communicate and collaborate across networks, frameworks, and organizations.

**The Problem**:
- A single agent can't do everything
- Different teams build different specialized agents
- Agents may use different languages/frameworks
- Need standard way for agents to collaborate

**The Solution**: Agent2Agent (A2A) Protocol - a standard for agent communication.

**A2A Protocol Benefits**:
- ✨ Communicate over networks (agents on different machines)
- ✨ Use each other's capabilities (one agent calls another)
- ✨ Work across frameworks (language/framework agnostic)
- ✨ Formal contracts (agent cards describe capabilities)

**Common A2A Patterns**:

| Pattern | Description | Example |
|---------|-------------|---------|
| **Cross-Framework** | ADK agent ↔ Other framework agent | ADK ↔ LangChain |
| **Cross-Language** | Python ↔ Java/Node.js | Python agent → Java service |
| **Cross-Organization** | Internal ↔ External vendor | Your agent → Vendor's catalog |

**Exposing an Agent via A2A**:

```python
from google.adk.a2a.utils.agent_to_a2a import to_a2a

# Convert agent to A2A-compatible application
product_catalog_a2a = to_a2a(
    product_catalog_agent,
    port=8001
)

# This creates:
# - FastAPI/Starlette server
# - Auto-generated agent card at /.well-known/agent-card.json
# - A2A protocol endpoints (/tasks)
```

**Agent Card**: JSON document describing agent capabilities
- Name, description, version
- Available skills (tools/functions)
- Protocol version, endpoints
- Input/output modes

**Consuming a Remote Agent**:

```python
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

# Create client-side proxy
remote_catalog = RemoteA2aAgent(
    name="product_catalog",
    description="Remote product catalog agent",
    agent_card="http://localhost:8001/.well-known/agent-card.json"
)

# Use it like a local sub-agent
support_agent = LlmAgent(
    sub_agents=[remote_catalog]  # Just add to sub_agents!
)
```

**Example Architecture**:
```
┌──────────────────────┐           ┌──────────────────────┐
│ Customer Support     │  ─A2A──▶  │ Product Catalog      │
│ Agent (Consumer)     │           │ Agent (Vendor)       │
│ Your Company         │           │ External Service     │
│ (localhost:8000)     │           │ (localhost:8001)     │
└──────────────────────┘           └──────────────────────┘
```

**Communication Flow**:
1. Customer asks Support Agent a question
2. Support Agent calls remote Product Catalog Agent
3. `RemoteA2aAgent` sends HTTP POST to `/tasks` endpoint
4. Product Catalog Agent processes request
5. Response returned via A2A protocol
6. Support Agent receives data and continues
7. Customer gets final answer

**Behind the Scenes**:
- HTTP POST requests to `/tasks` endpoint
- JSON request/response following A2A spec
- Standardized format works across any A2A-compatible agent
- Transparent to the consumer agent

**Key Learnings**:
1. A2A enables agent-to-agent collaboration across boundaries
2. `to_a2a()` exposes agents with auto-generated agent cards
3. `RemoteA2aAgent` consumes remote agents transparently
4. Use for microservices, third-party integrations, cross-language systems
5. Agent cards define formal contracts between agents

---

### Part B: Agent Deployment

**Core Concept**: Deploy agents to production with Vertex AI Agent Engine for scalable, managed hosting.

**The Problem**:
- Agent only works on your machine
- Stops when notebook session ends
- Not accessible to teammates or users
- No production-grade infrastructure

**The Solution**: Deploy to Vertex AI Agent Engine - fully managed agent hosting.

**Deployment Options**:

| Platform | Best For | Key Features |
|----------|----------|--------------|
| **Agent Engine** | Production agents | Auto-scaling, session management, Memory Bank |
| **Cloud Run** | Serverless demos | Easy start, pay-per-use |
| **GKE** | Complex systems | Full control, multi-agent orchestration |

**Agent Deployment Process**:

**1. Create Agent Directory**:
```
sample_agent/
├── agent.py                  # Agent logic
├── requirements.txt          # Dependencies
├── .env                      # Configuration
└── .agent_engine_config.json # Resource specs
```

**2. Configure Resources** (.agent_engine_config.json):
```json
{
  "min_instances": 0,        # Scale to zero when idle
  "max_instances": 1,        # Max concurrent instances
  "resource_limits": {
    "cpu": "1",              # 1 CPU core
    "memory": "1Gi"          # 1 GB RAM
  }
}
```

**3. Deploy with CLI**:
```bash
adk deploy agent_engine \
  --project=YOUR_PROJECT_ID \
  --region=us-east4 \
  sample_agent/
```

**4. Test Deployed Agent**:
```python
import vertexai
from vertexai import agent_engines

# Initialize and retrieve agent
vertexai.init(project=PROJECT_ID, location=REGION)
agents = list(agent_engines.list())
remote_agent = agents[0]

# Query the agent
async for item in remote_agent.async_stream_query(
    message="What is the weather in Tokyo?",
    user_id="user_123"
):
    print(item)
```

**Agent Engine Features**:

1. **Auto-scaling**: Scales up/down based on demand
2. **Session Management**: Built-in conversation state handling
3. **Memory Bank Integration**: Long-term memory across sessions
4. **Monitoring**: Integrated with Cloud Logging, Trace, Monitoring
5. **Free Tier**: Monthly free tier for getting started

**Memory Bank with Agent Engine**:

**What It Provides**:
- Long-term memory across all sessions
- Automatic fact extraction from conversations
- Intelligent recall of relevant information
- Personalized agent experiences

**How to Enable**:
1. Add memory tools to agent (`PreloadMemoryTool`)
2. Configure Memory Bank in agent code
3. Redeploy agent
4. Memory Bank works automatically

**Example**:
```
Session 1: User: "I prefer Celsius"
           → Memory Bank extracts and saves preference

Session 2: User: "Weather in Paris?"
(days later) → Memory Bank recalls Celsius preference
           → Agent responds: "15°C" (not Fahrenheit)
```

**Cost Management**:

**Free Tier**: Agent Engine offers monthly free tier

**Best Practices**:
- Set `min_instances: 0` to scale to zero
- Delete test deployments when finished
- Monitor usage in Cloud Console
- Use appropriate instance sizes

**Cleanup**:
```python
agent_engines.delete(
    resource_name=remote_agent.resource_name,
    force=True
)
```

**Key Learnings**:
1. Agent Engine provides production-ready agent hosting
2. Deploy with single CLI command: `adk deploy agent_engine`
3. Auto-scaling and session management built-in
4. Memory Bank enables long-term personalization
5. Always delete test deployments to avoid costs
6. Monitor via Vertex AI Console

---

## Summary

### Course Progress

**Day 1 - Foundations**:
- Built first AI agent with tools (Google Search)
- Learned multi-agent architectures
- Mastered 4 workflow patterns (Sequential, Parallel, Loop, LLM Orchestration)

**Day 2 - Advanced Tools**:
- Created custom function tools with best practices
- Used agents as tools for specialization
- Integrated external systems via MCP
- Implemented human-in-the-loop workflows with LRO

**Day 3 - Context & Memory**:
- Managed conversation state with sessions
- Implemented short-term and long-term memory
- Configured Memory Bank for personalized experiences
- Learned session lifecycle management

**Day 4 - Quality & Reliability**:
- Implemented observability with logs, traces, and metrics
- Used ADK Web UI for interactive debugging
- Created systematic evaluation tests
- Measured agent quality with Response Match and Tool Trajectory scores

**Day 5 - Production Deployment**:
- Enabled agent-to-agent communication with A2A protocol
- Deployed agents to Vertex AI Agent Engine
- Configured auto-scaling and resource management
- Integrated Memory Bank for production personalization

### Key Concepts Mastered

1. **Agent = LLM + Tools + Instructions**: Tools transform LLMs into action-taking agents
2. **Multi-Agent > Monolith**: Specialized agent teams beat do-it-all agents
3. **Workflow Patterns**: Sequential (order), Parallel (speed), Loop (refinement), LLM (dynamic)
4. **Custom Tools**: Any Python function can be a tool with proper docstring/types
5. **Code Execution**: Use `BuiltInCodeExecutor` for reliable calculations
6. **MCP**: Standard protocol for external integrations (zero custom code)
7. **LRO**: Pausable workflows for human approval and compliance
8. **Sessions**: Manage conversation state and multi-turn interactions
9. **Memory Bank**: Long-term personalization across all sessions
10. **Observability**: Logs + Traces + Metrics for debugging and monitoring
11. **Evaluation**: Systematic testing with Response Match and Tool Trajectory
12. **A2A Protocol**: Cross-framework, cross-organization agent communication
13. **Production Deployment**: Scalable hosting with Agent Engine

### Production-Ready Patterns

You now know how to build agents that:
- ✅ Scale with community tools (MCP)
- ✅ Handle time-spanning operations (LRO)
- ✅ Ensure compliance (human-in-the-loop)
- ✅ Maintain state (sessions and resumability)
- ✅ Compose complex workflows (nested patterns)
- ✅ Remember user preferences (Memory Bank)
- ✅ Debug systematically (observability)
- ✅ Test quality (evaluation)
- ✅ Collaborate across organizations (A2A)
- ✅ Deploy to production (Agent Engine)

### Quick Reference

**Agent Creation**:
```python
agent = LlmAgent(
    name="agent_name",
    model=Gemini(model="gemini-2.5-flash-lite"),
    instruction="Clear instructions",
    tools=[tool1, tool2, AgentTool(specialist)],
    output_key="result"  # For multi-agent state sharing
)
```

**Session Management**:
```python
session_service = InMemorySessionService()
await session_service.create_session(
    app_name="my_app",
    user_id="user_123",
    session_id="session_456"
)
runner = Runner(agent=agent, session_service=session_service)
```

**Memory Integration**:
```python
from google.adk.tools import PreloadMemoryTool

agent = LlmAgent(
    tools=[PreloadMemoryTool(memory_service=memory_service)]
)
```

**Workflow Patterns**:
- Sequential: `SequentialAgent(sub_agents=[a, b, c])`
- Parallel: `ParallelAgent(sub_agents=[a, b, c])`
- Loop: `LoopAgent(sub_agents=[a, b], max_iterations=3)`
- LLM: `Agent(tools=[AgentTool(a), AgentTool(b)])`

**Tool Types**:
- Function: Regular Python function
- Agent: `AgentTool(agent=specialist)`
- MCP: `McpToolset(connection_params=...)`
- Code: `code_executor=BuiltInCodeExecutor()`

**Long-Running Operations**:
```python
def my_tool(param: str, tool_context: ToolContext) -> dict:
    if needs_approval:
        if not tool_context.tool_confirmation:
            tool_context.request_confirmation(hint="Approve?")
            return {"status": "pending"}
        return {"status": "approved" if tool_context.tool_confirmation.confirmed else "rejected"}
    return {"status": "completed"}
```

**Observability**:
```python
# Development
adk web --log_level DEBUG

# Production
runner = InMemoryRunner(
    agent=agent,
    plugins=[LoggingPlugin()]
)
```

**Evaluation**:
```bash
adk eval agent_dir/ test.evalset.json --config_file_path=test_config.json
```

**A2A Communication**:
```python
# Expose agent
app = to_a2a(agent, port=8001)

# Consume remote agent
remote_agent = RemoteA2aAgent(
    name="remote_agent",
    agent_card="http://localhost:8001/.well-known/agent-card.json"
)
```

**Deployment**:
```bash
adk deploy agent_engine --project=PROJECT_ID --region=REGION agent_dir/
```

### Resources

- **ADK Documentation**: https://google.github.io/adk-docs/
- **MCP Servers**: https://modelcontextprotocol.io/examples
- **Kaggle Course**: https://www.kaggle.com/learn/5-day-agents
- **A2A Protocol**: https://a2a-protocol.org/
- **Agent Engine**: https://cloud.google.com/agent-builder/agent-engine/overview

---

**End of Report**
