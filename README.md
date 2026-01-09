[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/llm-use-iot-edge-mcp-server-badge.png)](https://mseep.ai/app/llm-use-iot-edge-mcp-server)

# 🏭 IoT/Edge MCP Server

**Production-ready MCP server for Industrial IoT, Edge Computing and SCADA/PLC systems.**

Secure, enterprise-grade Model Context Protocol (MCP) server that exposes a unified tool interface over HTTP (FastAPI) and integrates MQTT + Modbus with InfluxDB (time-series) and Redis (cache).

Designed to work seamlessly with **[PolyMCP](https://github.com/poly-mcp/Polymcp)** - enabling AI agents (Claude, OpenAI, Ollama, and more) to control industrial infrastructure through natural language.

---

## Features

### Core Capabilities

| Protocol | Description |
|----------|-------------|
| **MQTT** | IoT sensors and actuators (optional TLS) |
| **Modbus TCP/RTU** | PLCs and industrial devices |
| **InfluxDB 2.x** | Time-series data storage |
| **Redis** | High-performance caching |
| **Simulation Mode** | Full testing without hardware |

### Security-First Design (Production Mode)

- **Authentication**: API key (`X-API-Key`) + JWT bearer tokens
- **Access Control**: IP allowlisting (CIDR), rate limiting
- **Data Protection**: Input validation, Fernet encryption, HMAC signatures
- **Audit Trail**: Tamper-evident logging with HMAC chaining

### Industrial Operations

- Real-time sensor monitoring
- Historical queries with aggregation (mean, max, min, sum, count, median)
- Actuator command execution
- PLC register / coil read-write
- Multi-priority alarm system with acknowledge workflow
- Device topology and system status reporting

---

## Requirements

### Simulation Mode (No external dependencies)
| Requirement | Notes |
|-------------|-------|
| Python 3.8+ | 3.9+ recommended |

### Production Mode
| Requirement | Notes |
|-------------|-------|
| Python 3.8+ | 3.9+ recommended |
| MQTT Broker | Optional if using Modbus only |
| InfluxDB 2.0+ | Optional, for historical data |
| Redis | Optional, for caching |
| Modbus Devices | Optional, for PLC integration |

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/iot-mcp-server.git
cd iot-mcp-server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 📦 requirements.txt

```txt
# MCP Server Framework
polymcp>=1.2.6

# Web Framework
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
docstring-parser>=0.15
python-multipart>=0.0.6

# Protocols (Production Mode)
paho-mqtt>=1.6.1
pymodbus>=3.5.2
pyserial>=3.5

# Storage (Production Mode)
redis>=5.0.1
influxdb-client>=1.38.0

# Configuration
pyyaml>=6.0.1

# Security (Production Mode)
bleach>=6.0.0
cryptography>=41.0.0
passlib[bcrypt]>=1.7.4
pyjwt>=2.8.0

# Monitoring (Production Mode)
prometheus-client>=0.19.0
```

### 📦 requirements-sim.txt (Simulation Only)

```txt
# Minimal dependencies for simulation mode
polymcp>=1.2.6
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
docstring-parser>=0.15
```

> ⚠️ **Note**: This project uses Pydantic v2 APIs. Pydantic v1 is not supported.

---

## 🚀 Quick Start

### Option 1: Simulation Mode (No Hardware Required)

Perfect for testing and development:

```bash
python IoT_mcp_sim.py
```

Output:
```
============================================================
IoT/Edge MCP Server - MODALITÀ SIMULAZIONE
============================================================

Dispositivi simulati disponibili:
- 10 Sensori
- 6 Attuatori
- 1 PLC Modbus

Server in ascolto su http://localhost:8000
Documentazione API: http://localhost:8000/docs
============================================================
```

### Option 2: Production Mode (Real Hardware)

```bash
# Set required environment variables
export IOT_ENV=production
export JWT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
export ENCRYPTION_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
export AUDIT_HMAC_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
export MQTT_COMMAND_HMAC_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
export ALLOWED_IP_RANGES="10.0.0.0/8,192.168.0.0/16"
export API_KEYS='{"admin":"your-secure-api-key"}'

# Start server
python IoT_mcp.py
```

**Default bind**: `http://0.0.0.0:8000`

---

## 🤖 Using with PolyMCP

This MCP server is designed to work seamlessly with **[PolyMCP](https://github.com/poly-mcp/Polymcp)** - a powerful framework for orchestrating MCP servers with AI agents.

### Install PolyMCP

```bash
pip install polymcp>=1.2.8
```

### Example: AI-Controlled Industrial System

```python
#!/usr/bin/env python3
"""IoT MCP Chat - Control industrial equipment with AI"""
import asyncio
from polymcp.polyagent import UnifiedPolyAgent, OllamaProvider

async def main():
    # Initialize your LLM provider
    llm = OllamaProvider(model="llama3.1:8b", temperature=0.1)
    
    # Connect to IoT MCP server
    agent = UnifiedPolyAgent(
        llm_provider=llm, 
        mcp_servers=["http://localhost:8000/mcp"],  
        verbose=True
    )
    
    async with agent:
        print("✅ IoT MCP Server connected!\n")
        
        # Chat loop
        while True:
            user_input = input("\n🏭 You: ")
            
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            result = await agent.run_async(user_input, max_steps=5)
            print(f"\n🤖 System: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Using with Different LLM Providers

PolyMCP supports multiple LLM providers out of the box:

#### Ollama (Local)
```python
from polymcp.polyagent import UnifiedPolyAgent, OllamaProvider

llm = OllamaProvider(model="llama3.1:8b", temperature=0.1)
agent = UnifiedPolyAgent(llm_provider=llm, mcp_servers=["http://localhost:8000/mcp"])
```

#### OpenAI
```python
from polymcp.polyagent import UnifiedPolyAgent, OpenAIProvider

llm = OpenAIProvider(model="gpt-4", api_key="your-api-key")
agent = UnifiedPolyAgent(llm_provider=llm, mcp_servers=["http://localhost:8000/mcp"])
```

#### Anthropic Claude
```python
from polymcp.polyagent import UnifiedPolyAgent, AnthropicProvider

llm = AnthropicProvider(model="claude-3-5-sonnet-20241022", api_key="your-api-key")
agent = UnifiedPolyAgent(llm_provider=llm, mcp_servers=["http://localhost:8000/mcp"])
```

### Example Natural Language Commands

Once connected, you can ask the AI agent to:

| Command | What it does |
|---------|--------------|
| *"Check all temperature sensors"* | Reads values from all temp sensors |
| *"What's the average pressure in tank 1 over the last 6 hours?"* | Queries historical data with aggregation |
| *"Open the main valve to 75%"* | Executes actuator command |
| *"Show me all critical alarms"* | Lists active alarms filtered by priority |
| *"Read Modbus registers 0-10 from PLC 01"* | Direct PLC communication |
| *"If temperature exceeds 50°C, alert me"* | Conditional monitoring |
| *"Generate a status report"* | Gets system topology and status |

**That's it!** PolyMCP handles all the complexity of:
- Tool discovery and selection
- Multi-step industrial process automation
- Real-time monitoring and alerting
- Complex decision logic implementation

---

## 📡 MCP HTTP API

This server exposes MCP tools through HTTP endpoints provided by `polymcp-toolkit`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/mcp/list_tools` | GET | List all available tools |
| `/mcp/invoke/{tool_name}` | POST | Invoke a tool (JSON body) |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics (production) |
| `/docs` | GET | OpenAPI documentation |

---

## 🛠️ Available MCP Tools

### 📊 Sensor Operations

| Tool | Parameters | Description |
|------|------------|-------------|
| `read_sensor` | `sensor_id` | Read current value from a sensor |
| `read_multiple_sensors` | `sensor_ids` (list) | Batch read multiple sensors |
| `get_sensor_history` | `sensor_id`, `hours`, `aggregation` | Query historical data (max 168h) |

### ⚡ Actuator Control

| Tool | Parameters | Description |
|------|------------|-------------|
| `execute_actuator_command` | `actuator_id`, `command`, `parameters` | Send command to actuator |

### 🖥️ Device Management

| Tool | Parameters | Description |
|------|------------|-------------|
| `get_device_topology` | - | View complete system architecture |
| `list_devices` | `device_type` (optional) | List devices filtered by type |
| `get_system_status` | - | System health and statistics |

### 🚨 Alarm Management

| Tool | Parameters | Description |
|------|------------|-------------|
| `get_active_alarms` | `priority` (optional) | View active alarms |
| `acknowledge_alarm` | `alarm_id` | Acknowledge an alarm |

### 🔧 Modbus Operations

| Tool | Parameters | Description |
|------|------------|-------------|
| `read_modbus_registers` | `device_id`, `address`, `count` | Read holding registers |
| `write_modbus_register` | `device_id`, `address`, `value` | Write single register |

**Production mode adds:**
| Tool | Parameters | Description |
|------|------------|-------------|
| `read_modbus_coils` | `device_id`, `address`, `count` | Read coils (digital inputs) |
| `write_modbus_coil` | `device_id`, `address`, `value` | Write single coil |

---

## 🎮 Simulated Devices

In simulation mode, the following devices are available for testing:

### Sensors
| ID | Type | Location |
|----|------|----------|
| `temp_sensor_01` | Temperature | production_line_1 |
| `temp_sensor_02` | Temperature | production_line_2 |
| `humidity_sensor_01` | Humidity | room_a |
| `pressure_sensor_01` | Pressure | tank_1 |
| `pressure_sensor_02` | Pressure | tank_2 |
| `flow_sensor_01` | Flow | main_pipe |
| `level_sensor_01` | Level | tank_1 |
| `vibration_sensor_01` | Vibration | motor_1 |
| `current_sensor_01` | Current | motor_1 |
| `voltage_sensor_01` | Voltage | main_panel |

### Actuators
| ID | Type | Commands |
|----|------|----------|
| `valve_01` | Valve | open, close, set_position |
| `valve_02` | Valve | open, close, set_position |
| `pump_01` | Pump | on, off, set_speed |
| `motor_01` | Motor | start, stop, set_speed |
| `motor_02` | Motor | start, stop, set_speed |
| `relay_01` | Relay | on, off |

### PLC
| ID | Type | Registers |
|----|------|-----------|
| `plc_01` | Modbus PLC | 0-99 (simulated) |

---

## 🔧 Production Configuration

### 🔑 Environment Variables

#### Required for Production (`IOT_ENV=production`)

```bash
export IOT_ENV=production
export JWT_SECRET_KEY="your-jwt-secret"
export ENCRYPTION_KEY="your-fernet-key"
export AUDIT_HMAC_KEY="your-audit-hmac-key"
export MQTT_COMMAND_HMAC_KEY="your-mqtt-hmac-key"
```

#### Strongly Recommended

```bash
export ALLOWED_IP_RANGES="10.0.0.0/8,192.168.0.0/16"
export API_KEYS='{"monitoring":"key1","automation":"key2"}'
```

#### Optional Configuration

```bash
# MQTT
export MQTT_BROKER="localhost"
export MQTT_PORT="8883"
export MQTT_USE_TLS="true"
export MQTT_USERNAME="user"
export MQTT_PASSWORD="password"

# InfluxDB
export INFLUX_URL="https://localhost:8086"
export INFLUX_TOKEN="your-token"
export INFLUX_ORG="iot"
export INFLUX_BUCKET="sensors"

# Redis
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
export REDIS_PASSWORD="redis-password"

# Rate Limits
export MAX_REQUESTS_PER_MINUTE="60"
export MAX_COMMANDS_PER_MINUTE="10"
```

### 🔐 Generating Keys

```bash
# ENCRYPTION_KEY (Fernet)
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# JWT_SECRET_KEY / AUDIT_HMAC_KEY / MQTT_COMMAND_HMAC_KEY
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 📄 Configuration File

Create `iot_config.yaml`:

```yaml
mqtt:
  broker: localhost
  port: 8883
  use_tls: true
  ca_cert: /path/to/ca.crt
  client_cert: /path/to/client.crt
  client_key: /path/to/client.key
  username: iot_user
  password_encrypted: null

modbus:
  devices:
    - device_id: plc_01
      type: tcp
      host: 192.168.1.100
      port: 502
      unit: 1
      max_read_registers: 100
      allowed_addresses: [0, 1, 2, 3, 4, 5]

devices:
  mqtt:
    - id: temp_sensor_01
      name: "Temperature Sensor Zone A"
      type: sensor
      topic: sensors/zone_a/temperature
      metadata:
        sensor_type: temperature
        unit: celsius

    - id: valve_01
      name: "Main Water Valve"
      type: actuator
      topic: actuators/valves/main
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         AI Agent                                 │
│              (Claude / OpenAI / Ollama / etc.)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                          PolyMCP                                 │
│                (Tool Discovery & Orchestration)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     IoT MCP Server                               │
│                  (FastAPI + polymcp-toolkit)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  MQTT Connector │ │ Modbus Connector│ │   Data Store    │
│  (TLS, HMAC)    │ │ (TCP/RTU)       │ │ (Influx+Redis)  │
└────────┬────────┘ └────────┬────────┘ └─────────────────┘
         │                   │
         ▼                   ▼
┌─────────────────┐ ┌─────────────────┐
│  IoT Sensors &  │ │  PLCs & RTUs    │
│   Actuators     │ │                 │
└─────────────────┘ └─────────────────┘
```

---

## 🔐 Security Model (Production)

### Request Flow

```
Incoming Request
       │
       ▼
┌─────────────────┐
│ Payload Size    │ → 413 if > 1MB
│ Check           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Public Path?    │ → /health, /metrics allowed
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ IP Allowlist    │ → 403 if not in CIDR range
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Rate Limit      │ → 429 if exceeded
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Authentication  │ → 401 if invalid
│ (API Key / JWT) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Tool Execution  │
└─────────────────┘
```

### Rate Limiting

| Type | Scope | Default Limit |
|------|-------|---------------|
| HTTP Requests | Per IP | 60/minute |
| Actuator Commands | Global | 10/minute |
| Modbus Operations | Per device | 10/minute |

---

## 🐛 Troubleshooting

### Common Errors

| Error | Code | Solution |
|-------|------|----------|
| Service not ready | 503 | Wait for startup; check logs |
| IP not allowed | 403 | Add IP to `ALLOWED_IP_RANGES` |
| Unauthorized | 401 | Check API key or JWT |
| Rate limit exceeded | 429 | Reduce frequency |
| Sensor not found | 400 | Check `list_devices()` for valid IDs |

### Debug Commands

```bash
# Check health
curl http://localhost:8000/health

# List available tools
curl http://localhost:8000/mcp/list_tools

# View logs
tail -f iot_mcp_server.log
```

---

## 💡 Use Cases

| Use Case | Description |
|----------|-------------|
| **Smart Factory** | AI-driven production line optimization |
| **Building Automation** | Intelligent HVAC and lighting control |
| **Energy Management** | Real-time consumption monitoring |
| **Predictive Maintenance** | Equipment failure prediction |
| **Quality Control** | Automated anomaly detection |
| **Emergency Response** | AI-managed incident handling |

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file

---

## 🤝 Contributing

1. Fork the repository
2. Create a branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m "Add my feature"`
4. Push branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 🔗 Related Projects

- **[PolyMCP](https://github.com/poly-mcp/Polymcp)** - AI agent framework for MCP servers
- **[Model Context Protocol](https://modelcontextprotocol.io/)** - Open protocol for AI tool integration
- **[MQTT Protocol](https://mqtt.org/)** - Lightweight IoT messaging
- **[Modbus Protocol](https://modbus.org/)** - Industrial communication standard

---

## 💡 Why This Project?

This MCP server bridges **Industrial IoT** and **AI agents**. With PolyMCP, you can:

1. **Natural Language Control** - "Check all pressure sensors and alert if any are abnormal"
2. **Complex Automation** - AI orchestrates multi-step industrial processes
3. **Predictive Maintenance** - AI analyzes trends and predicts failures
4. **Energy Optimization** - AI optimizes equipment usage
5. **Incident Response** - AI handles alarms and executes emergency procedures

**No complex industrial protocols to learn** - PolyMCP and AI handle everything!

---

> ⚠️ **Production Deployment**: Always use strong secrets, proper network isolation, and TLS termination via reverse proxy.

---

**Designed for [PolyMCP](https://github.com/poly-mcp/Polymcp)** 🚀

*Star ⭐ this repo if you find it useful!*
