[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/llm-use-iot-edge-mcp-server-badge.png)](https://mseep.ai/app/llm-use-iot-edge-mcp-server)

# IoT/Edge MCP Server

**Model Context Protocol (MCP) server for Industrial IoT, Edge Computing and Automation**

This server transforms industrial infrastructure into an AI-orchestrable system, exposing 11 powerful tools for complete IoT/SCADA/PLC control via HTTP endpoints. Perfect for AI-driven industrial automation, predictive maintenance, and smart factory operations.

## ✨ Features

- **🏭 Multi-Protocol Support** - Unified interface for industrial systems:
  - MQTT for wireless IoT sensors and actuators
  - Modbus TCP/RTU for PLC and industrial devices
  - Time-series data with InfluxDB integration
  - Real-time caching with Redis
  - Simulated mode for testing without hardware

- **📊 Complete Sensor Management** - Monitor and analyze industrial data:
  - Real-time sensor readings (temperature, pressure, flow, etc.)
  - Historical data with aggregations (mean, max, min)
  - Multi-sensor batch operations
  - Quality indicators and signal monitoring

- **⚡ Actuator Control** - Command industrial equipment:
  - Valve control (open/close)
  - Motor management (start/stop/speed)
  - Pump operations
  - PLC register manipulation

- **🚨 Alarm System** - Enterprise-grade monitoring:
  - Multi-priority alarms (LOW, MEDIUM, HIGH, CRITICAL)
  - Automatic threshold monitoring
  - Alarm acknowledgment tracking
  - Real-time notifications

- **🔒 Production Features** - Enterprise-ready:
  - Rate limiting for command safety
  - Comprehensive error handling
  - Connection pooling and auto-reconnect
  - Full audit logging
  - Thread-safe operations

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/llm-use/iot-mcp-server.git
cd iot-mcp-server

# Install dependencies
pip install -r requirements.txt

# For testing without hardware, use simulation mode
pip install -r requirements-sim.txt
```

### Starting the Server

**Option 1: Simulation Mode (No Hardware Required)**
```bash
python IoT_mcp_sim.py
```

**Option 2: Production Mode (Real Hardware)**
```bash
# Configure your devices in iot_config.yaml
# Start required services (MQTT, InfluxDB, Redis)
docker-compose up -d

# Start the server
python IoT_mcp.py
```

Server will start on `http://localhost:8000`

## 🤖 Using with PolyMCP

This MCP server is designed to work seamlessly with **[PolyMCP](https://github.com/llm-use/Polymcp)** - a powerful framework for orchestrating MCP servers with AI agents.

### Example: AI-Controlled Industrial System

```python
#!/usr/bin/env python3
import asyncio
from polymcp.polyagent import UnifiedPolyAgent, OllamaProvider

async def main():
    # Initialize your LLM provider
    llm = OllamaProvider(model="gpt-oss:120b-cloud", temperature=0.1)
    
    # Connect to IoT MCP server
    agent = UnifiedPolyAgent(
        llm_provider=llm, 
        mcp_servers=["http://localhost:8000/mcp"],  
        verbose=True
    )
    
    async with agent:
        print("✅ IoT MCP Server connected!\n")
        
        # Chat with your AI to control the industrial system
        while True:
            user_input = input("\n🏭 You: ")
            
            if user_input.lower() in ['exit', 'quit']:
                break
            
            result = await agent.run_async(user_input, max_steps=5)
            print(f"\n🤖 System: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Example Commands

Once connected, you can ask the AI agent to:

- *"Check the status of all temperature sensors and report any anomalies"*
- *"What's the average pressure in tank 1 over the last 6 hours?"*
- *"Open the main valve and start pump 1"*
- *"Show me all critical alarms that haven't been acknowledged"*
- *"If temperature exceeds 50°C, activate cooling system"*
- *"Generate an end-of-shift report with key metrics"*
- *"Monitor vibration sensor and alert if it exceeds normal range"*
- *"Optimize energy consumption by analyzing motor usage patterns"*

**That's it!** PolyMCP handles all the complexity of:
- Tool discovery and selection
- Multi-step industrial process automation
- Real-time monitoring and alerting
- Complex decision logic implementation

## 📡 API Endpoints

Once the server is running, you can access:

- **API Documentation**: `http://localhost:8000/docs`
- **List All Tools**: `http://localhost:8000/mcp/list_tools`
- **Invoke Tool**: `POST http://localhost:8000/mcp/invoke/{tool_name}`

## 🛠️ Available Tools

<details>
<summary>View all available tools (11 tools)</summary>

### Sensor Operations
- **`read_sensor`** - Read current value from a single sensor
- **`read_multiple_sensors`** - Batch read multiple sensors
- **`get_sensor_history`** - Retrieve historical data with optional aggregation

### Actuator Control
- **`execute_actuator_command`** - Send commands to actuators (valves, motors, pumps)

### System Management
- **`get_device_topology`** - View complete system architecture
- **`list_devices`** - List all registered devices with status
- **`get_system_status`** - Overall system health and statistics

### Alarm Management
- **`get_active_alarms`** - View active alarms by priority
- **`acknowledge_alarm`** - Confirm alarm acknowledgment

### PLC Operations
- **`read_modbus_registers`** - Read PLC registers via Modbus
- **`write_modbus_register`** - Write values to PLC registers

</details>

## 🔧 Configuration

### Basic Configuration (`iot_config.yaml`)

```yaml
mqtt:
  broker: "localhost"
  port: 1883
  username: "iot_user"
  password: "secure_password"

modbus:
  devices:
    - device_id: "plc_01"
      name: "Main PLC"
      type: "tcp"
      host: "192.168.1.100"
      port: 502

influxdb:
  url: "http://localhost:8086"
  token: "your-token"
  org: "iot"
  bucket: "sensors"

redis:
  host: "localhost"
  port: 6379
```

### Environment Variables

```bash
export MQTT_BROKER="broker.hivemq.com"
export INFLUX_TOKEN="your-token"
export REDIS_HOST="localhost"
```

## 📋 Requirements

**For Production Mode:**
- Python 3.8+
- MQTT Broker (Mosquitto, EMQX, HiveMQ, or any MQTT 3.1.1/5.0 broker)
- InfluxDB 2.0+ (for time-series data)
- Redis (for caching)
- Industrial devices (PLCs with Modbus, IoT sensors)

**For Simulation Mode:**
- Python 3.8+
- No external dependencies!

## 🐛 Troubleshooting

**Can't connect to MQTT?**
- Check broker is running: `mosquitto_sub -h localhost -t '#'`
- Verify credentials in config
- Check firewall settings

**Modbus connection failed?**
- Ensure PLC is accessible: `ping <plc_ip>`
- Verify Modbus is enabled on device
- Check port (usually 502 for TCP)

**No sensor data?**
- Check MQTT topics match configuration
- Verify sensor is publishing data
- Look at Redis cache: `redis-cli get sensor:*`

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ IoT Sensors │────▶│  MQTT       │────▶│             │
└─────────────┘     │  Broker     │     │             │
                    └─────────────┘     │   IoT MCP   │     ┌─────────────┐
┌─────────────┐     ┌─────────────┐     │   Server    │────▶│  PolyMCP    │
│    PLCs     │────▶│  Modbus     │────▶│             │     │  AI Agent   │
└─────────────┘     └─────────────┘     │             │     └─────────────┘
                                        │             │
┌─────────────┐                         │             │     ┌─────────────┐
│  InfluxDB   │◀────────────────────────│             │────▶│   Redis     │
└─────────────┘                         └─────────────┘     └─────────────┘
```

## 🤝 Contributing

Contributions are welcome! This project demonstrates industrial IoT integration with MCP protocol.

### Development Setup

```bash
# Clone repo
git clone https://github.com/llm-use/iot-mcp-server.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

## 📝 License

MIT License - See LICENSE file for details

## 🔗 Related Projects

- **[PolyMCP](https://github.com/llm-use/Polymcp)** - Simple and efficient way to interact with MCP servers using custom agents
- **[Model Context Protocol](https://modelcontextprotocol.io/)** - Open protocol for tool integration with LLMs
- **[MQTT Protocol](https://mqtt.org/)** - Lightweight messaging protocol for IoT
- **[Modbus Protocol](https://modbus.org/)** - Industrial communication protocol
- **[Mqttcpp](https://github.com/JustVugg/Mqttcpp)** - A lightweight and fast C++ library for building MQTT clients and brokers


## 💡 Why This Project?

This MCP server bridges the gap between **Industrial IoT** and **AI agents**. With PolyMCP, you can:

1. **Natural Language Control** - "Check all pressure sensors and alert if any are abnormal"
2. **Complex Automation** - AI can orchestrate multi-step industrial processes
3. **Predictive Maintenance** - AI analyzes trends and predicts failures
4. **Energy Optimization** - AI optimizes equipment usage for efficiency
5. **Incident Response** - AI handles alarms and executes emergency procedures

No complex industrial protocols to learn - PolyMCP and AI handle everything!

## 🚀 Use Cases

- **Smart Factory** - AI-driven production line optimization
- **Building Automation** - Intelligent HVAC and lighting control
- **Energy Management** - Real-time consumption optimization
- **Predictive Maintenance** - Equipment failure prediction
- **Quality Control** - Automated anomaly detection
- **Emergency Response** - AI-managed incident handling

---

**Designed for [PolyMCP](https://github.com/llm-use/Polymcp)**

*Star ⭐ this repo if you find it useful!*
