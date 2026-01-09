"""
IoT/Edge MCP Server - VERSIONE SIMULATA PER TEST
Funziona senza dipendenze esterne, simula sensori e attuatori
"""

import asyncio
import json
import logging
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque

# Framework expose_tools 
from polymcp.polymcp_toolkit import expose_tools

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== MODELLI DATI ====================

class SensorType(Enum):
    """Tipi di sensori supportati"""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    FLOW = "flow"
    LEVEL = "level"
    VIBRATION = "vibration"
    CURRENT = "current"
    VOLTAGE = "voltage"
    POWER = "power"
    SPEED = "speed"

class ActuatorType(Enum):
    """Tipi di attuatori supportati"""
    VALVE = "valve"
    PUMP = "pump"
    MOTOR = "motor"
    RELAY = "relay"

class AlarmPriority(Enum):
    """Priorità allarmi"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class SensorReading:
    """Lettura da sensore"""
    sensor_id: str
    timestamp: datetime
    value: float
    unit: str
    quality: int  # 0-100, qualità del segnale

@dataclass
class Alarm:
    """Allarme sistema"""
    alarm_id: str
    sensor_id: str
    timestamp: datetime
    priority: AlarmPriority
    message: str
    threshold_value: float
    actual_value: float
    acknowledged: bool = False

@dataclass
class Device:
    """Dispositivo IoT/Edge"""
    device_id: str
    name: str
    type: str  # "sensor", "actuator", "plc", "gateway"
    protocol: str  # "mqtt", "modbus", "opcua"
    address: str
    status: str  # "online", "offline", "error"
    last_seen: datetime
    metadata: Dict[str, Any] = None

# ==================== SIMULATORI ====================

class SensorSimulator:
    """Simula valori sensori realistici"""
    
    def __init__(self):
        self.base_values = {
            SensorType.TEMPERATURE: 25.0,
            SensorType.HUMIDITY: 60.0,
            SensorType.PRESSURE: 1.0,
            SensorType.FLOW: 50.0,
            SensorType.LEVEL: 70.0,
            SensorType.VIBRATION: 0.1,
            SensorType.CURRENT: 10.0,
            SensorType.VOLTAGE: 220.0,
            SensorType.POWER: 2200.0,
            SensorType.SPEED: 1500.0
        }
        
        self.noise_levels = {
            SensorType.TEMPERATURE: 2.0,
            SensorType.HUMIDITY: 5.0,
            SensorType.PRESSURE: 0.1,
            SensorType.FLOW: 5.0,
            SensorType.LEVEL: 3.0,
            SensorType.VIBRATION: 0.05,
            SensorType.CURRENT: 0.5,
            SensorType.VOLTAGE: 5.0,
            SensorType.POWER: 100.0,
            SensorType.SPEED: 50.0
        }
        
        self.units = {
            SensorType.TEMPERATURE: "°C",
            SensorType.HUMIDITY: "%",
            SensorType.PRESSURE: "bar",
            SensorType.FLOW: "l/min",
            SensorType.LEVEL: "%",
            SensorType.VIBRATION: "mm/s",
            SensorType.CURRENT: "A",
            SensorType.VOLTAGE: "V",
            SensorType.POWER: "W",
            SensorType.SPEED: "rpm"
        }
        
        self.time_offset = 0
    
    def generate_value(self, sensor_type: SensorType) -> float:
        """Genera valore realistico con rumore e trend"""
        base = self.base_values[sensor_type]
        noise = self.noise_levels[sensor_type]
        
        # Aggiungi componente sinusoidale per simulare variazioni giornaliere
        time_factor = math.sin(self.time_offset / 100) * 0.1
        self.time_offset += 1
        
        # Aggiungi rumore random
        random_noise = random.gauss(0, noise * 0.3)
        
        # Occasionalmente genera spike anomali (5% probabilità)
        if random.random() < 0.05:
            random_noise *= 3
        
        value = base * (1 + time_factor) + random_noise
        
        # Limita valori a range realistici
        if sensor_type == SensorType.HUMIDITY or sensor_type == SensorType.LEVEL:
            value = max(0, min(100, value))
        elif sensor_type == SensorType.PRESSURE:
            value = max(0, value)
        
        return round(value, 2)
    
    def get_unit(self, sensor_type: SensorType) -> str:
        """Ottiene unità di misura per tipo sensore"""
        return self.units.get(sensor_type, "")

class MockDatabase:
    """Database in memoria per test"""
    
    def __init__(self):
        self.sensor_history = {}  # sensor_id -> deque di readings
        self.max_history = 1000
        self.active_alarms = {}
        self.command_log = []
    
    def store_reading(self, reading: SensorReading):
        """Salva lettura in memoria"""
        if reading.sensor_id not in self.sensor_history:
            self.sensor_history[reading.sensor_id] = deque(maxlen=self.max_history)
        
        self.sensor_history[reading.sensor_id].append({
            'timestamp': reading.timestamp,
            'value': reading.value,
            'unit': reading.unit,
            'quality': reading.quality
        })
    
    def get_history(
        self, 
        sensor_id: str, 
        start_time: datetime, 
        end_time: datetime,
        aggregation: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Ottiene storico da memoria"""
        if sensor_id not in self.sensor_history:
            return []
        
        # Filtra per periodo
        history = [
            h for h in self.sensor_history[sensor_id]
            if start_time <= h['timestamp'] <= end_time
        ]
        
        if not history:
            return []
        
        # Applica aggregazione se richiesta
        if aggregation:
            if aggregation == 'mean':
                avg_value = sum(h['value'] for h in history) / len(history)
                return [{
                    'timestamp': end_time.isoformat(),
                    'value': round(avg_value, 2),
                    'sensor_id': sensor_id,
                    'aggregation': 'mean'
                }]
            elif aggregation == 'max':
                max_reading = max(history, key=lambda x: x['value'])
                return [{
                    'timestamp': max_reading['timestamp'].isoformat(),
                    'value': max_reading['value'],
                    'sensor_id': sensor_id,
                    'aggregation': 'max'
                }]
            elif aggregation == 'min':
                min_reading = min(history, key=lambda x: x['value'])
                return [{
                    'timestamp': min_reading['timestamp'].isoformat(),
                    'value': min_reading['value'],
                    'sensor_id': sensor_id,
                    'aggregation': 'min'
                }]
        
        # Ritorna dati raw
        return [
            {
                'timestamp': h['timestamp'].isoformat(),
                'value': h['value'],
                'sensor_id': sensor_id
            }
            for h in history
        ]

# ==================== MANAGER PRINCIPALE ====================

class IoTManagerSimulated:
    """Manager IoT con simulazione per test"""
    
    def __init__(self):
        self.simulator = SensorSimulator()
        self.database = MockDatabase()
        self.devices = {}
        self.sensors = {}
        self.actuators = {}
        self.command_timestamps = []
        
        # Inizializza dispositivi simulati
        self._init_simulated_devices()
        
        # Genera un po' di storico iniziale
        self._generate_initial_history()
    
    def _init_simulated_devices(self):
        """Crea dispositivi simulati per test"""
        
        # Sensori simulati
        simulated_sensors = [
            ("temp_sensor_01", "Temperature Line 1", SensorType.TEMPERATURE, "production_line_1"),
            ("temp_sensor_02", "Temperature Line 2", SensorType.TEMPERATURE, "production_line_2"),
            ("humidity_sensor_01", "Humidity Room A", SensorType.HUMIDITY, "room_a"),
            ("pressure_sensor_01", "Pressure Tank 1", SensorType.PRESSURE, "tank_1"),
            ("pressure_sensor_02", "Pressure Tank 2", SensorType.PRESSURE, "tank_2"),
            ("flow_sensor_01", "Flow Meter Main", SensorType.FLOW, "main_pipe"),
            ("level_sensor_01", "Level Tank 1", SensorType.LEVEL, "tank_1"),
            ("vibration_sensor_01", "Vibration Motor 1", SensorType.VIBRATION, "motor_1"),
            ("current_sensor_01", "Current Motor 1", SensorType.CURRENT, "motor_1"),
            ("voltage_sensor_01", "Voltage Main", SensorType.VOLTAGE, "main_panel"),
        ]
        
        for sensor_id, name, sensor_type, location in simulated_sensors:
            device = Device(
                device_id=sensor_id,
                name=name,
                type="sensor",
                protocol="simulated",
                address=f"sim/{sensor_id}",
                status="online",
                last_seen=datetime.now(),
                metadata={
                    'sensor_type': sensor_type.value,
                    'location': location,
                    'unit': self.simulator.get_unit(sensor_type)
                }
            )
            self.devices[sensor_id] = device
            self.sensors[sensor_id] = device
        
        # Attuatori simulati
        simulated_actuators = [
            ("valve_01", "Main Valve", ActuatorType.VALVE),
            ("valve_02", "Secondary Valve", ActuatorType.VALVE),
            ("pump_01", "Main Pump", ActuatorType.PUMP),
            ("motor_01", "Conveyor Motor", ActuatorType.MOTOR),
            ("motor_02", "Mixer Motor", ActuatorType.MOTOR),
            ("relay_01", "Light Relay", ActuatorType.RELAY),
        ]
        
        for actuator_id, name, actuator_type in simulated_actuators:
            device = Device(
                device_id=actuator_id,
                name=name,
                type="actuator",
                protocol="simulated",
                address=f"sim/{actuator_id}",
                status="online",
                last_seen=datetime.now(),
                metadata={
                    'actuator_type': actuator_type.value,
                    'state': 'off'
                }
            )
            self.devices[actuator_id] = device
            self.actuators[actuator_id] = device
        
        # PLC simulato
        plc_device = Device(
            device_id="plc_01",
            name="Simulated PLC",
            type="plc",
            protocol="modbus",
            address="192.168.1.100:502",
            status="online",
            last_seen=datetime.now(),
            metadata={
                'registers': {str(i): random.randint(0, 65535) for i in range(100)}
            }
        )
        self.devices["plc_01"] = plc_device
        
        logger.info(f"Inizializzati {len(self.sensors)} sensori e {len(self.actuators)} attuatori simulati")
    
    def _generate_initial_history(self):
        """Genera storico iniziale per test"""
        now = datetime.now()
        
        for sensor_id, sensor in self.sensors.items():
            sensor_type = SensorType(sensor.metadata['sensor_type'])
            
            # Genera 100 punti nelle ultime 24 ore
            for i in range(100):
                timestamp = now - timedelta(hours=24) + timedelta(hours=i*0.24)
                
                reading = SensorReading(
                    sensor_id=sensor_id,
                    timestamp=timestamp,
                    value=self.simulator.generate_value(sensor_type),
                    unit=sensor.metadata['unit'],
                    quality=random.randint(95, 100)
                )
                
                self.database.store_reading(reading)
    
    def read_sensor(self, sensor_id: str) -> Dict[str, Any]:
        """Legge valore simulato da sensore"""
        if sensor_id not in self.sensors:
            raise ValueError(f"Sensore {sensor_id} non trovato")
        
        sensor = self.sensors[sensor_id]
        sensor_type = SensorType(sensor.metadata['sensor_type'])
        
        # Genera valore simulato
        value = self.simulator.generate_value(sensor_type)
        quality = random.randint(95, 100)
        
        # Simula occasionali errori di lettura (2%)
        if random.random() < 0.02:
            sensor.status = "error"
            raise Exception(f"Errore lettura sensore {sensor_id} (simulato)")
        
        sensor.status = "online"
        sensor.last_seen = datetime.now()
        
        reading = SensorReading(
            sensor_id=sensor_id,
            timestamp=datetime.now(),
            value=value,
            unit=sensor.metadata['unit'],
            quality=quality
        )
        
        # Salva in "database"
        self.database.store_reading(reading)
        
        # Simula allarme se valore anomalo
        if sensor_type == SensorType.TEMPERATURE and value > 50:
            alarm_id = f"alarm_{sensor_id}_{datetime.now().timestamp()}"
            alarm = Alarm(
                alarm_id=alarm_id,
                sensor_id=sensor_id,
                timestamp=datetime.now(),
                priority=AlarmPriority.HIGH,
                message=f"Temperatura elevata: {value}°C",
                threshold_value=50.0,
                actual_value=value,
                acknowledged=False
            )
            self.database.active_alarms[alarm_id] = alarm
            logger.warning(f"ALLARME GENERATO: {alarm.message}")
        
        return {
            'sensor_id': sensor_id,
            'value': value,
            'unit': reading.unit,
            'timestamp': reading.timestamp.isoformat(),
            'quality': quality,
            'source': 'simulated'
        }
    
    def execute_command(
        self, 
        actuator_id: str, 
        command: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Esegue comando simulato su attuatore"""
        
        # Check rate limiting (semplificato)
        now = datetime.now()
        self.command_timestamps = [
            ts for ts in self.command_timestamps 
            if (now - ts).seconds < 60
        ]
        
        if len(self.command_timestamps) >= 10:
            raise ValueError("Rate limit comandi superato (max 10/min)")
        
        self.command_timestamps.append(now)
        
        if actuator_id not in self.actuators:
            raise ValueError(f"Attuatore {actuator_id} non trovato")
        
        actuator = self.actuators[actuator_id]
        parameters = parameters or {}
        
        # Simula esecuzione comando
        actuator.metadata['state'] = command
        actuator.metadata['last_command'] = {
            'command': command,
            'parameters': parameters,
            'timestamp': now.isoformat()
        }
        
        # Log comando
        self.database.command_log.append({
            'actuator_id': actuator_id,
            'command': command,
            'parameters': parameters,
            'timestamp': now
        })
        
        logger.info(f"COMANDO ESEGUITO: {actuator_id} -> {command} {parameters}")
        
        return {
            'actuator_id': actuator_id,
            'command': command,
            'parameters': parameters,
            'status': 'executed',
            'timestamp': now.isoformat(),
            'simulated': True
        }
    
    def get_device_topology(self) -> Dict[str, Any]:
        """Ottiene topologia sistema simulato"""
        return {
            'mode': 'SIMULATED',
            'total_devices': len(self.devices),
            'sensors': len(self.sensors),
            'actuators': len(self.actuators),
            'protocols': {
                'simulated': len([d for d in self.devices.values() if d.protocol == 'simulated']),
                'modbus': len([d for d in self.devices.values() if d.protocol == 'modbus'])
            },
            'status': {
                'online': len([d for d in self.devices.values() if d.status == 'online']),
                'offline': len([d for d in self.devices.values() if d.status == 'offline']),
                'error': len([d for d in self.devices.values() if d.status == 'error'])
            },
            'devices': [
                {
                    'id': d.device_id,
                    'name': d.name,
                    'type': d.type,
                    'protocol': d.protocol,
                    'status': d.status,
                    'last_seen': d.last_seen.isoformat()
                }
                for d in self.devices.values()
            ]
        }

# ==================== SINGLETON ====================

# Istanza singleton del manager simulato
iot_manager = IoTManagerSimulated()

# ==================== FUNZIONI MCP ====================

def read_sensor(sensor_id: str) -> Dict[str, Any]:
    """
    Legge valore corrente da un sensore IoT (simulato).
    
    Args:
        sensor_id: ID univoco del sensore (es: temp_sensor_01, pressure_sensor_01)
        
    Returns:
        Dizionario con valore, unità, timestamp e qualità del segnale
    """
    return iot_manager.read_sensor(sensor_id)


def read_multiple_sensors(sensor_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Legge valori da multipli sensori contemporaneamente.
    
    Args:
        sensor_ids: Lista di ID sensori da leggere
        
    Returns:
        Lista di letture sensori
    """
    results = []
    for sensor_id in sensor_ids:
        try:
            results.append(iot_manager.read_sensor(sensor_id))
        except Exception as e:
            results.append({
                'sensor_id': sensor_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    return results


def get_sensor_history(
    sensor_id: str,
    hours: int = 24,
    aggregation: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Ottiene storico letture di un sensore.
    
    Args:
        sensor_id: ID del sensore
        hours: Numero di ore di storico (default 24)
        aggregation: Tipo aggregazione (mean, max, min)
        
    Returns:
        Lista di punti dati storici
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    return iot_manager.database.get_history(
        sensor_id, 
        start_time, 
        end_time,
        aggregation
    )


def execute_actuator_command(
    actuator_id: str,
    command: str,
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Esegue comando su un attuatore.
    
    Args:
        actuator_id: ID dell'attuatore (es: valve_01, motor_01)
        command: Nome comando da eseguire (on, off, set_speed, open, close)
        parameters: Parametri opzionali del comando
        
    Returns:
        Stato esecuzione comando
    """
    return iot_manager.execute_command(actuator_id, command, parameters)


def get_device_topology() -> Dict[str, Any]:
    """
    Ottiene topologia completa del sistema IoT.
    
    Returns:
        Struttura gerarchica dispositivi, sensori e attuatori
    """
    return iot_manager.get_device_topology()


def list_devices(device_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Lista tutti i dispositivi IoT registrati.
    
    Args:
        device_type: Filtra per tipo (sensor, actuator, plc, gateway)
        
    Returns:
        Lista dispositivi con dettagli
    """
    devices = iot_manager.devices.values()
    
    if device_type:
        devices = [d for d in devices if d.type == device_type]
    
    return [
        {
            'device_id': d.device_id,
            'name': d.name,
            'type': d.type,
            'protocol': d.protocol,
            'address': d.address,
            'status': d.status,
            'last_seen': d.last_seen.isoformat(),
            'metadata': d.metadata
        }
        for d in devices
    ]


def get_active_alarms(priority: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Ottiene lista allarmi attivi nel sistema.
    
    Args:
        priority: Filtra per priorità (LOW, MEDIUM, HIGH, CRITICAL)
        
    Returns:
        Lista allarmi attivi ordinati per priorità
    """
    alarms = list(iot_manager.database.active_alarms.values())
    
    if priority:
        priority_enum = AlarmPriority[priority.upper()]
        alarms = [a for a in alarms if a.priority == priority_enum]
    
    # Ordina per priorità e timestamp
    alarms.sort(key=lambda x: (-x.priority.value, x.timestamp))
    
    return [
        {
            'alarm_id': a.alarm_id,
            'sensor_id': a.sensor_id,
            'timestamp': a.timestamp.isoformat(),
            'priority': a.priority.name,
            'message': a.message,
            'threshold_value': a.threshold_value,
            'actual_value': a.actual_value,
            'acknowledged': a.acknowledged
        }
        for a in alarms
    ]


def acknowledge_alarm(alarm_id: str) -> bool:
    """
    Conferma presa visione di un allarme.
    
    Args:
        alarm_id: ID dell'allarme da confermare
        
    Returns:
        True se confermato con successo
    """
    if alarm_id in iot_manager.database.active_alarms:
        iot_manager.database.active_alarms[alarm_id].acknowledged = True
        logger.info(f"Allarme {alarm_id} confermato")
        return True
    return False


def read_modbus_registers(
    device_id: str,
    address: int,
    count: int = 1
) -> List[int]:
    """
    Legge registri Modbus da dispositivo industriale (simulato).
    
    Args:
        device_id: ID dispositivo Modbus
        address: Indirizzo registro iniziale (0-99)
        count: Numero di registri da leggere
        
    Returns:
        Lista valori registri
    """
    if device_id not in iot_manager.devices:
        raise ValueError(f"Device {device_id} non trovato")
    
    device = iot_manager.devices[device_id]
    if device.type != "plc":
        raise ValueError(f"Device {device_id} non è un PLC")
    
    # Simula lettura registri
    registers = []
    for i in range(count):
        reg_addr = str(address + i)
        if reg_addr in device.metadata['registers']:
            registers.append(device.metadata['registers'][reg_addr])
        else:
            # Genera valore random per registro non esistente
            value = random.randint(0, 65535)
            device.metadata['registers'][reg_addr] = value
            registers.append(value)
    
    logger.info(f"Letti registri {address}-{address+count-1} da {device_id}: {registers}")
    return registers


def write_modbus_register(
    device_id: str,
    address: int,
    value: int
) -> Dict[str, Any]:
    """
    Scrive valore in registro Modbus (simulato).
    
    Args:
        device_id: ID dispositivo Modbus  
        address: Indirizzo registro
        value: Valore da scrivere (0-65535)
        
    Returns:
        Stato operazione
    """
    if device_id not in iot_manager.devices:
        raise ValueError(f"Device {device_id} non trovato")
    
    device = iot_manager.devices[device_id]
    if device.type != "plc":
        raise ValueError(f"Device {device_id} non è un PLC")
    
    if not 0 <= value <= 65535:
        raise ValueError(f"Valore {value} fuori range (0-65535)")
    
    # Simula scrittura registro
    device.metadata['registers'][str(address)] = value
    
    logger.info(f"Scritto registro {address} = {value} su {device_id}")
    
    return {
        'device_id': device_id,
        'address': address,
        'value': value,
        'status': 'written',
        'timestamp': datetime.now().isoformat()
    }


def get_system_status() -> Dict[str, Any]:
    """
    Ottiene stato generale del sistema IoT.
    
    Returns:
        Statistiche e stato connessioni
    """
    return {
        'timestamp': datetime.now().isoformat(),
        'mode': 'SIMULATION',
        'devices': {
            'total': len(iot_manager.devices),
            'online': len([d for d in iot_manager.devices.values() if d.status == 'online']),
            'errors': len([d for d in iot_manager.devices.values() if d.status == 'error'])
        },
        'sensors': {
            'total': len(iot_manager.sensors),
            'types': list(set(s.metadata.get('sensor_type') for s in iot_manager.sensors.values()))
        },
        'actuators': {
            'total': len(iot_manager.actuators),
            'types': list(set(a.metadata.get('actuator_type') for a in iot_manager.actuators.values()))
        },
        'alarms': {
            'active': len(iot_manager.database.active_alarms),
            'unacknowledged': len([
                a for a in iot_manager.database.active_alarms.values() 
                if not a.acknowledged
            ])
        },
        'history': {
            'total_readings': sum(len(h) for h in iot_manager.database.sensor_history.values()),
            'sensors_with_data': len(iot_manager.database.sensor_history)
        },
        'commands': {
            'executed': len(iot_manager.database.command_log),
            'rate_limit': f"{len(iot_manager.command_timestamps)}/10 per minute"
        }
    }


# ==================== ESPOSIZIONE MCP ====================

# Lista funzioni da esporre come tools MCP
tools = [
    read_sensor,
    read_multiple_sensors,
    get_sensor_history,
    execute_actuator_command,
    get_device_topology,
    list_devices,
    get_active_alarms,
    acknowledge_alarm,
    read_modbus_registers,
    write_modbus_register,
    get_system_status
]

# Crea applicazione FastAPI con tools MCP
app = expose_tools(
    tools=tools,
    title="IoT/Edge MCP Server (Simulato)",
    description="Server MCP per test con simulazione completa di dispositivi IoT",
    version="1.0.0-sim"
)

# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("IoT/Edge MCP Server - MODALITÀ SIMULAZIONE")
    print("="*60)
    print("\nDispositivi simulati disponibili:")
    print(f"- {len(iot_manager.sensors)} Sensori")
    print(f"- {len(iot_manager.actuators)} Attuatori")
    print(f"- 1 PLC Modbus")
    
    print("\nEsempi di sensor_id disponibili:")
    for sensor_id in list(iot_manager.sensors.keys())[:5]:
        print(f"  - {sensor_id}")
    
    print("\nEsempi di actuator_id disponibili:")
    for actuator_id in list(iot_manager.actuators.keys())[:3]:
        print(f"  - {actuator_id}")
    
    print("\nTools MCP disponibili:")
    for tool in tools:
        print(f"  - {tool.__name__}")
    
    print("\nServer in ascolto su http://localhost:8000")
    print("Documentazione API: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    # Avvia server
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"

    )
