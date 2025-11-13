"""
IoT/Edge MCP Server - Production Ready
Interfaccia unificata per sistemi IoT, SCADA, PLC e automazione industriale
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import struct

# Dipendenze esterne (installare con pip)
import paho.mqtt.client as mqtt
from pymodbus.client import ModbusTcpClient, ModbusSerialClient
from pymodbus.exceptions import ModbusException
import redis
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import yaml

# Import del framework expose_tools fornito
from polymcp_toolkit import expose_tools

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== CONFIGURAZIONE ====================

class Config:
    """Configurazione centralizzata del sistema IoT"""
    
    def __init__(self, config_path: str = "iot_config.yaml"):
        """Carica configurazione da file YAML o variabili d'ambiente"""
        self.config = self._load_config(config_path)
        
        # MQTT
        self.mqtt_broker = self.config.get('mqtt', {}).get('broker', os.getenv('MQTT_BROKER', 'localhost'))
        self.mqtt_port = int(self.config.get('mqtt', {}).get('port', os.getenv('MQTT_PORT', 1883)))
        self.mqtt_username = self.config.get('mqtt', {}).get('username', os.getenv('MQTT_USERNAME'))
        self.mqtt_password = self.config.get('mqtt', {}).get('password', os.getenv('MQTT_PASSWORD'))
        
        # Modbus
        self.modbus_devices = self.config.get('modbus', {}).get('devices', [])
        
        # InfluxDB
        self.influx_url = self.config.get('influxdb', {}).get('url', os.getenv('INFLUX_URL', 'http://localhost:8086'))
        self.influx_token = self.config.get('influxdb', {}).get('token', os.getenv('INFLUX_TOKEN'))
        self.influx_org = self.config.get('influxdb', {}).get('org', os.getenv('INFLUX_ORG', 'iot'))
        self.influx_bucket = self.config.get('influxdb', {}).get('bucket', os.getenv('INFLUX_BUCKET', 'sensors'))
        
        # Redis
        self.redis_host = self.config.get('redis', {}).get('host', os.getenv('REDIS_HOST', 'localhost'))
        self.redis_port = int(self.config.get('redis', {}).get('port', os.getenv('REDIS_PORT', 6379)))
        self.redis_db = int(self.config.get('redis', {}).get('db', os.getenv('REDIS_DB', 0)))
        
        # Limiti di sicurezza
        self.max_command_rate = int(self.config.get('limits', {}).get('max_command_rate', 10))  # comandi/minuto
        self.max_query_size = int(self.config.get('limits', {}).get('max_query_size', 10000))  # punti dati
        
    def _load_config(self, config_path: str) -> dict:
        """Carica configurazione da file YAML se esiste"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}

config = Config()

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
    POSITION = "position"
    DIGITAL = "digital"
    ANALOG = "analog"

class ActuatorType(Enum):
    """Tipi di attuatori supportati"""
    VALVE = "valve"
    PUMP = "pump"
    MOTOR = "motor"
    RELAY = "relay"
    PWM = "pwm"
    SERVO = "servo"
    LIGHT = "light"

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
    metadata: Dict[str, Any] = None

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

# ==================== CONNETTORI PROTOCOLLI ====================

class MQTTConnector:
    """Gestore connessioni MQTT per sensori IoT"""
    
    def __init__(self):
        self.client = mqtt.Client()
        self.connected = False
        self.subscriptions = {}
        self.last_values = {}
        
        # Setup callbacks
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        # Autenticazione se configurata
        if config.mqtt_username and config.mqtt_password:
            self.client.username_pw_set(config.mqtt_username, config.mqtt_password)
        
        self._connect()
    
    def _connect(self):
        """Connette al broker MQTT"""
        try:
            self.client.connect(config.mqtt_broker, config.mqtt_port, 60)
            self.client.loop_start()
            logger.info(f"Connesso a MQTT broker {config.mqtt_broker}:{config.mqtt_port}")
        except Exception as e:
            logger.error(f"Errore connessione MQTT: {e}")
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback connessione MQTT"""
        if rc == 0:
            self.connected = True
            logger.info("MQTT connesso con successo")
            # Ripristina sottoscrizioni
            for topic in self.subscriptions:
                client.subscribe(topic)
        else:
            logger.error(f"MQTT connessione fallita, codice: {rc}")
    
    def _on_message(self, client, userdata, msg):
        """Callback ricezione messaggi MQTT"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            self.last_values[topic] = {
                'timestamp': datetime.now(),
                'value': payload
            }
            logger.debug(f"MQTT ricevuto: {topic} = {payload}")
        except Exception as e:
            logger.error(f"Errore parsing messaggio MQTT: {e}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback disconnessione MQTT"""
        self.connected = False
        logger.warning(f"MQTT disconnesso, codice: {rc}")
        if rc != 0:
            # Riconnessione automatica
            self._connect()
    
    def subscribe_sensor(self, sensor_id: str, topic: str):
        """Sottoscrive a topic per sensore"""
        self.subscriptions[topic] = sensor_id
        if self.connected:
            self.client.subscribe(topic)
            logger.info(f"Sottoscritto a {topic} per sensore {sensor_id}")
    
    def publish_command(self, topic: str, payload: Dict[str, Any]):
        """Pubblica comando a dispositivo"""
        if self.connected:
            self.client.publish(topic, json.dumps(payload))
            logger.info(f"Comando pubblicato su {topic}: {payload}")
        else:
            raise Exception("MQTT non connesso")
    
    def get_sensor_value(self, topic: str) -> Optional[Any]:
        """Ottiene ultimo valore da topic"""
        if topic in self.last_values:
            data = self.last_values[topic]
            # Controlla che il dato non sia troppo vecchio (max 60 secondi)
            if (datetime.now() - data['timestamp']).seconds < 60:
                return data['value']
        return None

class ModbusConnector:
    """Gestore connessioni Modbus per PLC e dispositivi industriali"""
    
    def __init__(self):
        self.clients = {}
        self._init_clients()
    
    def _init_clients(self):
        """Inizializza client Modbus da configurazione"""
        for device in config.modbus_devices:
            try:
                if device['type'] == 'tcp':
                    client = ModbusTcpClient(
                        host=device['host'],
                        port=device.get('port', 502)
                    )
                elif device['type'] == 'rtu':
                    client = ModbusSerialClient(
                        port=device['port'],
                        baudrate=device.get('baudrate', 9600),
                        stopbits=device.get('stopbits', 1),
                        bytesize=device.get('bytesize', 8),
                        parity=device.get('parity', 'N')
                    )
                else:
                    continue
                
                if client.connect():
                    self.clients[device['device_id']] = {
                        'client': client,
                        'unit': device.get('unit', 1),
                        'type': device['type']
                    }
                    logger.info(f"Connesso a Modbus device {device['device_id']}")
                else:
                    logger.error(f"Impossibile connettersi a {device['device_id']}")
                    
            except Exception as e:
                logger.error(f"Errore init Modbus {device['device_id']}: {e}")
    
    def read_registers(self, device_id: str, address: int, count: int = 1) -> List[int]:
        """Legge registri holding da dispositivo Modbus"""
        if device_id not in self.clients:
            raise ValueError(f"Device {device_id} non trovato")
        
        client_info = self.clients[device_id]
        client = client_info['client']
        unit = client_info['unit']
        
        try:
            result = client.read_holding_registers(address, count, unit=unit)
            if not result.isError():
                return result.registers
            else:
                raise ModbusException(f"Errore lettura: {result}")
        except Exception as e:
            logger.error(f"Errore lettura Modbus {device_id}: {e}")
            raise
    
    def write_register(self, device_id: str, address: int, value: int):
        """Scrive singolo registro"""
        if device_id not in self.clients:
            raise ValueError(f"Device {device_id} non trovato")
        
        client_info = self.clients[device_id]
        client = client_info['client']
        unit = client_info['unit']
        
        try:
            result = client.write_register(address, value, unit=unit)
            if result.isError():
                raise ModbusException(f"Errore scrittura: {result}")
            logger.info(f"Scritto {value} a {device_id}:{address}")
        except Exception as e:
            logger.error(f"Errore scrittura Modbus {device_id}: {e}")
            raise
    
    def read_coils(self, device_id: str, address: int, count: int = 1) -> List[bool]:
        """Legge coils (bit) da dispositivo"""
        if device_id not in self.clients:
            raise ValueError(f"Device {device_id} non trovato")
        
        client_info = self.clients[device_id]
        client = client_info['client']
        unit = client_info['unit']
        
        try:
            result = client.read_coils(address, count, unit=unit)
            if not result.isError():
                return result.bits[:count]
            else:
                raise ModbusException(f"Errore lettura coils: {result}")
        except Exception as e:
            logger.error(f"Errore lettura coils {device_id}: {e}")
            raise
    
    def write_coil(self, device_id: str, address: int, value: bool):
        """Scrive singolo coil"""
        if device_id not in self.clients:
            raise ValueError(f"Device {device_id} non trovato")
        
        client_info = self.clients[device_id]
        client = client_info['client']
        unit = client_info['unit']
        
        try:
            result = client.write_coil(address, value, unit=unit)
            if result.isError():
                raise ModbusException(f"Errore scrittura coil: {result}")
            logger.info(f"Scritto coil {value} a {device_id}:{address}")
        except Exception as e:
            logger.error(f"Errore scrittura coil {device_id}: {e}")
            raise

# ==================== DATABASE & CACHE ====================

class DataStore:
    """Gestione storage dati time-series e cache"""
    
    def __init__(self):
        # InfluxDB per time-series
        self.influx_client = None
        self.write_api = None
        self.query_api = None
        self._init_influx()
        
        # Redis per cache e stato
        self.redis_client = None
        self._init_redis()
        
        # Cache allarmi in memoria
        self.active_alarms = {}
    
    def _init_influx(self):
        """Inizializza client InfluxDB"""
        try:
            if config.influx_token:
                self.influx_client = InfluxDBClient(
                    url=config.influx_url,
                    token=config.influx_token,
                    org=config.influx_org
                )
                self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
                self.query_api = self.influx_client.query_api()
                logger.info("Connesso a InfluxDB")
        except Exception as e:
            logger.error(f"Errore connessione InfluxDB: {e}")
    
    def _init_redis(self):
        """Inizializza client Redis"""
        try:
            self.redis_client = redis.Redis(
                host=config.redis_host,
                port=config.redis_port,
                db=config.redis_db,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Connesso a Redis")
        except Exception as e:
            logger.error(f"Errore connessione Redis: {e}")
            self.redis_client = None
    
    def store_sensor_reading(self, reading: SensorReading):
        """Salva lettura sensore in InfluxDB"""
        if self.write_api:
            try:
                point = Point("sensor_reading") \
                    .tag("sensor_id", reading.sensor_id) \
                    .tag("unit", reading.unit) \
                    .field("value", float(reading.value)) \
                    .field("quality", reading.quality) \
                    .time(reading.timestamp)
                
                if reading.metadata:
                    for key, value in reading.metadata.items():
                        point.tag(key, str(value))
                
                self.write_api.write(bucket=config.influx_bucket, record=point)
                
                # Aggiorna cache Redis con ultimo valore
                if self.redis_client:
                    cache_key = f"sensor:{reading.sensor_id}:last"
                    self.redis_client.hset(cache_key, mapping={
                        'value': reading.value,
                        'unit': reading.unit,
                        'timestamp': reading.timestamp.isoformat(),
                        'quality': reading.quality
                    })
                    self.redis_client.expire(cache_key, 3600)  # TTL 1 ora
                    
            except Exception as e:
                logger.error(f"Errore storage reading: {e}")
    
    def query_sensor_history(
        self, 
        sensor_id: str, 
        start_time: datetime, 
        end_time: datetime,
        aggregation: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query storico sensore da InfluxDB"""
        if not self.query_api:
            return []
        
        try:
            # Costruisci query Flux
            if aggregation:
                query = f'''
                from(bucket: "{config.influx_bucket}")
                    |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
                    |> filter(fn: (r) => r["sensor_id"] == "{sensor_id}")
                    |> filter(fn: (r) => r["_field"] == "value")
                    |> aggregateWindow(every: 1m, fn: {aggregation}, createEmpty: false)
                    |> yield(name: "{aggregation}")
                '''
            else:
                query = f'''
                from(bucket: "{config.influx_bucket}")
                    |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
                    |> filter(fn: (r) => r["sensor_id"] == "{sensor_id}")
                    |> filter(fn: (r) => r["_field"] == "value")
                    |> yield(name: "raw")
                '''
            
            result = self.query_api.query(query=query)
            
            # Converti risultati
            data = []
            for table in result:
                for record in table.records:
                    data.append({
                        'timestamp': record.get_time().isoformat(),
                        'value': record.get_value(),
                        'sensor_id': sensor_id
                    })
            
            # Limita dimensione risultato
            if len(data) > config.max_query_size:
                data = data[:config.max_query_size]
            
            return data
            
        except Exception as e:
            logger.error(f"Errore query history: {e}")
            return []
    
    def get_cached_value(self, key: str) -> Optional[Any]:
        """Ottiene valore da cache Redis"""
        if self.redis_client:
            try:
                return self.redis_client.get(key)
            except Exception as e:
                logger.error(f"Errore lettura cache: {e}")
        return None
    
    def set_cached_value(self, key: str, value: Any, ttl: int = 300):
        """Imposta valore in cache Redis"""
        if self.redis_client:
            try:
                if isinstance(value, dict):
                    value = json.dumps(value)
                self.redis_client.set(key, value, ex=ttl)
            except Exception as e:
                logger.error(f"Errore scrittura cache: {e}")

# ==================== GESTIONE ALLARMI ====================

class AlarmManager:
    """Gestione allarmi e notifiche"""
    
    def __init__(self, data_store: DataStore):
        self.data_store = data_store
        self.alarm_rules = self._load_alarm_rules()
        self.rate_limiter = {}
    
    def _load_alarm_rules(self) -> Dict[str, Dict]:
        """Carica regole allarmi da configurazione"""
        # In produzione caricheresti da DB o file config
        return {
            'high_temperature': {
                'sensor_type': SensorType.TEMPERATURE,
                'condition': 'greater_than',
                'threshold': 80.0,
                'priority': AlarmPriority.HIGH,
                'message': 'Temperatura elevata rilevata'
            },
            'low_pressure': {
                'sensor_type': SensorType.PRESSURE,
                'condition': 'less_than',
                'threshold': 2.0,
                'priority': AlarmPriority.CRITICAL,
                'message': 'Pressione bassa critica'
            }
        }
    
    def check_alarm_condition(
        self, 
        sensor_id: str, 
        value: float, 
        sensor_type: SensorType
    ) -> Optional[Alarm]:
        """Controlla se valore genera allarme"""
        for rule_id, rule in self.alarm_rules.items():
            if rule['sensor_type'] != sensor_type:
                continue
            
            triggered = False
            threshold = rule['threshold']
            
            if rule['condition'] == 'greater_than' and value > threshold:
                triggered = True
            elif rule['condition'] == 'less_than' and value < threshold:
                triggered = True
            elif rule['condition'] == 'equals' and value == threshold:
                triggered = True
            
            if triggered:
                alarm_id = f"{sensor_id}_{rule_id}_{datetime.now().timestamp()}"
                alarm = Alarm(
                    alarm_id=alarm_id,
                    sensor_id=sensor_id,
                    timestamp=datetime.now(),
                    priority=rule['priority'],
                    message=rule['message'],
                    threshold_value=threshold,
                    actual_value=value,
                    acknowledged=False
                )
                
                # Salva in cache
                if self.data_store.redis_client:
                    alarm_key = f"alarm:{alarm_id}"
                    self.data_store.redis_client.hset(
                        alarm_key, 
                        mapping=asdict(alarm)
                    )
                    self.data_store.redis_client.expire(alarm_key, 86400)  # 24h
                
                # Aggiungi a lista allarmi attivi
                self.data_store.active_alarms[alarm_id] = alarm
                
                logger.warning(f"Allarme generato: {alarm.message} - Valore: {value}")
                return alarm
        
        return None
    
    def acknowledge_alarm(self, alarm_id: str) -> bool:
        """Conferma presa visione allarme"""
        if alarm_id in self.data_store.active_alarms:
            self.data_store.active_alarms[alarm_id].acknowledged = True
            
            # Aggiorna in Redis
            if self.data_store.redis_client:
                alarm_key = f"alarm:{alarm_id}"
                self.data_store.redis_client.hset(alarm_key, 'acknowledged', 'true')
            
            logger.info(f"Allarme {alarm_id} confermato")
            return True
        return False
    
    def get_active_alarms(self, priority: Optional[AlarmPriority] = None) -> List[Alarm]:
        """Ottiene lista allarmi attivi"""
        alarms = list(self.data_store.active_alarms.values())
        
        if priority:
            alarms = [a for a in alarms if a.priority == priority]
        
        # Ordina per priorità e timestamp
        alarms.sort(key=lambda x: (-x.priority.value, x.timestamp))
        
        return alarms

# ==================== MANAGER PRINCIPALE ====================

class IoTManager:
    """Manager principale sistema IoT"""
    
    def __init__(self):
        self.mqtt = MQTTConnector()
        self.modbus = ModbusConnector()
        self.data_store = DataStore()
        self.alarm_manager = AlarmManager(self.data_store)
        
        # Registry dispositivi
        self.devices = {}
        self.sensors = {}
        self.actuators = {}
        
        # Rate limiting per comandi
        self.command_timestamps = []
        
        # Inizializza dispositivi da config
        self._init_devices()
    
    def _init_devices(self):
        """Inizializza dispositivi da configurazione"""
        # Carica dispositivi MQTT
        for device_config in config.config.get('devices', {}).get('mqtt', []):
            device = Device(
                device_id=device_config['id'],
                name=device_config['name'],
                type=device_config['type'],
                protocol='mqtt',
                address=device_config['topic'],
                status='unknown',
                last_seen=datetime.now(),
                metadata=device_config.get('metadata', {})
            )
            self.devices[device.device_id] = device
            
            if device.type == 'sensor':
                self.sensors[device.device_id] = device
                self.mqtt.subscribe_sensor(device.device_id, device.address)
            elif device.type == 'actuator':
                self.actuators[device.device_id] = device
        
        # Carica dispositivi Modbus
        for device_config in config.modbus_devices:
            device = Device(
                device_id=device_config['device_id'],
                name=device_config.get('name', device_config['device_id']),
                type='plc',
                protocol='modbus',
                address=f"{device_config.get('host', 'serial')}:{device_config.get('port', 502)}",
                status='unknown',
                last_seen=datetime.now(),
                metadata=device_config
            )
            self.devices[device.device_id] = device
    
    def _check_rate_limit(self) -> bool:
        """Controlla rate limit comandi"""
        now = datetime.now()
        # Rimuovi timestamp vecchi (più di 1 minuto)
        self.command_timestamps = [
            ts for ts in self.command_timestamps 
            if (now - ts).seconds < 60
        ]
        
        if len(self.command_timestamps) >= config.max_command_rate:
            logger.warning("Rate limit comandi raggiunto")
            return False
        
        self.command_timestamps.append(now)
        return True
    
    def read_sensor(self, sensor_id: str) -> Dict[str, Any]:
        """Legge valore corrente da sensore"""
        if sensor_id not in self.sensors:
            raise ValueError(f"Sensore {sensor_id} non trovato")
        
        sensor = self.sensors[sensor_id]
        
        try:
            if sensor.protocol == 'mqtt':
                # Leggi da MQTT
                value = self.mqtt.get_sensor_value(sensor.address)
                if value is None:
                    # Prova cache Redis
                    cache_key = f"sensor:{sensor_id}:last"
                    if self.data_store.redis_client:
                        cached = self.data_store.redis_client.hgetall(cache_key)
                        if cached:
                            return {
                                'sensor_id': sensor_id,
                                'value': float(cached['value']),
                                'unit': cached['unit'],
                                'timestamp': cached['timestamp'],
                                'quality': int(cached['quality']),
                                'source': 'cache'
                            }
                    raise ValueError(f"Nessun dato disponibile per {sensor_id}")
                
                # Crea reading
                reading = SensorReading(
                    sensor_id=sensor_id,
                    timestamp=datetime.now(),
                    value=value.get('value', 0),
                    unit=value.get('unit', ''),
                    quality=value.get('quality', 100),
                    metadata=value.get('metadata')
                )
                
                # Salva in database
                self.data_store.store_sensor_reading(reading)
                
                # Controlla allarmi
                sensor_type = SensorType(sensor.metadata.get('sensor_type', 'analog'))
                self.alarm_manager.check_alarm_condition(
                    sensor_id, 
                    reading.value, 
                    sensor_type
                )
                
                # Aggiorna status dispositivo
                sensor.status = 'online'
                sensor.last_seen = datetime.now()
                
                return {
                    'sensor_id': sensor_id,
                    'value': reading.value,
                    'unit': reading.unit,
                    'timestamp': reading.timestamp.isoformat(),
                    'quality': reading.quality,
                    'metadata': reading.metadata,
                    'source': 'live'
                }
                
            elif sensor.protocol == 'modbus':
                # Leggi da Modbus
                device_id = sensor.metadata.get('modbus_device')
                address = sensor.metadata.get('register_address', 0)
                scale = sensor.metadata.get('scale', 1.0)
                
                registers = self.modbus.read_registers(device_id, address, 1)
                if registers:
                    value = registers[0] * scale
                    
                    reading = SensorReading(
                        sensor_id=sensor_id,
                        timestamp=datetime.now(),
                        value=value,
                        unit=sensor.metadata.get('unit', ''),
                        quality=100,
                        metadata={'modbus_address': address}
                    )
                    
                    self.data_store.store_sensor_reading(reading)
                    
                    return {
                        'sensor_id': sensor_id,
                        'value': value,
                        'unit': reading.unit,
                        'timestamp': reading.timestamp.isoformat(),
                        'quality': 100,
                        'source': 'modbus'
                    }
                    
        except Exception as e:
            logger.error(f"Errore lettura sensore {sensor_id}: {e}")
            sensor.status = 'error'
            raise
    
    def execute_command(
        self, 
        actuator_id: str, 
        command: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Esegue comando su attuatore"""
        if not self._check_rate_limit():
            raise ValueError("Rate limit comandi superato")
        
        if actuator_id not in self.actuators:
            raise ValueError(f"Attuatore {actuator_id} non trovato")
        
        actuator = self.actuators[actuator_id]
        parameters = parameters or {}
        
        try:
            if actuator.protocol == 'mqtt':
                # Comando via MQTT
                payload = {
                    'command': command,
                    'parameters': parameters,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.mqtt.publish_command(actuator.address, payload)
                
                # Log comando
                logger.info(f"Comando eseguito: {actuator_id} - {command}")
                
                return {
                    'actuator_id': actuator_id,
                    'command': command,
                    'parameters': parameters,
                    'status': 'sent',
                    'timestamp': datetime.now().isoformat()
                }
                
            elif actuator.protocol == 'modbus':
                # Comando via Modbus
                device_id = actuator.metadata.get('modbus_device')
                
                if command == 'write_coil':
                    address = parameters.get('address', 0)
                    value = parameters.get('value', False)
                    self.modbus.write_coil(device_id, address, value)
                    
                elif command == 'write_register':
                    address = parameters.get('address', 0)
                    value = parameters.get('value', 0)
                    self.modbus.write_register(device_id, address, value)
                    
                else:
                    raise ValueError(f"Comando Modbus non supportato: {command}")
                
                return {
                    'actuator_id': actuator_id,
                    'command': command,
                    'parameters': parameters,
                    'status': 'executed',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Errore esecuzione comando {actuator_id}: {e}")
            raise
    
    def get_device_topology(self) -> Dict[str, Any]:
        """Ottiene topologia completa sistema"""
        return {
            'total_devices': len(self.devices),
            'sensors': len(self.sensors),
            'actuators': len(self.actuators),
            'protocols': {
                'mqtt': len([d for d in self.devices.values() if d.protocol == 'mqtt']),
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

# Istanza singleton del manager
iot_manager = IoTManager()

# ==================== FUNZIONI MCP ====================

def read_sensor(sensor_id: str) -> Dict[str, Any]:
    """
    Legge valore corrente da un sensore IoT.
    
    Args:
        sensor_id: ID univoco del sensore
        
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
        aggregation: Tipo aggregazione (mean, max, min, sum)
        
    Returns:
        Lista di punti dati storici
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    return iot_manager.data_store.query_sensor_history(
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
        actuator_id: ID dell'attuatore
        command: Nome comando da eseguire
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
    priority_enum = None
    if priority:
        priority_enum = AlarmPriority[priority.upper()]
    
    alarms = iot_manager.alarm_manager.get_active_alarms(priority_enum)
    
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
    return iot_manager.alarm_manager.acknowledge_alarm(alarm_id)


def read_modbus_registers(
    device_id: str,
    address: int,
    count: int = 1
) -> List[int]:
    """
    Legge registri Modbus da dispositivo industriale.
    
    Args:
        device_id: ID dispositivo Modbus
        address: Indirizzo registro iniziale
        count: Numero di registri da leggere
        
    Returns:
        Lista valori registri
    """
    return iot_manager.modbus.read_registers(device_id, address, count)


def write_modbus_register(
    device_id: str,
    address: int,
    value: int
) -> Dict[str, Any]:
    """
    Scrive valore in registro Modbus.
    
    Args:
        device_id: ID dispositivo Modbus
        address: Indirizzo registro
        value: Valore da scrivere (0-65535)
        
    Returns:
        Stato operazione
    """
    iot_manager.modbus.write_register(device_id, address, value)
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
        'connections': {
            'mqtt': {
                'connected': iot_manager.mqtt.connected,
                'broker': config.mqtt_broker,
                'subscriptions': len(iot_manager.mqtt.subscriptions)
            },
            'modbus': {
                'devices': len(iot_manager.modbus.clients),
                'active': list(iot_manager.modbus.clients.keys())
            },
            'influxdb': {
                'connected': iot_manager.data_store.influx_client is not None
            },
            'redis': {
                'connected': iot_manager.data_store.redis_client is not None
            }
        },
        'devices': {
            'total': len(iot_manager.devices),
            'online': len([d for d in iot_manager.devices.values() if d.status == 'online']),
            'errors': len([d for d in iot_manager.devices.values() if d.status == 'error'])
        },
        'alarms': {
            'active': len(iot_manager.data_store.active_alarms),
            'unacknowledged': len([
                a for a in iot_manager.data_store.active_alarms.values() 
                if not a.acknowledged
            ])
        },
        'rate_limits': {
            'commands_per_minute': config.max_command_rate,
            'current_rate': len(iot_manager.command_timestamps)
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
    title="IoT/Edge MCP Server",
    description="Server MCP production-ready per infrastrutture IoT, Edge computing e automazione industriale",
    version="1.0.0"
)

# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Avvio IoT/Edge MCP Server...")
    logger.info(f"Tools disponibili: {[t.__name__ for t in tools]}")
    
    # Avvia server
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"

    )
