"""
IoT/Edge MCP Server - Production Ready with Full Security
Interfaccia unificata per sistemi IoT, SCADA, PLC e automazione industriale
Version: 2.1.0 - Security Hardened + Auth Middleware + Safe Startup
"""

from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import os
import re
import hashlib
import hmac
import secrets
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
from functools import wraps
from pathlib import Path
import ipaddress
import ssl

# Security imports
from cryptography.fernet import Fernet
import jwt
from passlib.context import CryptContext
import bleach

# External dependencies
import paho.mqtt.client as mqtt
from pymodbus.client import ModbusTcpClient, ModbusSerialClient
from pymodbus.exceptions import ModbusException

import redis
import redis.sentinel
from redis.exceptions import ConnectionError as RedisConnectionError

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from prometheus_client import Counter, Histogram, Gauge, generate_latest

# FastAPI (returned by expose_tools)
from fastapi import Request, HTTPException, Response

# MCP framework
from polymcp_toolkit import expose_tools


# ==================== LOGGING ====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            "iot_mcp_server.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10,
            mode="a",
        ),
    ],
)
logger = logging.getLogger(__name__)


# ==================== METRICS ====================

metrics_requests = Counter("iot_mcp_requests_total", "Total requests", ["method", "endpoint"])
metrics_errors = Counter("iot_mcp_errors_total", "Total errors", ["type"])
metrics_command_duration = Histogram("iot_mcp_command_duration_seconds", "Command duration")
metrics_active_devices = Gauge("iot_mcp_active_devices", "Active devices")
metrics_active_alarms = Gauge("iot_mcp_active_alarms", "Active alarms")


# ==================== SECURITY CONFIGURATION ====================

class SecurityConfig:
    """
    Security configuration and utilities.

    Notes:
    - In production (IOT_ENV=production), secrets must be provided via env vars.
    - In development, ephemeral secrets are generated with a warning.
    """

    ENV = os.getenv("IOT_ENV", "development").strip().lower()

    # JWT
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRATION_HOURS = 24

    # Headers
    API_KEY_HEADER = "X-API-Key"

    # Rate limiting / limits
    MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))
    MAX_COMMANDS_PER_MINUTE = int(os.getenv("MAX_COMMANDS_PER_MINUTE", "10"))
    MAX_QUERY_SIZE = int(os.getenv("MAX_QUERY_SIZE", "10000"))
    MAX_PAYLOAD_SIZE = int(os.getenv("MAX_PAYLOAD_SIZE", str(1024 * 1024)))  # 1MB
    MAX_QUERY_HOURS = int(os.getenv("MAX_QUERY_HOURS", "168"))  # 1 week

    # Input validation patterns
    SENSOR_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
    DEVICE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
    TOPIC_PATTERN = re.compile(r"^[a-zA-Z0-9/_-]{1,256}$")
    COMMAND_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

    # Password hashing
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    # Session configuration (reserved)
    SESSION_TIMEOUT_MINUTES = 30
    MAX_CONCURRENT_SESSIONS = 100

    # Loaded API keys map: API_KEY -> metadata
    API_KEYS: Dict[str, Dict[str, Any]] = {}

    # Secrets (fail-fast in production)
    JWT_SECRET_KEY: str
    ENCRYPTION_KEY: bytes
    AUDIT_HMAC_KEY: bytes
    MQTT_COMMAND_HMAC_KEY: bytes

    # Allowed IP ranges (CIDR)
    ALLOWED_IP_RANGES: List[ipaddress._BaseNetwork] = []

    @classmethod
    def _load_or_generate_secret(cls, env_name: str, byte_output: bool) -> Any:
        val = os.getenv(env_name)
        if cls.ENV == "production":
            if not val:
                raise RuntimeError(f"{env_name} must be set when IOT_ENV=production")
        if not val:
            # development fallback: ephemeral secret
            logger.warning(f"{env_name} is not set; using ephemeral secret (NOT suitable for production).")
            val = secrets.token_urlsafe(32)
        if byte_output:
            return val.encode() if isinstance(val, str) else val
        return val

    @classmethod
    def initialize(cls):
        # Secrets
        cls.JWT_SECRET_KEY = cls._load_or_generate_secret("JWT_SECRET_KEY", byte_output=False)
        enc_key = os.getenv("ENCRYPTION_KEY")
        if cls.ENV == "production" and not enc_key:
            raise RuntimeError("ENCRYPTION_KEY must be set when IOT_ENV=production")
        if not enc_key:
            logger.warning("ENCRYPTION_KEY is not set; generating ephemeral Fernet key (NOT suitable for production).")
            cls.ENCRYPTION_KEY = Fernet.generate_key()
        else:
            cls.ENCRYPTION_KEY = enc_key.encode() if isinstance(enc_key, str) else enc_key

        cls.AUDIT_HMAC_KEY = cls._load_or_generate_secret("AUDIT_HMAC_KEY", byte_output=True)
        cls.MQTT_COMMAND_HMAC_KEY = cls._load_or_generate_secret("MQTT_COMMAND_HMAC_KEY", byte_output=True)

        # API keys
        cls.initialize_api_keys()

        # Allowed IP ranges
        cls._load_ip_allowlist()

    @classmethod
    def initialize_api_keys(cls):
        """Load API keys from environment variable API_KEYS as JSON: {"name":"key", ...}"""
        cls.API_KEYS = {}
        api_keys_json = os.getenv("API_KEYS", "{}")
        try:
            api_keys = json.loads(api_keys_json)
            if not isinstance(api_keys, dict):
                raise ValueError("API_KEYS must be a JSON object")
            for name, key in api_keys.items():
                if not isinstance(key, str) or not key.strip():
                    continue
                cls.API_KEYS[key] = {
                    "name": str(name),
                    "created": datetime.utcnow().isoformat(),
                    "permissions": ["read", "write", "execute"],
                }
        except Exception as e:
            logger.error(f"Invalid API_KEYS JSON in environment: {e}")
            cls.API_KEYS = {}

    @classmethod
    def _load_ip_allowlist(cls):
        cls.ALLOWED_IP_RANGES = []
        ip_ranges_str = os.getenv("ALLOWED_IP_RANGES")
        # Secure-by-default:
        # - in development: allow localhost
        # - in production: require explicit allowlist (or it blocks everything except none)
        if not ip_ranges_str:
            if cls.ENV == "production":
                logger.warning("ALLOWED_IP_RANGES not set in production: defaulting to deny-all.")
                ip_ranges_str = ""  # deny all
            else:
                ip_ranges_str = "127.0.0.1/32,::1/128"

        for net in [n.strip() for n in ip_ranges_str.split(",") if n.strip()]:
            try:
                cls.ALLOWED_IP_RANGES.append(ipaddress.ip_network(net, strict=False))
            except ValueError:
                logger.warning(f"Invalid IP range in ALLOWED_IP_RANGES: {net}")

    @classmethod
    def get_cipher(cls) -> Fernet:
        return Fernet(cls.ENCRYPTION_KEY)

    @classmethod
    def encrypt_data(cls, data: str) -> str:
        cipher = cls.get_cipher()
        return cipher.encrypt(data.encode("utf-8")).decode("utf-8")

    @classmethod
    def decrypt_data(cls, encrypted_data: str) -> str:
        cipher = cls.get_cipher()
        return cipher.decrypt(encrypted_data.encode("utf-8")).decode("utf-8")

    @classmethod
    def hash_password(cls, password: str) -> str:
        return cls.pwd_context.hash(password)

    @classmethod
    def verify_password(cls, plain_password: str, hashed_password: str) -> bool:
        return cls.pwd_context.verify(plain_password, hashed_password)

    @classmethod
    def generate_api_key(cls) -> str:
        return secrets.token_urlsafe(32)

    @classmethod
    def validate_sensor_id(cls, sensor_id: str) -> str:
        if not isinstance(sensor_id, str) or not cls.SENSOR_ID_PATTERN.match(sensor_id):
            raise ValueError(f"Invalid sensor ID format: {sensor_id!r}")
        return bleach.clean(sensor_id, tags=[], strip=True)

    @classmethod
    def validate_device_id(cls, device_id: str) -> str:
        if not isinstance(device_id, str) or not cls.DEVICE_ID_PATTERN.match(device_id):
            raise ValueError(f"Invalid device ID format: {device_id!r}")
        return bleach.clean(device_id, tags=[], strip=True)

    @classmethod
    def validate_mqtt_topic(cls, topic: str) -> str:
        if not isinstance(topic, str) or not cls.TOPIC_PATTERN.match(topic):
            raise ValueError(f"Invalid MQTT topic format: {topic!r}")
        return bleach.clean(topic, tags=[], strip=True)

    @classmethod
    def validate_command(cls, command: str) -> str:
        if not isinstance(command, str) or not cls.COMMAND_PATTERN.match(command):
            raise ValueError(f"Invalid command format: {command!r}")
        return bleach.clean(command, tags=[], strip=True)

    @classmethod
    def validate_modbus_address(cls, address: int) -> int:
        if not isinstance(address, int) or not 0 <= address <= 65535:
            raise ValueError(f"Invalid Modbus address: {address!r}")
        return address

    @classmethod
    def validate_modbus_value(cls, value: int) -> int:
        if not isinstance(value, int) or not 0 <= value <= 65535:
            raise ValueError(f"Invalid Modbus value: {value!r}")
        return value

    @classmethod
    def sanitize_string(cls, value: str, max_length: int = 1000) -> str:
        if not isinstance(value, str):
            raise ValueError("Value must be a string")
        value = bleach.clean(value, tags=[], strip=True)
        if len(value) > max_length:
            raise ValueError(f"String too long: {len(value)} > {max_length}")
        return value

    @classmethod
    def sanitize_dict(cls, data: dict, max_depth: int = 10) -> dict:
        def _sanitize(obj, depth=0):
            if depth > max_depth:
                raise ValueError("Dictionary nesting too deep")
            if isinstance(obj, dict):
                out = {}
                for k, v in obj.items():
                    if not isinstance(k, str):
                        continue
                    sk = cls.sanitize_string(k, 100)
                    out[sk] = _sanitize(v, depth + 1)
                return out
            if isinstance(obj, list):
                return [_sanitize(item, depth + 1) for item in obj[:1000]]
            if isinstance(obj, str):
                return cls.sanitize_string(obj, 10000)
            if isinstance(obj, (int, float, bool)) or obj is None:
                return obj
            return cls.sanitize_string(str(obj), 10000)

        if not isinstance(data, dict):
            raise ValueError("Expected dict")
        return _sanitize(data)

    @classmethod
    def validate_ip_address(cls, ip: str) -> bool:
        if not cls.ALLOWED_IP_RANGES:
            return False  # deny-all when allowlist is empty
        try:
            addr = ipaddress.ip_address(ip)
            return any(addr in network for network in cls.ALLOWED_IP_RANGES)
        except ValueError:
            return False

    @classmethod
    def validate_hours_param(cls, hours: int) -> int:
        if not isinstance(hours, int) or hours < 1:
            raise ValueError(f"Invalid hours parameter: {hours!r}")
        return min(hours, cls.MAX_QUERY_HOURS)


SecurityConfig.initialize()


# ==================== SECURE CONFIGURATION ====================

class SecureConfig:
    """Secure configuration management"""

    def __init__(self, config_path: str = "iot_config.yaml"):
        self.config_path = self._validate_config_path(config_path)
        self.config = self._load_secure_config()
        self._decrypt_sensitive_data()
        self._validate_configuration()

    def _validate_config_path(self, config_path: str) -> Path:
        config_path_p = Path(config_path).resolve()
        allowed_dir = Path(os.getcwd()).resolve()
        if not str(config_path_p).startswith(str(allowed_dir)):
            raise ValueError(f"Config path outside allowed directory: {config_path_p}")
        return config_path_p

    def _load_secure_config(self) -> dict:
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return self._get_default_config()

        if os.name != "nt":
            stat_info = os.stat(self.config_path)
            if stat_info.st_mode & 0o077:
                logger.warning("Config file has excessive permissions (recommended 600)")

        try:
            import yaml  # local import to keep top clean

            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                if not isinstance(config, dict):
                    raise ValueError("Invalid configuration format")
                return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        return {
            "mqtt": {
                "broker": os.getenv("MQTT_BROKER", "localhost"),
                "port": int(os.getenv("MQTT_PORT", "8883")),
                "use_tls": os.getenv("MQTT_USE_TLS", "true").lower() == "true",
                "ca_cert": os.getenv("MQTT_CA_CERT", ""),
                "client_cert": os.getenv("MQTT_CLIENT_CERT", ""),
                "client_key": os.getenv("MQTT_CLIENT_KEY", ""),
                "username": os.getenv("MQTT_USERNAME"),
                "password_encrypted": os.getenv("MQTT_PASSWORD_ENCRYPTED"),
                "verify_incoming_signatures": os.getenv("MQTT_VERIFY_INCOMING", "false").lower() == "true",
            },
            "modbus": {"devices": []},
            "influxdb": {
                "url": os.getenv("INFLUX_URL", "https://localhost:8086"),
                "token_encrypted": os.getenv("INFLUX_TOKEN_ENCRYPTED"),
                "org": os.getenv("INFLUX_ORG", "iot"),
                "bucket": os.getenv("INFLUX_BUCKET", "sensors"),
                "verify_ssl": True,
            },
            "redis": {
                "use_sentinel": os.getenv("REDIS_USE_SENTINEL", "false").lower() == "true",
                "sentinels": [("localhost", 26379)],
                "master_name": "mymaster",
                "host": os.getenv("REDIS_HOST", "localhost"),
                "port": int(os.getenv("REDIS_PORT", "6379")),
                "db": int(os.getenv("REDIS_DB", "0")),
                "password_encrypted": os.getenv("REDIS_PASSWORD_ENCRYPTED"),
                "ssl": os.getenv("REDIS_SSL", "false").lower() == "true",
            },
            "security": {
                "enable_auth": True,
                "enable_encryption": True,
                "enable_audit_log": True,
                "max_failed_attempts": 5,
                "lockout_duration_minutes": 30,
                # If true, docs/openapi require auth too
                "protect_docs": (SecurityConfig.ENV == "production"),
            },
            "devices": {"mqtt": [], "modbus": []},
        }

    def _decrypt_sensitive_data(self):
        try:
            # MQTT password
            if self.config.get("mqtt", {}).get("password_encrypted"):
                self.config["mqtt"]["password"] = SecurityConfig.decrypt_data(
                    self.config["mqtt"]["password_encrypted"]
                )
            elif os.getenv("MQTT_PASSWORD"):
                self.config["mqtt"]["password"] = os.getenv("MQTT_PASSWORD")

            # Influx token
            if self.config.get("influxdb", {}).get("token_encrypted"):
                self.config["influxdb"]["token"] = SecurityConfig.decrypt_data(
                    self.config["influxdb"]["token_encrypted"]
                )
            elif os.getenv("INFLUX_TOKEN"):
                self.config["influxdb"]["token"] = os.getenv("INFLUX_TOKEN")

            # Redis password
            if self.config.get("redis", {}).get("password_encrypted"):
                self.config["redis"]["password"] = SecurityConfig.decrypt_data(
                    self.config["redis"]["password_encrypted"]
                )
            elif os.getenv("REDIS_PASSWORD"):
                self.config["redis"]["password"] = os.getenv("REDIS_PASSWORD")
        except Exception as e:
            logger.error(f"Failed to decrypt sensitive data: {e}")

    def _validate_configuration(self):
        mqtt_config = self.config.get("mqtt", {})
        broker = mqtt_config.get("broker")
        if broker:
            try:
                ipaddress.ip_address(broker)
            except ValueError:
                if not re.match(r"^[a-zA-Z0-9.-]+$", broker):
                    raise ValueError(f"Invalid MQTT broker: {broker}")

        for service in ["mqtt", "redis"]:
            port = self.config.get(service, {}).get("port")
            if port and not (1 <= int(port) <= 65535):
                raise ValueError(f"Invalid port for {service}: {port}")

    def get(self, *keys, default=None):
        value: Any = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value


# ==================== AUDIT LOGGING ====================

class AuditLogger:
    """Secure audit logging with HMAC chaining (tamper-evident)."""

    def __init__(self):
        self.audit_file = Path("audit.log")
        self.audit_lock = threading.Lock()
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        self._setup_audit_log()
        self.prev_hmac = self._load_prev_hmac()

    def _setup_audit_log(self):
        if not self.audit_file.exists():
            self.audit_file.touch()
            if os.name != "nt":
                os.chmod(self.audit_file, 0o600)

        handler = logging.handlers.RotatingFileHandler(
            self.audit_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=100,
            mode="a",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        # Avoid duplicate handlers on reload
        if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in self.logger.handlers):
            self.logger.addHandler(handler)

    def _load_prev_hmac(self) -> str:
        try:
            if not self.audit_file.exists() or self.audit_file.stat().st_size == 0:
                return "0"
            # Read last non-empty line
            with open(self.audit_file, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                offset = min(size, 4096)
                f.seek(-offset, os.SEEK_END)
                tail = f.read().decode("utf-8", errors="ignore")
            lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
            if not lines:
                return "0"
            last = lines[-1]
            # Format is: "timestamp - {json}"
            if " - " in last:
                last_json = last.split(" - ", 1)[1]
                obj = json.loads(last_json)
                return str(obj.get("hmac", "0"))
            return "0"
        except Exception:
            return "0"

    def log_event(
        self,
        event_type: str,
        user: str,
        action: str,
        target: str,
        result: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        with self.audit_lock:
            event = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "event_type": str(event_type),
                "user": str(user),
                "action": str(action),
                "target": str(target),
                "result": str(result),
                "metadata": SecurityConfig.sanitize_dict(metadata) if isinstance(metadata, dict) else (metadata or {}),
                "prev_hmac": self.prev_hmac,
            }
            # HMAC over canonical JSON
            event_str = json.dumps(event, sort_keys=True, separators=(",", ":"))
            event["hmac"] = hmac.new(SecurityConfig.AUDIT_HMAC_KEY, event_str.encode("utf-8"), hashlib.sha256).hexdigest()
            self.prev_hmac = event["hmac"]
            self.logger.info(json.dumps(event, separators=(",", ":")))


# ==================== DATA MODELS ====================

class SensorType(Enum):
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
    VALVE = "valve"
    PUMP = "pump"
    MOTOR = "motor"
    RELAY = "relay"
    PWM = "pwm"
    SERVO = "servo"
    LIGHT = "light"


class AlarmPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SensorReading:
    sensor_id: str
    timestamp: datetime
    value: float
    unit: str
    quality: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.sensor_id = SecurityConfig.validate_sensor_id(self.sensor_id)
        if not 0 <= int(self.quality) <= 100:
            raise ValueError(f"Invalid quality: {self.quality}")
        if not isinstance(self.value, (int, float)):
            raise ValueError(f"Invalid value type: {type(self.value)}")
        self.unit = SecurityConfig.sanitize_string(self.unit, 50)
        if isinstance(self.metadata, dict):
            self.metadata = SecurityConfig.sanitize_dict(self.metadata)
        else:
            self.metadata = {}


@dataclass
class Alarm:
    alarm_id: str
    sensor_id: str
    timestamp: datetime
    priority: AlarmPriority
    message: str
    threshold_value: float
    actual_value: float
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    def __post_init__(self):
        self.sensor_id = SecurityConfig.validate_sensor_id(self.sensor_id)
        self.message = SecurityConfig.sanitize_string(self.message, 500)


@dataclass
class Device:
    device_id: str
    name: str
    type: str
    protocol: str
    address: str
    status: str
    last_seen: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None

    def __post_init__(self):
        self.device_id = SecurityConfig.validate_device_id(self.device_id)
        self.name = SecurityConfig.sanitize_string(self.name, 100)
        if self.protocol not in ["mqtt", "modbus", "opcua"]:
            raise ValueError(f"Invalid protocol: {self.protocol}")
        if self.type not in ["sensor", "actuator", "plc", "gateway"]:
            raise ValueError(f"Invalid device type: {self.type}")
        if isinstance(self.metadata, dict):
            self.metadata = SecurityConfig.sanitize_dict(self.metadata)
        else:
            self.metadata = {}


# ==================== SECURE MQTT CONNECTOR ====================

class SecureMQTTConnector:
    """Secure MQTT connector with TLS, authentication and background reconnect."""

    def __init__(self, config: SecureConfig, audit_logger: AuditLogger):
        self.config = config
        self.audit = audit_logger
        self.client: mqtt.Client = None  # type: ignore
        self.connected = False

        self.subscriptions: Dict[str, str] = {}
        self.last_values: Dict[str, Dict[str, Any]] = {}
        self.value_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._connect_thread: Optional[threading.Thread] = None

        self.max_reconnect_attempts = 10
        self._init_client()
        self._start_connect_loop()

    def _init_client(self):
        client_id = f"iot_mcp_{secrets.token_hex(8)}"
        self.client = mqtt.Client(client_id=client_id, clean_session=True, protocol=mqtt.MQTTv311)

        # TLS
        if self.config.get("mqtt", "use_tls", False):
            ca_cert = self.config.get("mqtt", "ca_cert")
            client_cert = self.config.get("mqtt", "client_cert")
            client_key = self.config.get("mqtt", "client_key")

            try:
                if ca_cert and os.path.exists(ca_cert):
                    self.client.tls_set(
                        ca_certs=ca_cert,
                        certfile=client_cert if client_cert and os.path.exists(client_cert) else None,
                        keyfile=client_key if client_key and os.path.exists(client_key) else None,
                        cert_reqs=ssl.CERT_REQUIRED,
                        tls_version=ssl.PROTOCOL_TLSv1_2,
                    )
                    self.client.tls_insecure_set(False)
                    logger.info("MQTT TLS configured")
                else:
                    logger.warning("MQTT TLS enabled but CA cert path is missing/invalid")
            except Exception as e:
                logger.warning(f"Failed to setup MQTT TLS: {e}")

        # Authentication
        username = self.config.get("mqtt", "username")
        password = self.config.get("mqtt", "password")
        if username and password:
            self.client.username_pw_set(username, password)

        # Callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    def _start_connect_loop(self):
        def loop():
            broker = self.config.get("mqtt", "broker", "localhost")
            port = int(self.config.get("mqtt", "port", 1883))

            attempt = 0
            while not self._stop_event.is_set():
                if self.connected:
                    time.sleep(1)
                    continue

                attempt = min(attempt + 1, self.max_reconnect_attempts)
                try:
                    self.audit.log_event("mqtt_connection", "system", "connect", f"{broker}:{port}", "attempt")
                    self.client.connect(broker, port, keepalive=60)
                    self.client.loop_start()
                    # Wait briefly for on_connect
                    time.sleep(0.5)
                    attempt = 0
                except Exception as e:
                    logger.error(f"MQTT connection failed (attempt {attempt}): {e}")
                    time.sleep(min(2 ** attempt, 60))

        self._connect_thread = threading.Thread(target=loop, daemon=True)
        self._connect_thread.start()

    def stop(self):
        self._stop_event.set()
        try:
            if self.client:
                try:
                    self.client.loop_stop()
                except Exception:
                    pass
                try:
                    self.client.disconnect()
                except Exception:
                    pass
        except Exception:
            pass

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            logger.info("MQTT connected successfully")
            # Subscribe to existing topics
            for topic in list(self.subscriptions.keys()):
                try:
                    client.subscribe(topic, qos=1)
                except Exception as e:
                    logger.warning(f"Failed to subscribe to {topic}: {e}")
            self.audit.log_event("mqtt_connection", "system", "connect", "broker", "success")
        else:
            self.connected = False
            logger.error(f"MQTT connection failed with code: {rc}")
            self.audit.log_event("mqtt_connection", "system", "connect", "broker", f"failure_rc_{rc}")

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        logger.warning(f"MQTT disconnected with code: {rc}")
        if rc != 0:
            self.audit.log_event("mqtt_connection", "system", "disconnect", "broker", f"unexpected_rc_{rc}")
        else:
            self.audit.log_event("mqtt_connection", "system", "disconnect", "broker", "graceful")

    def _verify_signature_if_present(self, payload: dict) -> bool:
        """
        Optional verification for incoming payloads if they contain 'signature'.
        Enable via config mqtt.verify_incoming_signatures.
        """
        if not self.config.get("mqtt", "verify_incoming_signatures", False):
            return True
        if not isinstance(payload, dict):
            return False
        sig = payload.get("signature")
        if not isinstance(sig, str) or not sig:
            return False

        payload_to_sign = dict(payload)
        payload_to_sign.pop("signature", None)
        payload_str = json.dumps(payload_to_sign, sort_keys=True, separators=(",", ":"))
        expected = hmac.new(SecurityConfig.MQTT_COMMAND_HMAC_KEY, payload_str.encode("utf-8"), hashlib.sha256).hexdigest()
        return secrets.compare_digest(sig, expected)

    def _on_message(self, client, userdata, msg):
        try:
            topic = SecurityConfig.validate_mqtt_topic(msg.topic)

            if len(msg.payload) > SecurityConfig.MAX_PAYLOAD_SIZE:
                logger.warning(f"MQTT payload too large for topic {topic}")
                return

            try:
                payload_str = msg.payload.decode("utf-8", errors="strict")
                payload = json.loads(payload_str)
                if isinstance(payload, dict):
                    payload = SecurityConfig.sanitize_dict(payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                logger.error(f"Invalid MQTT payload for topic {topic}: {e}")
                metrics_errors.labels(type="mqtt_message").inc()
                return

            if isinstance(payload, dict) and "signature" in payload:
                if not self._verify_signature_if_present(payload):
                    logger.warning(f"MQTT signature verification failed for topic {topic}")
                    metrics_errors.labels(type="mqtt_signature").inc()
                    return

            with self.value_lock:
                self.last_values[topic] = {"timestamp": datetime.utcnow(), "value": payload}

        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
            metrics_errors.labels(type="mqtt_message").inc()

    def subscribe_sensor(self, sensor_id: str, topic: str):
        sensor_id = SecurityConfig.validate_sensor_id(sensor_id)
        topic = SecurityConfig.validate_mqtt_topic(topic)

        self.subscriptions[topic] = sensor_id
        if self.connected:
            try:
                self.client.subscribe(topic, qos=1)
                logger.info(f"Subscribed to {topic} for sensor {sensor_id}")
            except Exception as e:
                logger.warning(f"Subscribe failed for {topic}: {e}")

    def _sign_payload(self, payload: Dict[str, Any]) -> str:
        payload_to_sign = dict(payload)
        payload_to_sign.pop("signature", None)
        payload_str = json.dumps(payload_to_sign, sort_keys=True, separators=(",", ":"))
        return hmac.new(SecurityConfig.MQTT_COMMAND_HMAC_KEY, payload_str.encode("utf-8"), hashlib.sha256).hexdigest()

    def publish_command(self, topic: str, payload: Dict[str, Any]):
        topic = SecurityConfig.validate_mqtt_topic(topic)

        if not self.connected:
            raise RuntimeError("MQTT not connected")

        payload = SecurityConfig.sanitize_dict(payload)
        payload["timestamp"] = datetime.utcnow().isoformat() + "Z"
        payload["signature"] = self._sign_payload(payload)

        result = self.client.publish(topic, json.dumps(payload, separators=(",", ":")), qos=1)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"MQTT publish failed: {result.rc}")

        logger.info(f"Command published to {topic}")
        self.audit.log_event("mqtt_publish", "system", "publish", topic, "success")

    def get_sensor_value(self, topic: str) -> Optional[Any]:
        topic = SecurityConfig.validate_mqtt_topic(topic)
        with self.value_lock:
            data = self.last_values.get(topic)
            if not data:
                return None
            age = (datetime.utcnow() - data["timestamp"]).total_seconds()
            if age < 60:
                return data["value"]
        return None


# ==================== SECURE MODBUS CONNECTOR ====================

class SecureModbusConnector:
    """Secure Modbus connector with validation and simple rate limiting."""

    def __init__(self, config: SecureConfig, audit_logger: AuditLogger):
        self.config = config
        self.audit = audit_logger
        self.clients: Dict[str, Dict[str, Any]] = {}
        self.client_lock = threading.Lock()
        self.rate_limiter: Dict[str, List[float]] = {}
        self._init_clients()

    def stop(self):
        with self.client_lock:
            for info in self.clients.values():
                try:
                    info["client"].close()
                except Exception:
                    pass

    def _init_clients(self):
        devices = self.config.get("modbus", "devices", default=[]) or []
        for device in devices:
            try:
                device_id = SecurityConfig.validate_device_id(device["device_id"])
                dev_type = device.get("type", "").lower().strip()

                if dev_type == "tcp":
                    host = device["host"]
                    port = int(device.get("port", 502))
                    client = ModbusTcpClient(host=host, port=port, timeout=3, retries=3)
                elif dev_type == "rtu":
                    port = device["port"]
                    if not (str(port).startswith("/dev/") or str(port).upper().startswith("COM")):
                        logger.warning(f"Suspicious serial port for {device_id}: {port}")
                        continue
                    client = ModbusSerialClient(
                        port=port,
                        baudrate=int(device.get("baudrate", 9600)),
                        stopbits=int(device.get("stopbits", 1)),
                        bytesize=int(device.get("bytesize", 8)),
                        parity=str(device.get("parity", "N")),
                        timeout=3,
                    )
                else:
                    logger.warning(f"Unknown Modbus device type for {device_id}: {dev_type}")
                    continue

                ok = False
                try:
                    ok = bool(client.connect())
                except Exception:
                    ok = False

                if ok:
                    self.clients[device_id] = {
                        "client": client,
                        "unit": int(device.get("unit", 1)),
                        "type": dev_type,
                        "max_read_registers": int(device.get("max_read_registers", 100)),
                        "max_write_registers": int(device.get("max_write_registers", 10)),
                        # IMPORTANT: use range for O(1) membership checks, not a huge list
                        "allowed_addresses": device.get("allowed_addresses", range(0, 65536)),
                    }
                    logger.info(f"Connected to Modbus device {device_id}")
                    self.audit.log_event("modbus_connection", "system", "connect", device_id, "success")
                else:
                    logger.warning(f"Failed to connect to Modbus device {device_id}")
                    self.audit.log_event("modbus_connection", "system", "connect", device_id, "failure")
            except Exception as e:
                logger.error(f"Error initializing Modbus device: {e}")

    def _check_rate_limit(self, device_id: str, operation: str) -> bool:
        key = f"{device_id}:{operation}"
        now = time.time()
        window = 60.0
        limit = 10

        times = self.rate_limiter.get(key, [])
        times = [t for t in times if now - t < window]
        if len(times) >= limit:
            logger.warning(f"Rate limit exceeded for {device_id}:{operation}")
            self.rate_limiter[key] = times
            return False
        times.append(now)
        self.rate_limiter[key] = times
        return True

    def read_registers(self, device_id: str, address: int, count: int = 1) -> List[int]:
        device_id = SecurityConfig.validate_device_id(device_id)
        address = SecurityConfig.validate_modbus_address(address)

        if not 1 <= int(count) <= 125:
            raise ValueError(f"Invalid register count: {count}")

        if not self._check_rate_limit(device_id, "read"):
            raise RuntimeError("Rate limit exceeded")

        with self.client_lock:
            info = self.clients.get(device_id)
            if not info:
                raise ValueError(f"Device {device_id} not found")

            if int(count) > int(info["max_read_registers"]):
                raise ValueError("Read count exceeds limit")

            allowed = info["allowed_addresses"]
            if address not in allowed:
                raise ValueError(f"Address {address} not allowed")

            client = info["client"]
            unit = info["unit"]

            try:
                result = client.read_holding_registers(address, count, unit=unit)
                if not result.isError():
                    values = list(result.registers)
                    self.audit.log_event(
                        "modbus_read", "system", "read_registers", f"{device_id}:{address}", "success", {"count": int(count)}
                    )
                    return values
                raise ModbusException(f"Modbus read error: {result}")
            except Exception as e:
                logger.error(f"Error reading Modbus {device_id}: {e}")
                self.audit.log_event(
                    "modbus_read", "system", "read_registers", f"{device_id}:{address}", "failure", {"error": str(e)}
                )
                raise

    def write_register(self, device_id: str, address: int, value: int):
        device_id = SecurityConfig.validate_device_id(device_id)
        address = SecurityConfig.validate_modbus_address(address)
        value = SecurityConfig.validate_modbus_value(value)

        if not self._check_rate_limit(device_id, "write"):
            raise RuntimeError("Rate limit exceeded")

        with self.client_lock:
            info = self.clients.get(device_id)
            if not info:
                raise ValueError(f"Device {device_id} not found")

            allowed = info["allowed_addresses"]
            if address not in allowed:
                raise ValueError(f"Write to address {address} not allowed")

            client = info["client"]
            unit = info["unit"]

            try:
                result = client.write_register(address, value, unit=unit)
                if not result.isError():
                    logger.info(f"Written {value} to {device_id}:{address}")
                    self.audit.log_event(
                        "modbus_write", "system", "write_register", f"{device_id}:{address}", "success", {"value": int(value)}
                    )
                    return
                raise ModbusException(f"Modbus write error: {result}")
            except Exception as e:
                logger.error(f"Error writing Modbus {device_id}: {e}")
                self.audit.log_event(
                    "modbus_write", "system", "write_register", f"{device_id}:{address}", "failure", {"error": str(e)}
                )
                raise

    def read_coils(self, device_id: str, address: int, count: int = 1) -> List[bool]:
        device_id = SecurityConfig.validate_device_id(device_id)
        address = SecurityConfig.validate_modbus_address(address)

        if not 1 <= int(count) <= 2000:
            raise ValueError(f"Invalid coil count: {count}")

        if not self._check_rate_limit(device_id, "read"):
            raise RuntimeError("Rate limit exceeded")

        with self.client_lock:
            info = self.clients.get(device_id)
            if not info:
                raise ValueError(f"Device {device_id} not found")

            client = info["client"]
            unit = info["unit"]

            try:
                result = client.read_coils(address, count, unit=unit)
                if not result.isError():
                    return list(result.bits[:count])
                raise ModbusException(f"Modbus read coils error: {result}")
            except Exception as e:
                logger.error(f"Error reading coils {device_id}: {e}")
                self.audit.log_event(
                    "modbus_read", "system", "read_coils", f"{device_id}:{address}", "failure", {"error": str(e)}
                )
                raise

    def write_coil(self, device_id: str, address: int, value: bool):
        device_id = SecurityConfig.validate_device_id(device_id)
        address = SecurityConfig.validate_modbus_address(address)
        if not isinstance(value, bool):
            raise ValueError(f"Invalid coil value: {value}")

        if not self._check_rate_limit(device_id, "write"):
            raise RuntimeError("Rate limit exceeded")

        with self.client_lock:
            info = self.clients.get(device_id)
            if not info:
                raise ValueError(f"Device {device_id} not found")

            client = info["client"]
            unit = info["unit"]

            try:
                result = client.write_coil(address, value, unit=unit)
                if not result.isError():
                    logger.info(f"Written coil {value} to {device_id}:{address}")
                    self.audit.log_event(
                        "modbus_write", "system", "write_coil", f"{device_id}:{address}", "success", {"value": bool(value)}
                    )
                    return
                raise ModbusException(f"Modbus write coil error: {result}")
            except Exception as e:
                logger.error(f"Error writing coil {device_id}: {e}")
                self.audit.log_event(
                    "modbus_write", "system", "write_coil", f"{device_id}:{address}", "failure", {"error": str(e)}
                )
                raise


# ==================== SECURE DATA STORE ====================

class SecureDataStore:
    """Secure data storage (Influx + Redis cache) with optional encryption."""

    def __init__(self, config: SecureConfig, audit_logger: AuditLogger):
        self.config = config
        self.audit = audit_logger

        self.influx_client: Optional[InfluxDBClient] = None
        self.write_api = None
        self.query_api = None

        self.redis_client: Optional[redis.Redis] = None
        self.redis_lock = threading.Lock()

        self.active_alarms: Dict[str, Alarm] = {}
        self.alarm_lock = threading.Lock()

        self._init_influx()
        self._init_redis()
        self._start_alarm_cleanup()

    def stop(self):
        try:
            if self.influx_client:
                self.influx_client.close()
        except Exception:
            pass
        try:
            if self.redis_client:
                self.redis_client.close()
        except Exception:
            pass

    def _init_influx(self):
        try:
            token = self.config.get("influxdb", "token")
            if token:
                self.influx_client = InfluxDBClient(
                    url=self.config.get("influxdb", "url", default="https://localhost:8086"),
                    token=token,
                    org=self.config.get("influxdb", "org", default="iot"),
                    verify_ssl=bool(self.config.get("influxdb", "verify_ssl", default=True)),
                    timeout=10000,
                )
                self.influx_client.ping()
                self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
                self.query_api = self.influx_client.query_api()
                logger.info("Connected to InfluxDB")
                self.audit.log_event("influx_connection", "system", "connect", "influxdb", "success")
        except Exception as e:
            logger.error(f"Failed to connect to InfluxDB: {e}")
            self.audit.log_event("influx_connection", "system", "connect", "influxdb", "failure", {"error": str(e)})

    def _init_redis(self):
        try:
            password = self.config.get("redis", "password")

            if self.config.get("redis", "use_sentinel", default=False):
                sentinels = self.config.get("redis", "sentinels", default=[]) or []
                master_name = self.config.get("redis", "master_name", default="mymaster")
                sentinel = redis.sentinel.Sentinel(
                    sentinels,
                    socket_connect_timeout=0.5,
                    password=password,
                )
                self.redis_client = sentinel.master_for(
                    master_name,
                    socket_connect_timeout=0.5,
                    decode_responses=True,
                    password=password,
                )
            else:
                self.redis_client = redis.Redis(
                    host=self.config.get("redis", "host", default="localhost"),
                    port=int(self.config.get("redis", "port", default=6379)),
                    db=int(self.config.get("redis", "db", default=0)),
                    password=password,
                    ssl=bool(self.config.get("redis", "ssl", default=False)),
                    decode_responses=True,
                    socket_connect_timeout=2,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )

            self.redis_client.ping()
            logger.info("Connected to Redis")
            self.audit.log_event("redis_connection", "system", "connect", "redis", "success")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.audit.log_event("redis_connection", "system", "connect", "redis", "failure", {"error": str(e)})
            self.redis_client = None

    def _start_alarm_cleanup(self):
        def cleanup():
            while True:
                time.sleep(3600)
                self._cleanup_old_alarms()

        threading.Thread(target=cleanup, daemon=True).start()

    def _cleanup_old_alarms(self):
        with self.alarm_lock:
            cutoff = datetime.utcnow() - timedelta(days=7)
            to_remove = []
            for alarm_id, alarm in self.active_alarms.items():
                if alarm.timestamp < cutoff and alarm.acknowledged:
                    to_remove.append(alarm_id)
            for alarm_id in to_remove:
                del self.active_alarms[alarm_id]
                logger.info(f"Cleaned up old alarm: {alarm_id}")

    def store_sensor_reading(self, reading: SensorReading):
        try:
            # Influx
            if self.write_api:
                point = (
                    Point("sensor_reading")
                    .tag("sensor_id", reading.sensor_id)
                    .tag("unit", reading.unit)
                    .field("value", float(reading.value))
                    .field("quality", int(reading.quality))
                    .time(reading.timestamp)
                )

                if reading.metadata:
                    for key, value in reading.metadata.items():
                        if isinstance(key, str) and isinstance(value, (str, int, float, bool)):
                            point.tag(SecurityConfig.sanitize_string(str(key), 50), str(value))

                bucket = self.config.get("influxdb", "bucket", default="sensors")
                self.write_api.write(bucket=bucket, record=point)

            # Redis cache
            if self.redis_client:
                with self.redis_lock:
                    cache_key = f"sensor:{reading.sensor_id}:last"
                    cache_data = {
                        "value": reading.value,
                        "unit": reading.unit,
                        "timestamp": reading.timestamp.isoformat() + "Z",
                        "quality": reading.quality,
                    }
                    if self.config.get("security", "enable_encryption", default=True):
                        cache_data = {"encrypted": SecurityConfig.encrypt_data(json.dumps(cache_data, separators=(",", ":")))}

                    self.redis_client.hset(cache_key, mapping=cache_data)
                    self.redis_client.expire(cache_key, 3600)

        except RedisConnectionError:
            logger.warning("Redis connection lost, attempting reconnect")
            self._init_redis()
        except Exception as e:
            logger.error(f"Error storing sensor reading: {e}")
            metrics_errors.labels(type="storage").inc()

    def query_sensor_history(
        self,
        sensor_id: str,
        start_time: datetime,
        end_time: datetime,
        aggregation: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        sensor_id = SecurityConfig.validate_sensor_id(sensor_id)

        time_diff_hours = (end_time - start_time).total_seconds() / 3600
        if time_diff_hours > SecurityConfig.MAX_QUERY_HOURS:
            end_time = start_time + timedelta(hours=SecurityConfig.MAX_QUERY_HOURS)
            logger.warning(f"Query time range limited to {SecurityConfig.MAX_QUERY_HOURS} hours")

        if time_diff_hours > 24 and not aggregation:
            aggregation = "mean"

        allowed_aggregations = {"mean", "max", "min", "sum", "count", "median"}
        if aggregation and aggregation not in allowed_aggregations:
            raise ValueError(f"Invalid aggregation: {aggregation}")

        if not self.query_api:
            return []

        try:
            bucket = self.config.get("influxdb", "bucket", default="sensors")
            # sensor_id is pattern-validated; still use strict formatting
            start = (start_time.replace(microsecond=0)).isoformat() + "Z"
            stop = (end_time.replace(microsecond=0)).isoformat() + "Z"

            if aggregation:
                query = f"""
from(bucket: "{bucket}")
  |> range(start: {start}, stop: {stop})
  |> filter(fn: (r) => r["sensor_id"] == "{sensor_id}")
  |> filter(fn: (r) => r["_field"] == "value")
  |> aggregateWindow(every: 1m, fn: {aggregation}, createEmpty: false)
  |> limit(n: {SecurityConfig.MAX_QUERY_SIZE})
"""
            else:
                query = f"""
from(bucket: "{bucket}")
  |> range(start: {start}, stop: {stop})
  |> filter(fn: (r) => r["sensor_id"] == "{sensor_id}")
  |> filter(fn: (r) => r["_field"] == "value")
  |> limit(n: {SecurityConfig.MAX_QUERY_SIZE})
"""

            result = self.query_api.query(query=query)
            data: List[Dict[str, Any]] = []
            for table in result:
                for record in table.records:
                    data.append(
                        {
                            "timestamp": record.get_time().isoformat(),
                            "value": record.get_value(),
                            "sensor_id": sensor_id,
                        }
                    )
            return data

        except Exception as e:
            logger.error(f"Error querying sensor history: {e}")
            metrics_errors.labels(type="influx_query").inc()
            return []

    def get_cached_value(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.redis_client:
            return None

        with self.redis_lock:
            try:
                data = self.redis_client.hgetall(key)
                if not data:
                    return None
                if "encrypted" in data and self.config.get("security", "enable_encryption", default=True):
                    decrypted = SecurityConfig.decrypt_data(data["encrypted"])
                    return json.loads(decrypted)
                return data
            except RedisConnectionError:
                logger.warning("Redis connection lost")
                self._init_redis()
                return None
            except Exception as e:
                logger.error(f"Error reading cache: {e}")
                return None


# ==================== ALARM MANAGER ====================

def _alarm_to_redis_hash(alarm: Alarm) -> Dict[str, str]:
    return {
        "alarm_id": alarm.alarm_id,
        "sensor_id": alarm.sensor_id,
        "timestamp": alarm.timestamp.isoformat() + "Z",
        "priority": alarm.priority.name,
        "message": alarm.message,
        "threshold_value": str(alarm.threshold_value),
        "actual_value": str(alarm.actual_value),
        "acknowledged": "true" if alarm.acknowledged else "false",
        "acknowledged_by": alarm.acknowledged_by or "",
        "acknowledged_at": (alarm.acknowledged_at.isoformat() + "Z") if alarm.acknowledged_at else "",
    }


class AlarmManager:
    """Alarm management system"""

    def __init__(self, data_store: SecureDataStore):
        self.data_store = data_store
        self.alarm_rules = self._load_alarm_rules()

    def _load_alarm_rules(self) -> Dict[str, Dict[str, Any]]:
        # Replace with config-driven rules if needed
        return {
            "high_temperature": {
                "sensor_type": SensorType.TEMPERATURE,
                "condition": "greater_than",
                "threshold": 80.0,
                "priority": AlarmPriority.HIGH,
                "message": "High temperature detected",
            },
            "low_pressure": {
                "sensor_type": SensorType.PRESSURE,
                "condition": "less_than",
                "threshold": 2.0,
                "priority": AlarmPriority.CRITICAL,
                "message": "Critical low pressure",
            },
        }

    def check_alarm_condition(self, sensor_id: str, value: float, sensor_type: SensorType) -> Optional[Alarm]:
        for rule_id, rule in self.alarm_rules.items():
            if rule["sensor_type"] != sensor_type:
                continue

            threshold = float(rule["threshold"])
            triggered = False

            if rule["condition"] == "greater_than" and float(value) > threshold:
                triggered = True
            elif rule["condition"] == "less_than" and float(value) < threshold:
                triggered = True

            if triggered:
                alarm_id = f"{sensor_id}_{rule_id}_{int(time.time())}"
                alarm = Alarm(
                    alarm_id=alarm_id,
                    sensor_id=sensor_id,
                    timestamp=datetime.utcnow(),
                    priority=rule["priority"],
                    message=rule["message"],
                    threshold_value=threshold,
                    actual_value=float(value),
                    acknowledged=False,
                )

                with self.data_store.alarm_lock:
                    self.data_store.active_alarms[alarm_id] = alarm

                if self.data_store.redis_client:
                    alarm_key = f"alarm:{alarm_id}"
                    try:
                        self.data_store.redis_client.hset(alarm_key, mapping=_alarm_to_redis_hash(alarm))
                        self.data_store.redis_client.expire(alarm_key, 86400)
                    except Exception:
                        pass

                logger.warning(f"Alarm triggered: {alarm.message} - Value: {value}")
                metrics_active_alarms.inc()
                return alarm

        return None

    def acknowledge_alarm(self, alarm_id: str, user: str = "system") -> bool:
        with self.data_store.alarm_lock:
            alarm = self.data_store.active_alarms.get(alarm_id)
            if not alarm:
                return False
            alarm.acknowledged = True
            alarm.acknowledged_by = user
            alarm.acknowledged_at = datetime.utcnow()

        if self.data_store.redis_client:
            alarm_key = f"alarm:{alarm_id}"
            try:
                self.data_store.redis_client.hset(
                    alarm_key,
                    mapping={
                        "acknowledged": "true",
                        "acknowledged_by": user,
                        "acknowledged_at": alarm.acknowledged_at.isoformat() + "Z",
                    },
                )
            except Exception:
                pass

        logger.info(f"Alarm {alarm_id} acknowledged by {user}")
        return True

    def get_active_alarms(self, priority: Optional[AlarmPriority] = None) -> List[Alarm]:
        with self.data_store.alarm_lock:
            alarms = list(self.data_store.active_alarms.values())

        if priority:
            alarms = [a for a in alarms if a.priority == priority]

        alarms.sort(key=lambda x: (-x.priority.value, x.timestamp))
        return alarms


# ==================== IOT MANAGER ====================

class IoTManager:
    """Main IoT system manager"""

    def __init__(self):
        self.config = SecureConfig()
        self.audit = AuditLogger()

        self.mqtt = SecureMQTTConnector(self.config, self.audit)
        self.modbus = SecureModbusConnector(self.config, self.audit)
        self.data_store = SecureDataStore(self.config, self.audit)
        self.alarm_manager = AlarmManager(self.data_store)

        self.devices: Dict[str, Device] = {}
        self.sensors: Dict[str, Device] = {}
        self.actuators: Dict[str, Device] = {}
        self.device_lock = threading.Lock()

        self.command_timestamps: List[datetime] = []
        self.command_lock = threading.Lock()

        self._init_devices()
        self._start_monitoring()

    def shutdown(self):
        try:
            self.mqtt.stop()
        except Exception:
            pass
        try:
            self.modbus.stop()
        except Exception:
            pass
        try:
            self.data_store.stop()
        except Exception:
            pass

    def _init_devices(self):
        mqtt_devices = self.config.get("devices", "mqtt", default=[]) or []
        for device_config in mqtt_devices:
            try:
                device = Device(
                    device_id=device_config["id"],
                    name=device_config["name"],
                    type=device_config["type"],
                    protocol="mqtt",
                    address=device_config["topic"],
                    status="unknown",
                    last_seen=datetime.utcnow(),
                    metadata=device_config.get("metadata", {}),
                )
                with self.device_lock:
                    self.devices[device.device_id] = device
                    if device.type == "sensor":
                        self.sensors[device.device_id] = device
                        self.mqtt.subscribe_sensor(device.device_id, device.address)
                    elif device.type == "actuator":
                        self.actuators[device.device_id] = device
            except Exception as e:
                logger.error(f"Error initializing device: {e}")

        modbus_devices = self.config.get("modbus", "devices", default=[]) or []
        for device_config in modbus_devices:
            try:
                device = Device(
                    device_id=device_config["device_id"],
                    name=device_config.get("name", device_config["device_id"]),
                    type="plc",
                    protocol="modbus",
                    address=f"{device_config.get('host', 'serial')}:{device_config.get('port', 502)}",
                    status="unknown",
                    last_seen=datetime.utcnow(),
                    metadata=device_config,
                )
                with self.device_lock:
                    self.devices[device.device_id] = device
            except Exception as e:
                logger.error(f"Error initializing Modbus device: {e}")

    def _start_monitoring(self):
        def monitor():
            while True:
                time.sleep(30)
                self._check_device_status()
                self._update_metrics()

        threading.Thread(target=monitor, daemon=True).start()

    def _check_device_status(self):
        with self.device_lock:
            now = datetime.utcnow()
            for device in self.devices.values():
                if (now - device.last_seen).total_seconds() > 300:
                    if device.status != "offline":
                        device.status = "offline"
                        logger.warning(f"Device {device.device_id} is offline")

    def _update_metrics(self):
        with self.device_lock:
            online_devices = len([d for d in self.devices.values() if d.status == "online"])
            metrics_active_devices.set(online_devices)
        with self.data_store.alarm_lock:
            metrics_active_alarms.set(len(self.data_store.active_alarms))

    def _check_command_rate_limit(self) -> bool:
        with self.command_lock:
            now = datetime.utcnow()
            self.command_timestamps = [ts for ts in self.command_timestamps if (now - ts).total_seconds() < 60]
            if len(self.command_timestamps) >= SecurityConfig.MAX_COMMANDS_PER_MINUTE:
                logger.warning("Command rate limit reached")
                return False
            self.command_timestamps.append(now)
            return True

    def read_sensor(self, sensor_id: str) -> Dict[str, Any]:
        sensor_id = SecurityConfig.validate_sensor_id(sensor_id)

        with self.device_lock:
            sensor = self.sensors.get(sensor_id)
            if not sensor:
                raise ValueError(f"Sensor {sensor_id} not found")

        try:
            if sensor.protocol == "mqtt":
                value = self.mqtt.get_sensor_value(sensor.address)

                if value is None:
                    cache_key = f"sensor:{sensor_id}:last"
                    cached = self.data_store.get_cached_value(cache_key)
                    if cached:
                        return {
                            "sensor_id": sensor_id,
                            "value": cached.get("value"),
                            "unit": cached.get("unit"),
                            "timestamp": cached.get("timestamp"),
                            "quality": cached.get("quality"),
                            "source": "cache",
                        }
                    raise ValueError(f"No data available for sensor {sensor_id}")

                # Expecting dict payload
                if not isinstance(value, dict):
                    raise ValueError("Invalid live payload (expected object)")

                reading = SensorReading(
                    sensor_id=sensor_id,
                    timestamp=datetime.utcnow(),
                    value=float(value.get("value", 0)),
                    unit=str(value.get("unit", "")),
                    quality=int(value.get("quality", 100)),
                    metadata=value.get("metadata", {}) if isinstance(value.get("metadata"), dict) else {},
                )

                self.data_store.store_sensor_reading(reading)

                sensor_type_str = str(sensor.metadata.get("sensor_type", "analog"))
                if sensor_type_str in {e.value for e in SensorType}:
                    sensor_type = SensorType(sensor_type_str)
                    self.alarm_manager.check_alarm_condition(sensor_id, reading.value, sensor_type)

                with self.device_lock:
                    sensor.status = "online"
                    sensor.last_seen = datetime.utcnow()

                return {
                    "sensor_id": sensor_id,
                    "value": reading.value,
                    "unit": reading.unit,
                    "timestamp": reading.timestamp.isoformat() + "Z",
                    "quality": reading.quality,
                    "metadata": reading.metadata,
                    "source": "live",
                }

            if sensor.protocol == "modbus":
                device_id = sensor.metadata.get("modbus_device")
                address = int(sensor.metadata.get("register_address", 0))
                scale = float(sensor.metadata.get("scale", 1.0))

                if not device_id:
                    raise ValueError("Missing modbus_device metadata")

                registers = self.modbus.read_registers(str(device_id), address, 1)
                if not registers:
                    raise ValueError("No Modbus data returned")

                value_num = float(registers[0]) * scale
                reading = SensorReading(
                    sensor_id=sensor_id,
                    timestamp=datetime.utcnow(),
                    value=value_num,
                    unit=str(sensor.metadata.get("unit", "")),
                    quality=100,
                    metadata={"modbus_address": address},
                )
                self.data_store.store_sensor_reading(reading)

                return {
                    "sensor_id": sensor_id,
                    "value": value_num,
                    "unit": reading.unit,
                    "timestamp": reading.timestamp.isoformat() + "Z",
                    "quality": 100,
                    "source": "modbus",
                }

            raise ValueError(f"Unsupported protocol: {sensor.protocol}")

        except Exception as e:
            logger.error(f"Error reading sensor {sensor_id}: {e}")
            with self.device_lock:
                # Mark sensor error if present
                if sensor_id in self.sensors:
                    self.sensors[sensor_id].status = "error"
            raise

    def execute_command(self, actuator_id: str, command: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        actuator_id = SecurityConfig.validate_device_id(actuator_id)
        command = SecurityConfig.validate_command(command)

        if not self._check_command_rate_limit():
            raise ValueError("Command rate limit exceeded")

        with self.device_lock:
            actuator = self.actuators.get(actuator_id)
            if not actuator:
                raise ValueError(f"Actuator {actuator_id} not found")

        parameters = SecurityConfig.sanitize_dict(parameters) if parameters else {}

        try:
            if actuator.protocol == "mqtt":
                payload = {"command": command, "parameters": parameters}
                self.mqtt.publish_command(actuator.address, payload)
                result = {
                    "actuator_id": actuator_id,
                    "command": command,
                    "parameters": parameters,
                    "status": "sent",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }

            elif actuator.protocol == "modbus":
                device_id = actuator.metadata.get("modbus_device")
                if not device_id:
                    raise ValueError("Missing modbus_device metadata")

                if command == "write_register":
                    address = SecurityConfig.validate_modbus_address(int(parameters.get("address", 0)))
                    value = SecurityConfig.validate_modbus_value(int(parameters.get("value", 0)))
                    self.modbus.write_register(str(device_id), address, value)

                elif command == "write_coil":
                    address = SecurityConfig.validate_modbus_address(int(parameters.get("address", 0)))
                    value = bool(parameters.get("value", False))
                    self.modbus.write_coil(str(device_id), address, value)

                else:
                    raise ValueError(f"Unsupported Modbus command: {command}")

                result = {
                    "actuator_id": actuator_id,
                    "command": command,
                    "parameters": parameters,
                    "status": "executed",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
            else:
                raise ValueError(f"Unsupported protocol: {actuator.protocol}")

            self.audit.log_event("actuator_command", "system", command, actuator_id, "success", parameters)
            return result

        except Exception as e:
            logger.error(f"Error executing command on {actuator_id}: {e}")
            self.audit.log_event("actuator_command", "system", command, actuator_id, "failure", {"error": str(e)})
            raise

    def get_device_topology(self) -> Dict[str, Any]:
        with self.device_lock:
            return {
                "total_devices": len(self.devices),
                "sensors": len(self.sensors),
                "actuators": len(self.actuators),
                "protocols": {
                    "mqtt": len([d for d in self.devices.values() if d.protocol == "mqtt"]),
                    "modbus": len([d for d in self.devices.values() if d.protocol == "modbus"]),
                },
                "status": {
                    "online": len([d for d in self.devices.values() if d.status == "online"]),
                    "offline": len([d for d in self.devices.values() if d.status == "offline"]),
                    "error": len([d for d in self.devices.values() if d.status == "error"]),
                },
                "devices": [
                    {
                        "id": d.device_id,
                        "name": d.name,
                        "type": d.type,
                        "protocol": d.protocol,
                        "status": d.status,
                        "last_seen": d.last_seen.isoformat() + "Z",
                    }
                    for d in self.devices.values()
                ],
            }


# ==================== GLOBAL MANAGER (created at startup) ====================

_iot_manager: Optional[IoTManager] = None


def get_manager() -> IoTManager:
    global _iot_manager
    if _iot_manager is None:
        raise RuntimeError("IoTManager not initialized yet")
    return _iot_manager


# ==================== TOOL DECORATOR ====================

def tool_guarded(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            # Tool duration (best-effort)
            try:
                metrics_command_duration.observe(elapsed)
            except Exception:
                pass

    return wrapper


# ==================== MCP TOOLS ====================

@tool_guarded
def read_sensor(sensor_id: str) -> Dict[str, Any]:
    """Read current value from an IoT sensor."""
    return get_manager().read_sensor(sensor_id)


@tool_guarded
def read_multiple_sensors(sensor_ids: List[str]) -> List[Dict[str, Any]]:
    """Read values from multiple sensors."""
    results = []
    for sensor_id in sensor_ids:
        try:
            results.append(get_manager().read_sensor(sensor_id))
        except Exception as e:
            results.append(
                {"sensor_id": str(sensor_id), "error": str(e), "timestamp": datetime.utcnow().isoformat() + "Z"}
            )
    return results


@tool_guarded
def get_sensor_history(sensor_id: str, hours: int = 24, aggregation: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get historical readings from a sensor (default 24h, max 168h)."""
    sensor_id = SecurityConfig.validate_sensor_id(sensor_id)
    hours = SecurityConfig.validate_hours_param(hours)

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)

    return get_manager().data_store.query_sensor_history(sensor_id, start_time, end_time, aggregation)


@tool_guarded
def execute_actuator_command(actuator_id: str, command: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute command on an actuator."""
    return get_manager().execute_command(actuator_id, command, parameters)


@tool_guarded
def get_device_topology() -> Dict[str, Any]:
    """Get complete IoT system topology."""
    return get_manager().get_device_topology()


@tool_guarded
def list_devices(device_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all registered IoT devices (optional filter by type)."""
    mgr = get_manager()
    with mgr.device_lock:
        devices = list(mgr.devices.values())
        if device_type:
            device_type = SecurityConfig.sanitize_string(device_type, 20)
            devices = [d for d in devices if d.type == device_type]
        return [
            {
                "device_id": d.device_id,
                "name": d.name,
                "type": d.type,
                "protocol": d.protocol,
                "address": d.address,
                "status": d.status,
                "last_seen": d.last_seen.isoformat() + "Z",
                "metadata": d.metadata,
            }
            for d in devices
        ]


@tool_guarded
def get_active_alarms(priority: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get list of active system alarms (optional filter by priority)."""
    mgr = get_manager()
    priority_enum = None
    if priority:
        priority = SecurityConfig.sanitize_string(priority, 20)
        try:
            priority_enum = AlarmPriority[priority.upper()]
        except KeyError:
            raise ValueError(f"Invalid priority: {priority}")

    alarms = mgr.alarm_manager.get_active_alarms(priority_enum)
    return [
        {
            "alarm_id": a.alarm_id,
            "sensor_id": a.sensor_id,
            "timestamp": a.timestamp.isoformat() + "Z",
            "priority": a.priority.name,
            "message": a.message,
            "threshold_value": a.threshold_value,
            "actual_value": a.actual_value,
            "acknowledged": a.acknowledged,
            "acknowledged_by": a.acknowledged_by,
            "acknowledged_at": (a.acknowledged_at.isoformat() + "Z") if a.acknowledged_at else None,
        }
        for a in alarms
    ]


@tool_guarded
def acknowledge_alarm(alarm_id: str) -> bool:
    """Acknowledge an alarm."""
    alarm_id = SecurityConfig.sanitize_string(alarm_id, 200)
    return get_manager().alarm_manager.acknowledge_alarm(alarm_id, "system")


@tool_guarded
def read_modbus_registers(device_id: str, address: int, count: int = 1) -> List[int]:
    """Read Modbus registers."""
    return get_manager().modbus.read_registers(device_id, address, count)


@tool_guarded
def write_modbus_register(device_id: str, address: int, value: int) -> Dict[str, Any]:
    """Write value to Modbus register."""
    get_manager().modbus.write_register(device_id, address, value)
    return {
        "device_id": device_id,
        "address": int(address),
        "value": int(value),
        "status": "written",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@tool_guarded
def read_modbus_coils(device_id: str, address: int, count: int = 1) -> List[bool]:
    """Read Modbus coils."""
    return get_manager().modbus.read_coils(device_id, address, count)


@tool_guarded
def write_modbus_coil(device_id: str, address: int, value: bool) -> Dict[str, Any]:
    """Write value to Modbus coil."""
    get_manager().modbus.write_coil(device_id, address, value)
    return {
        "device_id": device_id,
        "address": int(address),
        "value": bool(value),
        "status": "written",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@tool_guarded
def get_system_status() -> Dict[str, Any]:
    """Get overall IoT system status."""
    mgr = get_manager()
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "connections": {
            "mqtt": {"connected": mgr.mqtt.connected, "broker": mgr.config.get("mqtt", "broker"), "subscriptions": len(mgr.mqtt.subscriptions)},
            "modbus": {"devices": len(mgr.modbus.clients), "active": list(mgr.modbus.clients.keys())},
            "influxdb": {"connected": mgr.data_store.influx_client is not None},
            "redis": {"connected": mgr.data_store.redis_client is not None},
        },
        "devices": {
            "total": len(mgr.devices),
            "online": len([d for d in mgr.devices.values() if d.status == "online"]),
            "errors": len([d for d in mgr.devices.values() if d.status == "error"]),
        },
        "alarms": {
            "active": len(mgr.data_store.active_alarms),
            "unacknowledged": len([a for a in mgr.data_store.active_alarms.values() if not a.acknowledged]),
        },
        "rate_limits": {"commands_per_minute": SecurityConfig.MAX_COMMANDS_PER_MINUTE},
        "security": {
            "env": SecurityConfig.ENV,
            "encryption_enabled": bool(mgr.config.get("security", "enable_encryption", default=True)),
            "audit_enabled": bool(mgr.config.get("security", "enable_audit_log", default=True)),
            "auth_enabled": bool(mgr.config.get("security", "enable_auth", default=True)),
            "ip_allowlist": [str(n) for n in SecurityConfig.ALLOWED_IP_RANGES],
        },
    }


# ==================== MCP TOOLS EXPOSURE ====================

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
    read_modbus_coils,
    write_modbus_coil,
    get_system_status,
]

app = expose_tools(
    tools=tools,
    title="IoT/Edge MCP Server - Secure Production",
    description="Production-ready MCP server for IoT infrastructure, Edge computing and industrial automation with full security",
    version="2.1.0",
)

# ==================== SECURITY MIDDLEWARE (AUTH + IP + LIMITS) ====================

_request_times: Dict[str, List[float]] = {}
_request_lock = threading.Lock()

def _rate_limit_ok(client_ip: str) -> bool:
    now = time.time()
    window = 60.0
    limit = SecurityConfig.MAX_REQUESTS_PER_MINUTE

    with _request_lock:
        times = _request_times.get(client_ip, [])
        times = [t for t in times if now - t < window]
        if len(times) >= limit:
            _request_times[client_ip] = times
            return False
        times.append(now)
        _request_times[client_ip] = times
        return True


def _is_public_path(path: str, protect_docs: bool) -> bool:
    # Always allow health/metrics. Docs can be protected via config.
    if path in {"/health", "/metrics"}:
        return True
    if not protect_docs and path in {"/docs", "/openapi.json", "/redoc"}:
        return True
    return False


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    endpoint = request.url.path
    method = request.method.upper()

    try:
        metrics_requests.labels(method=method, endpoint=endpoint).inc()
    except Exception:
        pass

    # Body size limit (best-effort via Content-Length)
    cl = request.headers.get("content-length")
    if cl:
        try:
            if int(cl) > SecurityConfig.MAX_PAYLOAD_SIZE:
                raise HTTPException(status_code=413, detail="Payload too large")
        except ValueError:
            pass

    # Basic request line length check
    if len(str(request.url)) > 4096:
        raise HTTPException(status_code=414, detail="URI too long")

    # If manager not initialized yet
    global _iot_manager
    if _iot_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    protect_docs = bool(_iot_manager.config.get("security", "protect_docs", default=(SecurityConfig.ENV == "production")))
    if _is_public_path(endpoint, protect_docs=protect_docs):
        return await call_next(request)

    # IP allowlist
    client_ip = request.client.host if request.client else ""
    if not SecurityConfig.validate_ip_address(client_ip):
        raise HTTPException(status_code=403, detail="IP not allowed")

    # Rate limit (per IP)
    if not _rate_limit_ok(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Auth toggle
    if not bool(_iot_manager.config.get("security", "enable_auth", default=True)):
        return await call_next(request)

    # API key auth
    api_key = request.headers.get(SecurityConfig.API_KEY_HEADER, "")
    if api_key:
        for stored_key in SecurityConfig.API_KEYS.keys():
            if secrets.compare_digest(api_key, stored_key):
                return await call_next(request)

    # JWT bearer auth
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth.split(" ", 1)[1].strip()
        try:
            jwt.decode(token, SecurityConfig.JWT_SECRET_KEY, algorithms=[SecurityConfig.JWT_ALGORITHM])
            return await call_next(request)
        except jwt.PyJWTError:
            pass

    raise HTTPException(status_code=401, detail="Unauthorized")


# ==================== EXTRA ENDPOINTS ====================

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain; version=0.0.4; charset=utf-8")


# ==================== STARTUP / SHUTDOWN ====================

@app.on_event("startup")
async def on_startup():
    global _iot_manager
    if _iot_manager is None:
        _iot_manager = IoTManager()
        logger.info("IoTManager initialized")

@app.on_event("shutdown")
async def on_shutdown():
    global _iot_manager
    if _iot_manager is not None:
        try:
            _iot_manager.shutdown()
        except Exception:
            pass
        _iot_manager = None
        logger.info("IoTManager stopped")


# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting IoT/Edge MCP Server (Security Hardened)")
    logger.info(f"Available MCP tools: {[t.__name__ for t in tools]}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level="info",
        access_log=True,
        use_colors=False,
    )
