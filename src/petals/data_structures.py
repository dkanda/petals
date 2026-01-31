import dataclasses
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple

import pydantic.v1 as pydantic
from hivemind import PeerID
from hivemind.moe.expert_uid import ExpertUID

ModuleUID = str
UID_DELIMITER = "."  # delimits parts of one module uid, e.g. "bloom.transformer.h.4.self_attention"
CHAIN_DELIMITER = " "  # delimits multiple uids in a sequence, e.g. "bloom.layer3 bloom.layer4"


def parse_uid(uid: ModuleUID) -> Tuple[str, int]:
    if CHAIN_DELIMITER in uid:
        raise ValueError(f"parse_uid() does not support chained UIDs (got: {uid!r})")
    if UID_DELIMITER not in uid:
        raise ValueError(f"parse_uid() expects UID in format 'prefix{UID_DELIMITER}index' (got: {uid!r})")

    dht_prefix, index = uid.rsplit(UID_DELIMITER, 1)
    try:
        index = int(index)
    except ValueError:
        raise ValueError(f"parse_uid() expects index to be an integer (got: {index!r} in {uid!r})")
    return dht_prefix, index


@pydantic.dataclasses.dataclass
class ModelInfo:
    num_blocks: pydantic.conint(ge=1, strict=True)
    repository: Optional[str] = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, source: dict):
        if not isinstance(source, dict):
            raise TypeError("source must be a dict")

        known_fields = {f.name for f in dataclasses.fields(cls)}
        unknown_fields = set(source.keys()) - known_fields
        if unknown_fields:
            raise ValueError(f"Unknown fields in ModelInfo: {sorted(unknown_fields)}")

        return cls(**source)


class ServerState(Enum):
    OFFLINE = 0
    JOINING = 1
    ONLINE = 2


RPS = pydantic.confloat(ge=0, allow_inf_nan=False, strict=True)


@pydantic.dataclasses.dataclass
class ServerInfo:
    state: ServerState
    throughput: RPS

    start_block: Optional[pydantic.conint(ge=0, strict=True)] = None
    end_block: Optional[pydantic.conint(ge=0, strict=True)] = None

    public_name: Optional[str] = None
    version: Optional[str] = None

    network_rps: Optional[RPS] = None
    forward_rps: Optional[RPS] = None
    inference_rps: Optional[RPS] = None

    adapters: Sequence[str] = ()
    torch_dtype: Optional[str] = None
    quant_type: Optional[str] = None
    using_relay: Optional[bool] = None
    cache_tokens_left: Optional[pydantic.conint(ge=0, strict=True)] = None
    next_pings: Optional[Dict[str, pydantic.confloat(ge=0, strict=True)]] = None

    def __post_init__(self):
        if self.start_block is not None and self.end_block is not None and self.start_block >= self.end_block:
            raise ValueError(f"start_block ({self.start_block}) must be less than end_block ({self.end_block})")

    def to_tuple(self) -> Tuple[int, float, dict]:
        extra_info = dataclasses.asdict(self)
        del extra_info["state"], extra_info["throughput"]
        return (self.state.value, self.throughput, extra_info)

    @classmethod
    def from_tuple(cls, source: tuple):
        if not isinstance(source, tuple):
            raise TypeError("info must be a tuple")
        if len(source) != 3:
            raise ValueError(f"info must have exactly 3 elements (got {len(source)})")
        if not isinstance(source[0], int):
            raise TypeError("info[0] must be an int")
        if not isinstance(source[1], (float, int)):
            raise TypeError("info[1] must be a float or int")

        state, throughput, extra_info = source

        if not isinstance(extra_info, dict):
            raise TypeError("info[2] must be a dict")

        try:
            state = ServerState(state)
        except ValueError:
            raise ValueError(f"Invalid server state: {state}")

        known_fields = {f.name for f in dataclasses.fields(cls)}
        unknown_fields = set(extra_info.keys()) - known_fields
        if unknown_fields:
            raise ValueError(f"Unknown fields in ServerInfo: {sorted(unknown_fields)}")

        # pydantic will validate existing fields and ignore extra ones
        return cls(state=state, throughput=throughput, **extra_info)


@dataclasses.dataclass
class RemoteModuleInfo:
    """A remote module that is served by one or more servers"""

    uid: ModuleUID
    servers: Dict[PeerID, ServerInfo]


@dataclasses.dataclass
class RemoteSpanInfo:
    """A chain of remote blocks served by one specific remote peer"""

    peer_id: PeerID
    start: int
    end: int
    server_info: ServerInfo

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def state(self) -> ServerState:
        return self.server_info.state

    @property
    def throughput(self) -> float:
        return self.server_info.throughput


RPCInfo = Dict[str, Any]

Handle = int


@dataclasses.dataclass(frozen=True)
class InferenceMetadata:
    uid: ExpertUID
    prefix_length: int
    cache_handles: Tuple[Handle, ...]
    active_adapter: Optional[str]
