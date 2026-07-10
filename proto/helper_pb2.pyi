from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Message(_message.Message):
    __slots__ = ("role", "content")
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    role: str
    content: str
    def __init__(self, role: _Optional[str] = ..., content: _Optional[str] = ...) -> None: ...

class AskRequest(_message.Message):
    __slots__ = ("question", "history", "system_prompt", "llm_provider", "skip_role_detection")
    QUESTION_FIELD_NUMBER: _ClassVar[int]
    HISTORY_FIELD_NUMBER: _ClassVar[int]
    SYSTEM_PROMPT_FIELD_NUMBER: _ClassVar[int]
    LLM_PROVIDER_FIELD_NUMBER: _ClassVar[int]
    SKIP_ROLE_DETECTION_FIELD_NUMBER: _ClassVar[int]
    question: str
    history: _containers.RepeatedCompositeFieldContainer[Message]
    system_prompt: str
    llm_provider: str
    skip_role_detection: bool
    def __init__(self, question: _Optional[str] = ..., history: _Optional[_Iterable[_Union[Message, _Mapping]]] = ..., system_prompt: _Optional[str] = ..., llm_provider: _Optional[str] = ..., skip_role_detection: _Optional[bool] = ...) -> None: ...

class AskResponse(_message.Message):
    __slots__ = ("answer", "detected_role")
    ANSWER_FIELD_NUMBER: _ClassVar[int]
    DETECTED_ROLE_FIELD_NUMBER: _ClassVar[int]
    answer: str
    detected_role: str
    def __init__(self, answer: _Optional[str] = ..., detected_role: _Optional[str] = ...) -> None: ...

class EmbedRequest(_message.Message):
    __slots__ = ("text", "model")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    text: str
    model: str
    def __init__(self, text: _Optional[str] = ..., model: _Optional[str] = ...) -> None: ...

class EmbedResponse(_message.Message):
    __slots__ = ("embedding", "model", "dimensions")
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    embedding: _containers.RepeatedScalarFieldContainer[float]
    model: str
    dimensions: int
    def __init__(self, embedding: _Optional[_Iterable[float]] = ..., model: _Optional[str] = ..., dimensions: _Optional[int] = ...) -> None: ...

class EmbedBatchRequest(_message.Message):
    __slots__ = ("texts", "model")
    TEXTS_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    texts: _containers.RepeatedScalarFieldContainer[str]
    model: str
    def __init__(self, texts: _Optional[_Iterable[str]] = ..., model: _Optional[str] = ...) -> None: ...

class EmbedBatchItem(_message.Message):
    __slots__ = ("index", "status", "embedding", "model", "dimensions", "error")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    index: int
    status: str
    embedding: _containers.RepeatedScalarFieldContainer[float]
    model: str
    dimensions: int
    error: str
    def __init__(self, index: _Optional[int] = ..., status: _Optional[str] = ..., embedding: _Optional[_Iterable[float]] = ..., model: _Optional[str] = ..., dimensions: _Optional[int] = ..., error: _Optional[str] = ...) -> None: ...

class EmbedBatchResponse(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[EmbedBatchItem]
    def __init__(self, items: _Optional[_Iterable[_Union[EmbedBatchItem, _Mapping]]] = ...) -> None: ...
