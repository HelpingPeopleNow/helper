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
