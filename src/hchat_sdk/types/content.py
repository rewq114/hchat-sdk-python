from typing import Literal, Union, List, Optional, Dict, Any
from pydantic import BaseModel, Field

# ========================================
# CONTENT SOURCES
# ========================================

class Base64ImageSource(BaseModel):
    type: Literal['base64']
    media_type: Literal['image/jpeg', 'image/png', 'image/gif', 'image/webp']
    data: str

class URLImageSource(BaseModel):
    type: Literal['url']
    url: str

class FileImageSource(BaseModel):
    type: Literal['file']
    file_id: str

ImageSource = Union[Base64ImageSource, URLImageSource, FileImageSource]

# ========================================
# CONTENT BLOCKS
# ========================================

class TextContent(BaseModel):
    type: Literal['text'] = 'text'
    text: str

class ImageContent(BaseModel):
    type: Literal['image'] = 'image'
    source: ImageSource

class ImageUrlContent(BaseModel):
    type: Literal['imageUrl'] = 'imageUrl'
    url: str

class ToolUseContent(BaseModel):
    type: Literal['tool_use'] = 'tool_use'
    id: str
    name: str
    input: Dict[str, Any]

class ToolResultContent(BaseModel):
    type: Literal['tool_result'] = 'tool_result'
    tool_use_id: str
    content: Union[str, List['ContentBlock']]
    is_error: Optional[bool] = None

class ThinkingContent(BaseModel):
    type: Literal['thinking'] = 'thinking'
    thinking: str
    signature: Optional[str] = None

class ErrorContent(BaseModel):
    type: Literal['error'] = 'error'
    error: Dict[str, Any]  # Simplified LLMError mapping

# Discriminator pattern for ContentBlock
ContentBlock = Union[
    TextContent, 
    ImageContent, 
    ImageUrlContent, 
    ToolUseContent, 
    ToolResultContent, 
    ThinkingContent, 
    ErrorContent
]

# Resolve forward reference for nested ContentBlock in ToolResultContent
ToolResultContent.model_rebuild()
