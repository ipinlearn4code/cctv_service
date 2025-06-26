from pydantic import BaseModel
from uuid import UUID

class CCTVCreate(BaseModel):
    name: str
    ip_address: str
    location: str
    status: str

class CCTVUpdate(BaseModel):
    name: str | None = None
    ip_address: str | None = None
    location: str | None = None
    status: str | None = None

class CCTVResponse(BaseModel):
    id: str
    name: str
    ip_address: str
    location: str
    status: str