from pydantic import BaseModel

class DetectionConfig(BaseModel):
    record_duration: int
    enable_video: bool
    enable_screenshot: bool
    external_endpoint: str