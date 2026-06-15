from pydantic import BaseModel, Field
from typing import List
class PredictRequest(BaseModel):
    features: List[float] = Field(..., min_length=10, max_length=10, description="Exactly 10 numeric features")
    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.5, -1.2, 0.8, 1.1, -0.3, 0.9, -0.7, 0.4, 1.5, -0.6]
            }
        }
class PredictResponse(BaseModel):
    prediction: int
    confidence: float
    model_version: str
