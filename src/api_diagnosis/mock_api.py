"""Small, deterministic FastAPI service used as the V0 API fixture.

The service deliberately has one state-free business endpoint.  It is a safe
target for later replay tooling and makes the first Ground Truth cases
repeatable without depending on a real third-party provider.
"""

from __future__ import annotations

from typing import List, Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field


API_NAME = "weather-mock"
API_VERSION: Literal["v2"] = "v2"
ENDPOINT_PATH = "/weather/forecast"


class ForecastRequest(BaseModel):
    """The complete request body for the current v2 contract."""

    model_config = ConfigDict(extra="forbid")

    city: str = Field(..., min_length=1, max_length=100)
    units: Literal["metric", "imperial"]
    days: int = Field(..., ge=1, le=7)



class ForecastDay(BaseModel):
    day: int
    temperature: float
    condition: Literal["clear"]


class ForecastResponse(BaseModel):
    api_version: Literal["v2"]
    city: str
    units: Literal["metric", "imperial"]
    days: int
    forecast: List[ForecastDay]


app = FastAPI(
    title="Weather Mock API",
    version=API_VERSION,
    description=(
        "A state-free JSON API fixture for deterministic API integration "
        "diagnosis. Only POST /weather/forecast is a business endpoint."
    ),
)


@app.post(
    ENDPOINT_PATH,
    response_model=ForecastResponse,
    status_code=200,
    summary="Return a deterministic weather forecast",
    responses={
        400: {
            "description": "The requested API version is not supported",
            "content": {
                "application/json": {
                    "example": {
                        "detail": {
                            "code": "API_VERSION_UNSUPPORTED",
                            "requested_version": "v1",
                            "current_version": API_VERSION,
                            "changelog_id": "weather-v2-001",
                        }
                    }
                }
            },
        },
        404: {"description": "Unknown path (FastAPI application-level response)"},
        405: {"description": "Method is not allowed for this path"},
    },
)
async def forecast(
    payload: ForecastRequest,
    x_api_version: str = Header(..., alias="X-API-Version"),
) -> ForecastResponse:
    """Return a fixed response after checking the declared API version.

    Body parsing and validation are intentionally delegated to FastAPI and
    Pydantic.  Invalid JSON or a body that violates ``ForecastRequest`` is a
    deterministic 422 response.  Version validation happens after the body
    has been parsed so a valid v2 body with a stale version produces the
    dedicated 400 version error.
    """

    if x_api_version != API_VERSION:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "API_VERSION_UNSUPPORTED",
                "message": "Use the current API version for this endpoint.",
                "requested_version": x_api_version,
                "current_version": API_VERSION,
                "changelog_id": "weather-v2-001",
            },
        )

    base_temperature = 24.0 if payload.units == "metric" else 75.2
    forecast_rows = [
        ForecastDay(
            day=day,
            temperature=round(base_temperature + (day - 1) * 0.5, 1),
            condition="clear",
        )
        for day in range(1, payload.days + 1)
    ]

    return ForecastResponse(
        api_version=API_VERSION,
        city=payload.city,
        units=payload.units,
        days=payload.days,
        forecast=forecast_rows,
    )


__all__ = [
    "API_NAME",
    "API_VERSION",
    "ENDPOINT_PATH",
    "ForecastDay",
    "ForecastRequest",
    "ForecastResponse",
    "app",
]
