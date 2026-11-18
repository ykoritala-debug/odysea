"""ODySEA HTTP API.   uvicorn odysea.api:app

POST /plan       demo / NetCDF pipeline (synthetic unless a NetCDF is configured)
POST /plan/live  conservative routing on the latest NOAA/NSIDC observation (not a forecast)
GET  /legs       available voyage legs
GET  /health
"""
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import LEGS
from .pipeline import run_leg, run_leg_live

app = FastAPI(title="ODySEA", version="0.3")

LegId = Literal["leg1", "leg2", "leg3"]


class Req(BaseModel):
    leg_id: LegId = "leg1"
    horizon: int = Field(5, ge=1, le=7)
    members: int = Field(5, ge=3, le=10)
    p_block: float = Field(0.25, gt=0.0, lt=1.0)
    seed: int = 0


class LiveReq(BaseModel):
    leg_id: LegId = "leg1"
    p_block: float = Field(0.25, gt=0.0, lt=1.0)
    max_snap_nm: float = Field(100.0, gt=0.0, le=500.0)


@app.get("/health")
def health():
    return {"ok": True, "mode": "synthetic-demo unless a NetCDF is configured", "not_for_navigation": True}


@app.get("/legs")
def legs():
    return {k: {"name": v.name, "start": v.start, "end": v.end, "window": [v.t0, v.t1]} for k, v in LEGS.items()}


@app.post("/plan")
def plan_endpoint(r: Req):
    out = run_leg(r.leg_id, r.horizon, r.members, r.p_block, r.seed)
    return {k: out[k] for k in ("leg", "result", "forecast_skill", "data_mode")}


@app.post("/plan/live")
def plan_live_endpoint(r: LiveReq):
    try:
        out = run_leg_live(r.leg_id, p_block=r.p_block, max_snap_nm=r.max_snap_nm)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Live data file missing: {exc}")
    return {k: out[k] for k in ("leg", "result", "route_status", "data_mode", "metadata")}
