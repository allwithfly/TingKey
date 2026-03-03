from __future__ import annotations

import asyncio
import base64
import json
from queue import Empty
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from desktop_service.config_models import AppConfig
from desktop_service.config_store import ConfigStore
from desktop_service.final_asr import FinalAsrConfig, FinalAsrRunner
from desktop_service.session_manager import SessionManager


class StartSessionRequest(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2
    language: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkRequest(BaseModel):
    audio_base64: str


class PartialRequest(BaseModel):
    text: str


class StopSessionRequest(BaseModel):
    run_final_asr: bool = True
    final_text: str | None = None


class CommitRequest(BaseModel):
    text: str | None = None


class PatchConfigRequest(BaseModel):
    values: dict[str, Any]


def _build_final_asr_runner(config: AppConfig) -> FinalAsrRunner:
    model = config.model
    cfg = FinalAsrConfig(
        enabled=model.enable_final_asr,
        model_dir=model.asr_model_dir,
        model_size=model.asr_model_size,
        device="auto",
        backend=model.asr_backend,
        dtype=model.asr_dtype,
        max_new_tokens=model.asr_max_new_tokens,
        max_inference_batch_size=model.asr_max_inference_batch_size,
        attn_implementation=model.asr_attn_implementation,
        enable_tf32=model.asr_enable_tf32,
        cudnn_benchmark=model.asr_cudnn_benchmark,
    )
    return FinalAsrRunner(cfg)


def create_app(
    config_path: str | Path = "desktop_service_config.json",
    audio_root: str | Path = "recordings",
) -> FastAPI:
    app = FastAPI(title="Desktop Voice Input Service", version="0.1.0")

    cfg_store = ConfigStore(config_path)
    session_manager = SessionManager(audio_root=audio_root)
    final_asr = _build_final_asr_runner(cfg_store.get())

    app.state.config_store = cfg_store
    app.state.session_manager = session_manager
    app.state.final_asr = final_asr

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "config_path": str(cfg_store.config_path),
            "audio_root": str(session_manager.audio_root),
            "final_asr_enabled": cfg_store.get().model.enable_final_asr,
        }

    @app.get("/v1/config")
    def get_config() -> dict[str, Any]:
        return cfg_store.get_dict()

    @app.put("/v1/config")
    def set_config(config: AppConfig) -> dict[str, Any]:
        cfg_store.set(config)
        app.state.final_asr = _build_final_asr_runner(config)
        return cfg_store.get_dict()

    @app.patch("/v1/config/{section}")
    def patch_config(section: str, req: PatchConfigRequest) -> dict[str, Any]:
        try:
            cfg = cfg_store.update_section(section, req.values)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        app.state.final_asr = _build_final_asr_runner(cfg)
        return cfg_store.get_dict()

    @app.get("/v1/sessions")
    def list_sessions() -> list[dict[str, Any]]:
        return [s.to_public_dict() for s in session_manager.list_sessions()]

    @app.post("/v1/sessions/start")
    def start_session(req: StartSessionRequest) -> dict[str, Any]:
        session = session_manager.start_session(
            sample_rate=req.sample_rate,
            channels=req.channels,
            sample_width=req.sample_width,
            language=req.language,
            metadata=req.metadata,
        )
        return session.to_public_dict()

    @app.get("/v1/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        try:
            session = session_manager.get_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return session.to_public_dict()

    @app.post("/v1/sessions/{session_id}/chunk")
    def add_chunk(session_id: str, req: ChunkRequest) -> dict[str, Any]:
        try:
            payload = base64.b64decode(req.audio_base64)
            session = session_manager.add_chunk(session_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return session.to_public_dict()

    @app.post("/v1/sessions/{session_id}/partial")
    def add_partial(session_id: str, req: PartialRequest) -> dict[str, Any]:
        try:
            session = session_manager.publish_partial(session_id, req.text)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return session.to_public_dict()

    async def _run_final_asr(session_id: str) -> None:
        try:
            runner: FinalAsrRunner = app.state.final_asr
            if not runner.cfg.enabled:
                return
            session_manager.mark_processing(session_id)
            session = session_manager.get_session(session_id)
            result = await asyncio.to_thread(
                runner.transcribe_file,
                session.wav_path,
                session.language,
            )
            session_manager.mark_completed(
                session_id,
                final_text=result["text"],
                final_model_time_s=result["model_inference_time_s"],
            )
        except Exception as exc:
            session_manager.mark_error(session_id, str(exc))

    @app.post("/v1/sessions/{session_id}/stop")
    async def stop_session(session_id: str, req: StopSessionRequest) -> dict[str, Any]:
        try:
            session = session_manager.stop_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if req.final_text is not None:
            session = session_manager.mark_completed(
                session_id,
                final_text=req.final_text,
                final_model_time_s=None,
            )
        elif req.run_final_asr:
            runner: FinalAsrRunner = app.state.final_asr
            if runner.cfg.enabled:
                asyncio.create_task(_run_final_asr(session_id))
        return session.to_public_dict()

    @app.post("/v1/sessions/{session_id}/commit")
    def commit_text(session_id: str, req: CommitRequest) -> dict[str, Any]:
        try:
            session = session_manager.get_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        text = req.text if req.text is not None else session.final_text
        return {
            "session_id": session_id,
            "committed_text": text or "",
            "mode": "placeholder",
        }

    @app.get("/v1/sessions/{session_id}/events")
    async def stream_events(session_id: str) -> StreamingResponse:
        try:
            subscriber_id, q = session_manager.subscribe(session_id, replay_history=True)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        async def event_generator():
            try:
                while True:
                    try:
                        event = await asyncio.to_thread(q.get, True, 10.0)
                    except Empty:
                        yield ": keepalive\n\n"
                        continue
                    payload = json.dumps(event, ensure_ascii=False)
                    event_name = str(event.get("type", "message"))
                    yield f"event: {event_name}\ndata: {payload}\n\n"
            finally:
                session_manager.unsubscribe(session_id, subscriber_id)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app
