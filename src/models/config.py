"""Configuration models for agents and pipeline."""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentConfig(BaseModel):
    """Configuration for solver and verifier agents."""

    role: Literal["solver", "verifier"]
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    prompt_template: str
    system_prompt: str = ""


class JudgeConfig(BaseModel):
    """Configuration for judge agent."""

    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    prompt_template: str
    system_prompt: str = ""
    scoring_rubric: str


class PipelineConfig(BaseSettings):
    """Pipeline configuration with YAML/JSON loading support."""

    model_config = SettingsConfigDict(
        yaml_file="config.yaml",
        json_file="config.json",
        env_file=".env",
        extra="ignore",
    )

    solver: AgentConfig
    verifier: AgentConfig
    judge: JudgeConfig
    mode: Literal["debate", "baseline"] = "debate"
    max_iterations: int = Field(default=5, ge=1, le=20)
    trajectory_output_dir: str = "trajectories"
    config_version: str

    def config_hash(self) -> str:
        """Generate deterministic hash of configuration.

        Returns:
            First 8 characters of SHA256 hash of config JSON.
        """
        config_dict = self.model_dump()
        config_json = json.dumps(config_dict, sort_keys=True)
        hash_obj = hashlib.sha256(config_json.encode("utf-8"))
        return hash_obj.hexdigest()[:8]
