"""Esquemas de salida estructurada del asistente regulatorio."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, model_validator


class Fuente(BaseModel):
    entidad: str = Field(
        description="Entidad emisora, por ejemplo SUGESE, CONASSIF o INS"
    )
    norma: str = Field(
        description="Número o nombre de la norma o circular, tal como aparece en el documento"
    )
    articulo: str | None = Field(
        default=None, description="Artículo citado, si el documento lo indica"
    )
    vigente_desde: date | None = Field(
        default=None,
        description="Fecha de entrada en vigencia (AAAA-MM-DD), solo si el documento la indica",
    )


class RespuestaRegulatoria(BaseModel):
    respuesta: str = Field(
        description="Respuesta en español, en texto plano y sin Markdown"
    )
    sustentada: bool = Field(
        description="true solo si la respuesta se basa en los documentos incluidos en la consulta"
    )
    fuentes: list[Fuente] = Field(
        default_factory=list,
        description="Normas que respaldan la respuesta; lista vacía si no hay sustento",
    )

    @model_validator(mode="after")
    def validar_coherencia(self) -> RespuestaRegulatoria:
        # Regla de negocio en código: no confiamos solo en que el modelo la cumpla.
        if self.sustentada and not self.fuentes:
            raise ValueError("una respuesta sustentada debe citar al menos una fuente")
        if not self.sustentada and self.fuentes:
            raise ValueError("una respuesta sin sustento no puede citar fuentes")
        return self
