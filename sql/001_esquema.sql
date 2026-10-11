-- Esquema de normativa versionada. Idempotente: se puede ejecutar varias veces.

CREATE EXTENSION IF NOT EXISTS vector;
-- btree_gist permite combinar "=" sobre bigint y "&&" sobre rangos en una misma
-- restricción EXCLUDE (ver version_documento).
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- Una norma, independientemente de sus reformas.
CREATE TABLE IF NOT EXISTS documento (
    id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    entidad    text NOT NULL,
    codigo     text NOT NULL,
    titulo     text NOT NULL,
    creado_en  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (entidad, codigo)
);

-- Cada reforma es una versión nueva con su propio período de vigencia.
CREATE TABLE IF NOT EXISTS version_documento (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    documento_id    bigint NOT NULL REFERENCES documento (id) ON DELETE CASCADE,
    numero          integer NOT NULL,
    vigente_desde   date NOT NULL,
    vigente_hasta   date,  -- NULL: vigente sin fecha de fin conocida
    fuente_url      text NOT NULL,
    hash_contenido  text NOT NULL,
    creado_en       timestamptz NOT NULL DEFAULT now(),
    UNIQUE (documento_id, numero),
    CHECK (vigente_hasta IS NULL OR vigente_hasta > vigente_desde),
    -- Dos versiones del mismo documento no pueden estar vigentes el mismo día.
    -- '[)' incluye vigente_desde y excluye vigente_hasta.
    CONSTRAINT version_sin_traslape EXCLUDE USING gist (
        documento_id WITH =,
        daterange(vigente_desde, vigente_hasta, '[)') WITH &&
    )
);

-- Fragmentos de una versión, con su vector y el modelo que lo generó.
CREATE TABLE IF NOT EXISTS fragmento (
    id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    version_id        bigint NOT NULL REFERENCES version_documento (id) ON DELETE CASCADE,
    orden             integer NOT NULL,
    articulo          text,
    texto             text NOT NULL,
    tokens            integer NOT NULL,
    modelo_embedding  text NOT NULL,
    embedding         vector(1024) NOT NULL,
    UNIQUE (version_id, orden)
);

CREATE INDEX IF NOT EXISTS fragmento_version_idx ON fragmento (version_id);
