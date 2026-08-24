-- src/latent/storage.sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE event_records (
    event_id TEXT PRIMARY KEY,
    dataset TEXT NOT NULL,
    modality TEXT NOT NULL,
    ts TIMESTAMPTZ NULL,
    canonical_json JSONB NOT NULL,
    semantic_text TEXT NOT NULL,
    mapping_confidence REAL NOT NULL,
    attack_family TEXT NULL,
    attack_subtype TEXT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dim INT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    provenance JSONB NOT NULL,
    latent_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX event_records_dataset_idx ON event_records (dataset);
CREATE INDEX event_records_modality_idx ON event_records (modality);
CREATE INDEX event_records_mapping_conf_idx ON event_records (mapping_confidence);
CREATE INDEX event_records_embedding_hnsw
ON event_records
USING hnsw (embedding vector_cosine_ops);
