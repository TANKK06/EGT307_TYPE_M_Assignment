-- services/db/init.sql
-- This SQL file runs automatically when the Postgres container starts for the first time
-- (mounted into /docker-entrypoint-initdb.d/).
-- It creates the table used to store prediction logs and adds indexes for faster queries.

-- Create a table to store every prediction request + response and some extracted fields
CREATE TABLE IF NOT EXISTS predictions (
    -- Unique row ID for each prediction log entry
    id SERIAL PRIMARY KEY,

    -- Timestamp when the log was created (timezone-aware)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- -----------------------------
    -- Raw payloads (keep full traceability)
    -- -----------------------------
    -- Original request JSON sent to inference/logger
    request_json JSONB NOT NULL,

    -- Full response JSON returned by inference
    response_json JSONB NOT NULL,

    -- -----------------------------
    -- Extracted input features (for easy filtering/analysis in the dashboard)
    -- -----------------------------
    type TEXT,                                   -- machine type (L/M/H)
    air_temperature_c DOUBLE PRECISION,          -- air temp in °C
    process_temperature_c DOUBLE PRECISION,      -- process temp in °C
    rotational_speed_rpm DOUBLE PRECISION,       -- RPM
    torque_nm DOUBLE PRECISION,                  -- torque in Nm
    tool_wear_min DOUBLE PRECISION,              -- tool wear in minutes

    -- -----------------------------
    -- Prediction outputs (for reporting + monitoring model performance)
    -- -----------------------------
    failure_probability DOUBLE PRECISION,        -- probability of failure (0..1)
    predicted_label INTEGER,                     -- predicted class (0/1)
    risk_level TEXT,                             -- e.g., low/medium/high
    model_path TEXT                              -- which model version/file was used
);

-- -----------------------------
-- Indexes (speed up common queries)
-- -----------------------------
-- Used by dashboard "recent logs" query sorting by newest first
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at DESC);

-- Used when filtering by predicted_label (e.g., show failures only)
CREATE INDEX IF NOT EXISTS idx_predictions_label ON predictions (predicted_label);

-- Used when filtering by risk_level (e.g., show only "high" risk)
CREATE INDEX IF NOT EXISTS idx_predictions_risk ON predictions (risk_level);

-- Used when filtering by machine type (L/M/H)
CREATE INDEX IF NOT EXISTS idx_predictions_type ON predictions (type);