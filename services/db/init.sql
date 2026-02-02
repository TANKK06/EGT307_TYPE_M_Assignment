-- services/db/init.sql

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Raw payloads (always keep these)
    request_json JSONB NOT NULL,
    response_json JSONB NOT NULL,

    -- Input features (easy querying + dashboards)
    type TEXT,
    air_temperature_c DOUBLE PRECISION,
    process_temperature_c DOUBLE PRECISION,
    rotational_speed_rpm DOUBLE PRECISION,
    torque_nm DOUBLE PRECISION,
    tool_wear_min DOUBLE PRECISION,

    -- Prediction outputs (easy querying)
    failure_probability DOUBLE PRECISION,
    predicted_label INTEGER,
    risk_level TEXT,
    model_path TEXT
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_label ON predictions (predicted_label);
CREATE INDEX IF NOT EXISTS idx_predictions_risk ON predictions (risk_level);
CREATE INDEX IF NOT EXISTS idx_predictions_type ON predictions (type);
