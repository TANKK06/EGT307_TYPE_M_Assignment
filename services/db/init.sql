-- Create the main table for storing prediction logs (only creates it if it doesn't exist)
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,                            -- Auto-incrementing unique ID for each log row
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),     -- Timestamp (timezone-aware), defaults to current time

    -- Raw payloads (always keep these for debugging/auditing/replay)
    request_json JSONB NOT NULL,                       -- Full input payload sent to inference
    response_json JSONB NOT NULL,                      -- Full response returned by inference

    -- Input features (stored as columns for fast filtering/analytics)
    type TEXT,                                         -- Categorical: product/machine type
    air_temperature_c DOUBLE PRECISION,                -- Air temperature in Celsius
    process_temperature_c DOUBLE PRECISION,            -- Process temperature in Celsius
    rotational_speed_rpm DOUBLE PRECISION,             -- Rotational speed in RPM
    torque_nm DOUBLE PRECISION,                        -- Torque in Newton-meters
    tool_wear_min DOUBLE PRECISION,                    -- Tool wear in minutes

    -- Prediction outputs (stored as columns for fast dashboards)
    failure_probability DOUBLE PRECISION,              -- Model probability of failure (0.0 to 1.0)
    predicted_label INTEGER,                           -- Predicted class label (0/1)
    risk_level TEXT,                                   -- Human-friendly risk band (e.g., low/med/high)
    model_path TEXT                                    -- Which model file produced this prediction
);

-- Helpful indexes (speed up common queries in the dashboard)
CREATE INDEX IF NOT EXISTS idx_predictions_created_at
  ON predictions (created_at DESC);                    -- Fast "latest logs" queries

CREATE INDEX IF NOT EXISTS idx_predictions_label
  ON predictions (predicted_label);                    -- Fast filtering by predicted class (0/1)

CREATE INDEX IF NOT EXISTS idx_predictions_risk
  ON predictions (risk_level);                         -- Fast filtering by risk level

CREATE INDEX IF NOT EXISTS idx_predictions_type
  ON predictions (type);                               -- Fast filtering/grouping by Type
