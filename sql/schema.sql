-- RouteWise AI - PostgreSQL schema for trip records and road-segment features
-- Optional: use when DATABASE_URL is set to persist data

-- Trip records (e.g. from 100K+ historical or synthetic trips)
CREATE TABLE IF NOT EXISTS trips (
    id BIGSERIAL PRIMARY KEY,
    origin_node BIGINT NOT NULL,
    dest_node BIGINT NOT NULL,
    travel_time_min DOUBLE PRECISION NOT NULL,
    length_m DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trips_origin ON trips(origin_node);
CREATE INDEX IF NOT EXISTS idx_trips_dest ON trips(dest_node);
CREATE INDEX IF NOT EXISTS idx_trips_created ON trips(created_at);

-- Road segments (for joining with graph edges)
CREATE TABLE IF NOT EXISTS segments (
    u BIGINT NOT NULL,
    v BIGINT NOT NULL,
    k INT NOT NULL DEFAULT 0,
    length_m DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (u, v, k)
);

-- Segment-level congestion features (Pandas/NetworkX output)
CREATE TABLE IF NOT EXISTS segment_features (
    u BIGINT NOT NULL,
    v BIGINT NOT NULL,
    k INT NOT NULL DEFAULT 0,
    length_m DOUBLE PRECISION NOT NULL,
    trip_count INT NOT NULL DEFAULT 0,
    mean_travel_time_min DOUBLE PRECISION,
    std_travel_time_min DOUBLE PRECISION,
    free_flow_time_min DOUBLE PRECISION,
    congestion_ratio DOUBLE PRECISION,
    out_degree_u INT,
    in_degree_v INT,
    PRIMARY KEY (u, v, k)
);

-- Route query log (optional, for analytics)
CREATE TABLE IF NOT EXISTS route_queries (
    id BIGSERIAL PRIMARY KEY,
    query_text TEXT,
    origin_place TEXT,
    dest_place TEXT,
    travel_time_min DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
