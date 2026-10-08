-- RouteWise AI PostgreSQL schema (optional; parquet is the default store)

CREATE TABLE IF NOT EXISTS trips (
    id BIGSERIAL PRIMARY KEY,
    origin_node BIGINT NOT NULL,
    dest_node BIGINT NOT NULL,
    travel_time_min DOUBLE PRECISION NOT NULL,
    length_m DOUBLE PRECISION NOT NULL,
    hour INT,
    dow INT,
    day_index INT,
    rain INT,
    event INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trips_origin ON trips(origin_node);
CREATE INDEX IF NOT EXISTS idx_trips_dest ON trips(dest_node);
CREATE INDEX IF NOT EXISTS idx_trips_hour ON trips(hour, dow);

CREATE TABLE IF NOT EXISTS segments (
    u BIGINT NOT NULL,
    v BIGINT NOT NULL,
    k INT NOT NULL DEFAULT 0,
    length_m DOUBLE PRECISION NOT NULL,
    highway_code INT,
    maxspeed DOUBLE PRECISION,
    PRIMARY KEY (u, v, k)
);

CREATE TABLE IF NOT EXISTS segment_features (
    u BIGINT NOT NULL,
    v BIGINT NOT NULL,
    k INT NOT NULL DEFAULT 0,
    hour INT NOT NULL,
    dow INT NOT NULL,
    length_m DOUBLE PRECISION,
    trip_count INT,
    mean_travel_time_min DOUBLE PRECISION,
    std_travel_time_min DOUBLE PRECISION,
    free_flow_time_min DOUBLE PRECISION,
    congestion_ratio DOUBLE PRECISION,
    highway_code INT,
    maxspeed DOUBLE PRECISION,
    out_degree_u INT,
    in_degree_v INT,
    rain_rate DOUBLE PRECISION,
    event_rate DOUBLE PRECISION,
    neighbor_congestion DOUBLE PRECISION,
    neighbor_travel_time DOUBLE PRECISION,
    PRIMARY KEY (u, v, k, hour, dow)
);

CREATE TABLE IF NOT EXISTS route_queries (
    id BIGSERIAL PRIMARY KEY,
    query_text TEXT,
    origin_place TEXT,
    dest_place TEXT,
    model_variant TEXT,
    travel_time_min DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
