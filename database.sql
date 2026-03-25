create table samples (
  sample_id          uuid primary key,
  subject_session_id uuid not null,
  timestamp          timestamptz not null,
  label              text not null,
  extracted_features jsonb not null,
  prediction_output  text not null
);
