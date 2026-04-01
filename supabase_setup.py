from supabase import create_client
import pandas as pd
import math
 
# --- CONFIGURATION ---
SUPABASE_URL = "https://xmxyschvrqcqhqznjdgu.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhteHlzY2h2cnFjcWhxem5qZGd1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzQ0NzEzNTgsImV4cCI6MjA5MDA0NzM1OH0.0odYckxFFkZhBaSxA6HB_p7VU1iUCGOeKmlbaR1EDZ4"
TABLE_NAME   = "samples"
 
client = create_client(SUPABASE_URL, SUPABASE_KEY)
 
# --- HELPER FUNCTIONS ---
def extract_features(df):
    def extract_features(df):
    """Compute required features from accelerometer CSV"""
    x, y, z = df['x'], df['y'], df['z']
    mag = (x**2 + y**2 + z**2).apply(math.sqrt)
    t   = df['seconds_elapsed']
 
    # RMS sway = root mean square of magnitude
    rms_sway = float(mag.pow(2).mean() ** 0.5)
 
    # Std dev per axis = directional variability
    std_x = float(x.std())
    std_y = float(y.std())
    std_z = float(z.std())
 
    # Mean jerk = rate of change of magnitude over time
    dt        = t.diff().fillna(0)
    jerk      = mag.diff().abs() / dt.replace(0, float('nan'))
    mean_jerk = float(jerk.mean())
 
    # Path length = total displacement in X-Y plane
    path_length = float(
        ((x.diff()**2 + y.diff()**2) ** 0.5).sum()
    )
 
    # Sway mean & peak
    sway_mean = float(mag.mean())
    sway_peak = float(mag.max())
 
    return {
        "rms_sway":    rms_sway,
        "std_x":       std_x,
        "std_y":       std_y,
        "std_z":       std_z,
        "mean_jerk":   mean_jerk,
        "path_length": path_length,
        "sway_mean":   sway_mean,
        "sway_peak":   sway_peak,
    }

# -- Save to storage bucket --
def save_sample(csv_path: str, label: str, subject_session_id: str, prediction_output: str = None):
    """
    Load a CSV, compute features, and insert one row into Supabase.
 
    Table columns:
      sample_id          = unique UUID for this recording
      timestamp          = UTC time of upload
      subject_session_id = shared UUID connecting a control + blindfolded pair
      label              = 'control' or 'blindfolded'
      extracted_features = computed metrics (stored as JSON)
      prediction_output  = 'stable', 'impaired', or None until model is finalized
    """
    df = pd.read_csv(csv_path)
 
    row = {
        "sample_id":          str(uuid.uuid4()),
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "subject_session_id": subject_session_id,
        "label":              label,
        "extracted_features": extract_features(df),
        "prediction_output":  prediction_output,
    }
 
    result = client.table(TABLE_NAME).insert(row).execute()
    print(f"Inserted | sample_id={row['sample_id']} | label={label} | session={subject_session_id}")
    return result
 
 
# --- USAGE ---
if __name__ == "__main__":
    import uuid
    session_id = str(uuid.uuid4())   # one shared ID per pair of recordings
 
    save_sample("eyes_open.csv",   label="control",     subject_session_id=session_id)
    save_sample("eyes_closed.csv", label="blindfolded", subject_session_id=session_id)
