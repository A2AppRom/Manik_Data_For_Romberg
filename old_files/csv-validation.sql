DROP POLICY IF EXISTS "anon can insert samples" ON public.samples;

DROP POLICY IF EXISTS "allow anon insert" ON public.samples;

CREATE POLICY "anon constrained insert"
  ON public.samples
  FOR INSERT
  TO anon
  WITH CHECK (

    sample_id          IS NOT NULL
    AND subject_session_id IS NOT NULL
    AND timestamp          IS NOT NULL
    AND extracted_features IS NOT NULL
    AND prediction_output  IS NOT NULL
    AND storage_path       IS NOT NULL


    AND prediction_output IN ('open', 'closed')

    AND extracted_features ?& ARRAY[
      'mean', 'median', 'std', 'skewness', 'kurtosis', 'path_length'
    ]

    AND jsonb_typeof(extracted_features -> 'mean')        = 'number'
    AND jsonb_typeof(extracted_features -> 'median')      = 'number'
    AND jsonb_typeof(extracted_features -> 'std')         = 'number'
    AND jsonb_typeof(extracted_features -> 'skewness')    = 'number'
    AND jsonb_typeof(extracted_features -> 'kurtosis')    = 'number'
    AND jsonb_typeof(extracted_features -> 'path_length') = 'number'

    AND (confidence IS NULL OR (confidence >= 0 AND confidence <= 1))
    AND (duration_seconds IS NULL OR (duration_seconds BETWEEN 5 AND 600))
    AND (sample_rate IS NULL OR (sample_rate BETWEEN 20 AND 500))
    AND (n_samples IS NULL OR (n_samples BETWEEN 100 AND 500000))

    AND model_version IN ('romberg_v1', 'romberg_v2')
  );
