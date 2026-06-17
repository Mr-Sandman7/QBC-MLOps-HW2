from __future__ import annotations

from typing import Iterable, List

import pandas as pd
from fastapi import HTTPException, status

from . import config
from .schemas import ListingFeatures, PredictionResponse

# Mapping from HW03 field names to HW02 model feature names
FIELD_MAPPING = {
    'host_is_superhost': 'is_superhost',
    'host_listing_count': 'count',
    'avg_comment_len_before_cutoff': 'name',
}

# HW02 model actual features
HW02_FEATURE_COLUMNS = [
    'room_type',
    'property_type',
    'accommodates',
    'bedrooms',
    'minimum_nights',
    'maximum_nights',
    'instant_bookable',
    'bathrooms',
    'is_superhost',
    'count',
    'name',
    'total_reviews_before_cutoff',
    'unique_reviewers_before_cutoff',
    'days_since_last_review',
    'available_days_last_90d',
    'available_rate_last_90d',
    'avg_minimum_nights_calendar_last_90d',
    'avg_maximum_nights_calendar_last_90d',
    'avg_minimum_nights_calendar_last_30d',
    'avg_maximum_nights_calendar_last_30d',
]


def records_to_dataframe(records: Iterable[ListingFeatures]) -> pd.DataFrame:
    """Convert validated API payloads into the exact DataFrame expected by the model."""
    rows = [record.model_dump() for record in records]
    df = pd.DataFrame(rows)

    # TODO 1: reject forbidden leakage fields
    forbidden_present = [col for col in config.FORBIDDEN_FIELDS if col in df.columns]
    if forbidden_present:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Forbidden leakage fields detected",
                "forbidden_fields": forbidden_present
            },
        )

    # TODO 2: Map HW03 field names to HW02 model feature names
    df = df.rename(columns=FIELD_MAPPING)

    # TODO 3: check missing fields against HW02 model features
    missing_cols = [c for c in HW02_FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Missing required feature fields",
                "missing_fields": missing_cols
            },
        )

    # TODO 4: return df with HW02 feature columns in correct order
    return df[HW02_FEATURE_COLUMNS]


def predict_records(model, records: List[ListingFeatures]) -> List[PredictionResponse]:
    """Run model prediction and return API responses."""
    X = records_to_dataframe(records)

    predictions = []

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        positive_proba = proba[:, 1]
        binary_predictions = (positive_proba >= config.PREDICTION_THRESHOLD).astype(int)

        for pred, prob in zip(binary_predictions, positive_proba):
            label = config.POSITIVE_LABEL if pred == 1 else config.NEGATIVE_LABEL
            predictions.append(
                PredictionResponse(
                    prediction=int(pred),
                    prediction_label=label,
                    probability=float(prob),
                    threshold=config.PREDICTION_THRESHOLD,
                )
            )
    else:
        binary_predictions = model.predict(X)

        for pred in binary_predictions:
            label = config.POSITIVE_LABEL if pred == 1 else config.NEGATIVE_LABEL
            predictions.append(
                PredictionResponse(
                    prediction=int(pred),
                    prediction_label=label,
                    probability=None,
                    threshold=config.PREDICTION_THRESHOLD,
                )
            )

    return predictions
