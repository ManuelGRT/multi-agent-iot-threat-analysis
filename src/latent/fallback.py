# src/latent/fallback.py
def routing_from_mapping_confidence(mapping_confidence: float) -> str:
    if mapping_confidence >= 0.80:
        return "full"
    if mapping_confidence >= 0.50:
        return "hybrid_review"
    return "latent_only"

def nearest_prototype_decision(similarity: float, high=0.85, medium=0.70) -> tuple[str, bool]:
    if similarity >= high:
        return ("accept", False)
    if similarity >= medium:
        return ("judge", False)
    return ("judge", True)  # abstention recommended
