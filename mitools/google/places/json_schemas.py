PLACE_SCHEMA = {
    "type": "object",
    "properties": {
        "place_id": {"type": "string"},
        "name": {"type": "string"},
        "geometry": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "object",
                    "properties": {
                        "lat": {"type": "number"},
                        "lng": {"type": "number"},
                    },
                    "required": ["lat", "lng"],
                }
            },
        },
        "types": {"type": "array", "items": {"type": "string"}},
        "price_level": {"type": ["integer", "null"]},
        "rating": {"type": ["number", "null"]},
        "user_ratings_total": {"type": ["integer", "null"]},
        "vicinity": {"type": ["string", "null"]},
        "permanently_closed": {"type": ["boolean", "null"]},
    },
    "required": ["place_id", "name", "geometry", "types"],
}
