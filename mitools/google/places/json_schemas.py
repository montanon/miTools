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

NEWPLACE_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "types": {"type": "array", "items": {"type": "string"}},
        "formattedAddress": {"type": "string"},
        "addressComponents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "long_name": {"type": "string"},
                    "short_name": {"type": "string"},
                    "types": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "plusCode": {
            "type": "object",
            "properties": {
                "globalCode": {"type": "string"},
                "compoundCode": {"type": "string"},
            },
        },
        "location": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"},
            },
            "required": ["latitude", "longitude"],
        },
        "viewport": {
            "type": "object",
            "properties": {
                "low": {
                    "type": "object",
                    "properties": {
                        "lat": {"type": "number"},
                        "lng": {"type": "number"},
                    },
                },
                "high": {
                    "type": "object",
                    "properties": {
                        "lat": {"type": "number"},
                        "lng": {"type": "number"},
                    },
                },
            },
        },
        "googleMapsUri": {"type": "string"},
        "utcOffsetMinutes": {"type": "integer"},
        "adrFormatAddress": {"type": "string"},
        "businessStatus": {"type": "string"},
        "iconMaskBaseUri": {"type": "string"},
        "iconBackgroundColor": {"type": "string"},
        "displayName": {"type": "object", "properties": {"text": {"type": "string"}}},
        "primaryTypeDisplayName": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
        },
        "primaryType": {"type": "string"},
        "shortFormattedAddress": {"type": "string"},
        "accessibilityOptions": {"type": "object"},
        "internationalPhoneNumber": {"type": "string"},
        "nationalPhoneNumber": {"type": "string"},
        "priceLevel": {"type": "string"},
        "rating": {"type": "number"},
        "userRatingCount": {"type": "integer"},
        "websiteUri": {"type": "string"},
        "currentOpeningHours": {"type": "string"},
        "currentSecondaryOpeningHours": {"type": "string"},
        "regularSecondaryOpeningHours": {"type": "string"},
        "regularOpeningHours": {"type": "string"},
    },
    "required": ["id", "displayName", "types", "location"],
}
