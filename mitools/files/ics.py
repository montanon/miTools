import os
from datetime import datetime
from typing import Dict, List, Optional, Set

from icalendar import Calendar, Event
from pandas import DataFrame


def read_ics_file(filepath: str) -> Calendar:
    with open(filepath, "r") as file:
        content = file.read()
    return Calendar.from_ical(content)


def extract_events(cal: Calendar) -> List[Dict[str, Optional[str]]]:
    events = []
    for component in cal.walk(name="VEVENT"):
        attendees = []
        if component.get("ATTENDEE"):
            attendees_raw = component.get("ATTENDEE")
            if isinstance(attendees_raw, list):
                attendees = [attendee for attendee in attendees_raw]
            else:
                attendees = [attendees_raw]
        event_details = {
            "summary": component.get("SUMMARY", ""),
            "description": component.get("DESCRIPTION", ""),
            "start": component.decoded("DTSTART").strftime("%Y-%m-%d %H:%M:%S")
            if component.get("DTSTART")
            else None,
            "end": component.decoded("DTEND").strftime("%Y-%m-%d %H:%M:%S")
            if component.get("DTEND")
            else None,
            "organizer": component.get("ORGANIZER", ""),
            "attendees": [str(attendee) for attendee in attendees],
            "url": component.get("URL", ""),
            "uid": component.get("UID", ""),
            "transp": component.get("TRANSP", ""),
            "status": component.get("STATUS", ""),
            "sequence": component.get("SEQUENCE", ""),
            "rrule": component.get("RRULE", ""),
            "recurrence_id": component.get("RECURRENCE-ID", ""),
            "location": component.get("LOCATION", ""),
            "last_modified": component.decoded("LAST-MODIFIED").strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            if component.get("LAST-MODIFIED")
            else None,
            "exdate": component.get("EXDATE", ""),
            "dtstamp": component.decoded("DTSTAMP").strftime("%Y-%m-%d %H:%M:%S")
            if component.get("DTSTAMP")
            else None,
            "created": component.decoded("CREATED").strftime("%Y-%m-%d %H:%M:%S")
            if component.get("CREATED")
            else None,
            "class": component.get("CLASS", ""),
            "attach": component.get("ATTACH", ""),
        }
        events.append(event_details)
    return events


def count_events_by_date(events: List[Dict[str, Optional[str]]]) -> Dict[str, int]:
    event_count = {}
    for event in events:
        if event["start"]:
            date_str = event["start"].split(" ")[0]  # Extract date part
            if date_str in event_count:
                event_count[date_str] += 1
            else:
                event_count[date_str] = 1
    return event_count


def get_unique_organizers(events: List[Dict[str, Optional[str]]]) -> Set[str]:
    organizers = set()
    for event in events:
        if event["organizer"]:
            organizers.add(event["organizer"])
    return organizers


def get_unique_attendees(events: List[Dict[str, Optional[str]]]) -> Set[str]:
    attendees = set()
    for event in events:
        for attendee in event["attendees"]:
            attendees.add(attendee)
    return attendees


def convert_to_dataframe(events: List[Dict[str, Optional[str]]]) -> DataFrame:
    return DataFrame(events)


def get_events_between_dates(
    events: List[Dict[str, Optional[str]]], start_date: datetime, end_date: datetime
) -> List[Dict[str, Optional[str]]]:
    filtered_events = []
    for event in events:
        if event["start"]:
            event_start = datetime.strptime(event["start"], "%Y-%m-%d %H:%M:%S")
            if start_date <= event_start <= end_date:
                filtered_events.append(event)
    return filtered_events


def format_event_for_display(event: Dict[str, Optional[str]]) -> str:
    return (
        f"Summary: {event['summary']}\n"
        f"Description: {event['description']}\n"
        f"Start: {event['start']}\n"
        f"End: {event['end']}\n"
        f"Organizer: {event['organizer']}\n"
        f"Attendees: {', '.join(event['attendees'])}\n"
    )


if __name__ == "__main__":
    filepath = "example.ics"
    if os.path.exists(filepath):
        calendar = read_ics_file(filepath)
        events = extract_events(calendar)
        event_count_by_date = count_events_by_date(events)
        for date, count in event_count_by_date.items():
            print(f"{date}: {count} events")
        df = convert_to_dataframe(events)
        print(df)
        organizers = get_unique_organizers(events)
        print(f"Unique organizers: {organizers}")
        attendees = get_unique_attendees(events)
        print(f"Unique attendees: {attendees}")
        if events:
            print(format_event_for_display(events[0]))
    else:
        print(f"File {filepath} not found.")
