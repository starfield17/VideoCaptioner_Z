"""Canonical SubtitleTrack JSON codec facade."""

from captioner.adapters.persistence.domain_codecs import decode_track, encode_track
from captioner.core.domain.subtitle import SubtitleTrack


def serialize(track: SubtitleTrack) -> bytes:
    return encode_track(track)


def parse(data: bytes):
    return decode_track(data)


serialize_bytes = serialize
