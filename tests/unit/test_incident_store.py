import pytest
from app.services.incident_store import IncidentStore

def test_store_and_retrieve():
    store = IncidentStore(max_size=3)
    store.store_incident("inc1", "frame1", [{"class_id": 0}], "node")
    
    inc = store.get_incident("inc1")
    assert inc is not None
    assert inc["frame"] == "frame1"
    assert inc["source"] == "node"

def test_lru_eviction():
    store = IncidentStore(max_size=2)
    store.store_incident("inc1", "frame1", [], "node")
    store.store_incident("inc2", "frame2", [], "node")
    store.store_incident("inc3", "frame3", [], "node")
    
    # inc1 should be evicted
    assert store.get_incident("inc1") is None
    assert store.get_incident("inc2") is not None
    assert store.get_incident("inc3") is not None

def test_delete_incident():
    store = IncidentStore(max_size=5)
    store.store_incident("inc1", "frame", [], "node")
    store.delete_incident("inc1")
    assert store.get_incident("inc1") is None
