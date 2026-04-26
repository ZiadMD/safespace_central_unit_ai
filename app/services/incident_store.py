import time
from collections import OrderedDict
from typing import Dict, Any, Optional

class IncidentStore:
    """In-memory store for incidents waiting for backend confirmation. Follows LRU eviction limits."""
    
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def store_incident(self, incident_id: str, frame: Any, detections: list, source: str) -> None:
        """Saves an incident frame and its detections into memory."""
        if incident_id in self._store:
            # Move to end as recently accessed
            self._store.move_to_end(incident_id)
            
        self._store[incident_id] = {
            "frame": frame,
            "detections": detections,
            "source": source,
            "timestamp": time.time()
        }
        
        # Enforce LRU cap
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def get_incident(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Fetch incident data by ID."""
        if incident_id in self._store:
            self._store.move_to_end(incident_id)
            return self._store[incident_id]
        return None

    def delete_incident(self, incident_id: str) -> None:
        """Remove incident from store."""
        if incident_id in self._store:
            del self._store[incident_id]

# Singleton instance
store = IncidentStore()
