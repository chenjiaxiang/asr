SCHEDULER_REGISTRY = dict()
SCHEDULER_DATACLASS_REGISTRY = dict()


def register_scheduler(name: str, dataclass = None):
    def register_scheduler_cls(cls):
        if name in SCHEDULER_REGISTRY:
            raise ValueError(f"Cannot register duplicate scheduler ({name})")

        SCHEDULER_REGISTRY[name] = cls
    
        cls.__dataclass__ = dataclass
        if dataclass is not None:
            if name in SCHEDULER_DATACLASS_REGISTRY:
                raise ValueError(f"Cannot register duplicate scheduler ({name})")
            SCHEDULER_DATACLASS_REGISTRY[name] = dataclass

        return cls

    return register_scheduler_cls
                