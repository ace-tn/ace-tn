from dataclasses import dataclass


@dataclass(frozen=True)
class Bond:
    """A bond connecting two sites in the tensor network.

    Attributes:
        s1: Coordinates of the first site, e.g. (0, 0).
        s2: Coordinates of the second site, e.g. (1, 0).
        k: Bond direction (0=left, 1=up, 2=right, 3=down).
    """
    s1: tuple
    s2: tuple
    k: int

    def __post_init__(self):
        object.__setattr__(self, 's1', tuple(self.s1))
        object.__setattr__(self, 's2', tuple(self.s2))

    def __iter__(self):
        return iter((self.s1, self.s2, self.k))
