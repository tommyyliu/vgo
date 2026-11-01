"""Certify a 7.4918r locality lower bound using exact rational arithmetic.

Run with Python's standard library only:
    python diagnostics/check_locality_floor.py

All assertions use Fraction, including the Voronoi clipping and disk coverage.
Decimals in the report are for display only. See docs/research/LOCAL_CONTESTABILITY_FLOOR.md.
"""

from fractions import Fraction
from itertools import combinations


def point(x, y):
    return Fraction(str(x)), Fraction(str(y))


STONES = [
    point("1", "1"),
    point("1", "3"),
    point("3.2371", "2.8783"),
    point("5.2070", "2.5322"),
    point("8.4918", "1"),
]
LABELS = ["s", "g", "b1", "b2", "t"]
BOARD = (0, 0, 12, 12)
WITNESS = point("6.4928", "1")
CONTESTER_RECTANGLE = (1, 1, Fraction("6.4934"), Fraction("3.463"))


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def subtract(a, b):
    return tuple(x - y for x, y in zip(a, b))


def scale(a, factor):
    return tuple(x * factor for x in a)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def squared_distance(a, b):
    offset = subtract(a, b)
    return dot(offset, offset)


def clip(polygon, normal, bound):
    """Intersect a convex polygon with dot(x, normal) <= bound."""
    result = []
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        da = dot(a, normal) - bound
        db = dot(b, normal) - bound
        if da <= 0:
            result.append(a)
        if da < 0 < db or db < 0 < da:
            result.append(add(a, scale(subtract(b, a), da / (da - db))))
    return result


def voronoi(owner, stones, rectangle):
    x0, y0, x1, y1 = rectangle
    polygon = [point(x0, y0), point(x1, y0), point(x1, y1), point(x0, y1)]
    for other in stones:
        if other != owner:
            polygon = clip(
                polygon,
                scale(subtract(other, owner), 2),
                dot(other, other) - dot(owner, owner),
            )
    return polygon


def verify():
    owner = STONES[0]
    assert all(1 <= coordinate <= 11 for stone in STONES for coordinate in stone)
    assert all(squared_distance(a, b) >= 4 for a, b in combinations(STONES, 2))

    cell = voronoi(owner, STONES, BOARD)
    assert cell == voronoi(owner, STONES[:-1], BOARD)
    print("Owner's unchanged cell:")
    for vertex in cell:
        print(" ", tuple(float(value) for value in vertex))

    # A placement contests some cell point iff it contests at least one vertex:
    # |v-s|^2 - |v-p|^2 is affine in v. Bound those vertex disks within the inset.
    _, _, right, top = CONTESTER_RECTANGLE
    for vertex in cell:
        rho_squared = squared_distance(vertex, owner)
        inset_vertical_gap = max(Fraction(1) - vertex[1], Fraction(0))
        assert right >= vertex[0]
        assert (right - vertex[0]) ** 2 + inset_vertical_gap**2 >= rho_squared
        assert top >= vertex[1]
        assert (top - vertex[1]) ** 2 >= rho_squared

    # Partition the containing rectangle by nearest stone. Squared distance
    # to that stone is convex, so its maximum on each polygon is at a vertex.
    # Strict < 4 at every vertex certifies open-disk coverage of the rectangle.
    print("Rectangle coverage, maximum squared nearest-stone distance (< 4):")
    for label, stone in zip(LABELS, STONES):
        polygon = voronoi(stone, STONES, CONTESTER_RECTANGLE)
        if not polygon:
            continue
        worst = max(squared_distance(vertex, stone) for vertex in polygon)
        assert worst < 4, (label, worst)
        print(f"  {label:>2}: {float(worst):.12f}; exact margin = {4 - worst}")

    # The witness is strictly legal after deleting t, strictly blocked before,
    # and strictly closer than s to the cell's rightmost bottom vertex.
    assert all(1 <= coordinate <= 11 for coordinate in WITNESS)
    assert all(squared_distance(WITNESS, stone) > 4 for stone in STONES[:-1])
    assert squared_distance(WITNESS, STONES[-1]) < 4
    territory = max(cell, key=lambda vertex: vertex[0])
    advantage = squared_distance(territory, owner) - squared_distance(territory, WITNESS)
    assert advantage > 0
    assert STONES[-1][1] == owner[1]
    reach = STONES[-1][0] - owner[0]
    assert reach == Fraction("7.4918")
    print("Witness:", tuple(float(value) for value in WITNESS))
    print("Minimum squared clearance after removal:", min(
        squared_distance(WITNESS, stone) for stone in STONES[:-1]
    ))
    print("Squared-distance contest advantage:", advantage)
    print(f"PASS: exact rational lower bound {reach}r = {float(reach)}r")


if __name__ == "__main__":
    verify()
