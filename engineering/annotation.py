"""Shared validation and layout math for tables and multileaders.

Every table cell is fitted the same way, whoever draws it (`entity_create_table`,
`arch_schedule`, `bom_table`, the hole table): text widths are MEASURED with
the font the headless renderer draws (`text_width`), never estimated from a
character count, and no cell text crosses its cell's rules:

* without ``column_widths`` each column is as wide as its widest header or
  value plus the cell margins (`fit_column_widths`), and the last column widens
  when the merged title would not fit;
* with ``column_widths`` the widths are kept and an overflowing cell wraps at
  its spaces, its row growing to the lines it needs (`cell_lines`). MTEXT does
  not break a word, so a single word wider than its cell still overhangs; the
  table reports it (``text_fit.overflowing``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

#: MTEXT line pitch at line spacing 1.0, in text heights.
LINE_PITCH = 5.0 / 3.0
#: Room below the last baseline for descenders, in text heights.
DESCENT = 0.3


@lru_cache(maxsize=1)
def _font():
    from ezdxf.fonts import fonts

    # the font ezdxf renders a Standard-style MTEXT with
    return fonts.make_font("DejaVuSans.ttf", 1.0)


def text_width(text: str, height: float) -> float:
    """Measured advance of one line of ``text`` lettered ``height`` high."""
    return _font().text_width(str(text)) * float(height) if text else 0.0


def cell_padding(text_height: float, row_height: float) -> float:
    """The cell margin: scales with the lettering, never more than 15 % of the row."""
    return min(0.4 * float(text_height), 0.15 * float(row_height))


def cell_lines(text: str, span: float, text_height: float, padding: float) -> tuple[int, bool]:
    """The lines ``text`` wraps to in a cell ``span`` wide, and whether one of
    its words is wider than the cell (MTEXT wraps at spaces only)."""
    available = float(span) - 2.0 * padding + 1e-9
    space = text_width("x x", text_height) - text_width("xx", text_height)
    lines, used, overflow = 1, 0.0, False
    for word in str(text).split():
        width = text_width(word, text_height)
        overflow = overflow or width > available
        if used and used + space + width > available:
            lines, used = lines + 1, width
        else:
            used += (space if used else 0.0) + width
    return lines, overflow


def fit_column_widths(
    cells: list[list[str]],
    text_height: float,
    row_height: float,
    minimums: list[float] | None = None,
) -> list[float]:
    """Each column as wide as its widest cell plus both margins, never under its minimum."""
    padding = cell_padding(text_height, row_height)
    columns = len(cells[0])
    floor = minimums if minimums is not None else [0.0] * columns
    return [
        max(
            float(floor[c]),
            max(text_width(row[c], text_height) for row in cells) + 2.0 * padding,
        )
        for c in range(columns)
    ]


@dataclass(frozen=True)
class TableLayout:
    cells: list[list[str]]
    column_widths: list[float]
    row_height: float
    text_height: float
    #: The first row is a title, merged across every column as AutoCAD's
    #: TABLE draws it.
    title_row: bool = False
    #: Each row's height: ``row_height``, or more where a cell wraps.
    row_heights: tuple[float, ...] = ()
    #: Per cell: the lines its text wraps to (`cell_lines`).
    lines: tuple[tuple[int, ...], ...] = ()
    #: Cells holding a word wider than the cell.
    overflowing: int = 0

    @property
    def padding(self) -> float:
        return cell_padding(self.text_height, self.row_height)

    def span(self, row: int, column: int) -> float:
        """A cell's width; a title row is merged across the table."""
        return self.width if self.title_row and row == 0 else self.column_widths[column]

    def row_top(self, row: int) -> float:
        """Distance from the table's top edge down to ``row``'s top rule."""
        return sum(self.row_heights[:row])

    @property
    def fitted(self) -> dict:
        """How many cells wrapped, and how many still overhang their cell."""
        return {
            "wrapped": sum(1 for row in self.lines for n in row if n > 1),
            "overflowing": self.overflowing,
        }

    @property
    def row_count(self) -> int:
        return len(self.cells)

    @property
    def column_count(self) -> int:
        return len(self.column_widths)

    @property
    def width(self) -> float:
        return sum(self.column_widths)

    @property
    def height(self) -> float:
        return sum(self.row_heights)


def prepare_table_layout(
    rows: list[list[str]],
    headers: list[str] | None = None,
    column_widths: list[float] | None = None,
    row_height: float = 7.0,
    text_height: float = 2.5,
    title: str | None = None,
) -> TableLayout:
    """Validate table input and calculate deterministic cross-backend sizes."""
    if not rows:
        raise RuntimeError("entity_create_table: at least one data row is required")
    if len(rows) > 200:
        raise RuntimeError("entity_create_table: maximum is 200 rows")
    if row_height <= 0 or text_height <= 0:
        raise RuntimeError("entity_create_table: row_height and text_height must be positive")

    column_count = len(headers) if headers is not None else len(rows[0])
    if column_count < 1 or column_count > 30:
        raise RuntimeError("entity_create_table: table must contain between 1 and 30 columns")
    if any(len(row) != column_count for row in rows):
        raise RuntimeError("entity_create_table: all rows must have the same number of columns")
    if headers is not None and len(headers) != column_count:
        raise RuntimeError("entity_create_table: headers must match the data column count")

    data = [[str(cell) for cell in row] for row in rows]
    if headers is not None:
        data.insert(0, [str(cell) for cell in headers])
    if title:
        data.insert(0, [str(title)] + [""] * (column_count - 1))

    padding = cell_padding(text_height, row_height)
    if column_widths is None:
        body = data[1:] if title else data
        widths = fit_column_widths(body, text_height, row_height, [10.0] * column_count)
        if title:
            short = text_width(title, text_height) + 2.0 * padding - sum(widths)
            if short > 0:
                widths[-1] += short
    else:
        if len(column_widths) != column_count:
            raise RuntimeError("entity_create_table: column_widths must match column count")
        widths = [float(width) for width in column_widths]
        if any(width <= 0 for width in widths):
            raise RuntimeError("entity_create_table: column widths must be positive")

    total = sum(widths)
    fitted = [
        [
            cell_lines(value, total if title and r == 0 else widths[c], text_height, padding)
            for c, value in enumerate(row)
        ]
        for r, row in enumerate(data)
    ]
    h = float(text_height)
    row_heights = tuple(
        max(float(row_height), 2.0 * padding + h + (most - 1) * LINE_PITCH * h + DESCENT * h)
        if (most := max(n for n, _ in row)) > 1
        else float(row_height)
        for row in fitted
    )
    return TableLayout(
        data,
        widths,
        float(row_height),
        h,
        title_row=bool(title),
        row_heights=row_heights,
        lines=tuple(tuple(n for n, _ in row) for row in fitted),
        overflowing=sum(1 for row in fitted for _, over in row if over),
    )


def validate_mleader(points: list[list[float]], text: str) -> list[tuple[float, float]]:
    if len(points) < 2:
        raise RuntimeError("leader_create_mleader: at least two points are required")
    if not str(text).strip():
        raise RuntimeError("leader_create_mleader: text must not be empty")
    normalized = []
    for point in points:
        if len(point) < 2:
            raise RuntimeError("leader_create_mleader: each point must contain x and y")
        normalized.append((float(point[0]), float(point[1])))
    return normalized
