from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

EXCEL_PATH = BASE_DIR / "data" / "RBI_Data.xlsx"
CHROMA_PATH = BASE_DIR / "data" / "chroma"

SHEET_NAME = "Weekly"
HEADER_ROW = 5        # row 4 is a merged group header; row 5 has real names
FIRST_DATA_ROW = 7    # row 6 is blank

COLLECTION_NAME = "rbi_weekly"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"