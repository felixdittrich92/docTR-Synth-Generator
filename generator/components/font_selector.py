import os
import random

from fontTools.ttLib import TTFont

__all__ = ["FontSelector"]


class FontSelector:
    """Handles font loading and selection based on character support

    Args:
        font_dir (str): Directory containing TTF/OTF font files
    """

    def __init__(self, font_dir: str):
        self.font_dir = font_dir
        self.font_support_table: dict[str, set[str]] = {}
        self._load_fonts()

    def _load_fonts(self):
        """Load all fonts and build character support table"""
        font_files = [f for f in os.listdir(self.font_dir) if f.lower().endswith((".ttf", ".otf"))]

        print(f"Loading {len(font_files)} fonts...")
        for i, file in enumerate(font_files):
            font_path = os.path.join(self.font_dir, file)
            try:
                font = TTFont(font_path, fontNumber=0, lazy=True)
                supported_chars = self._extract_supported_chars(font)
                self.font_support_table[font_path] = supported_chars
                font.close()
                if (i + 1) % 10 == 0:
                    print(f"Loaded {i + 1}/{len(font_files)} fonts")
            except Exception as e:
                print(f"Error reading font {file}: {e}")

    def _extract_supported_chars(self, font: TTFont) -> set[str]:
        """Extract supported characters from font"""
        supported_chars: set[str] = set()
        for cmap in font["cmap"].tables:
            if cmap.isUnicode():
                supported_chars.update(chr(cp) for cp in cmap.cmap.keys())
        return supported_chars

    def get_font_for_text(self, text: str) -> str | None:
        """Get a random font that supports all characters in the text

        Args:
            text (str): Text to render

        Returns:
            str: Path to a suitable font file, or None if no font supports all characters
        """
        required_chars = set(text)
        matching_fonts = [path for path, chars in self.font_support_table.items() if required_chars.issubset(chars)]
        if matching_fonts:
            return random.choice(matching_fonts)
        return None
