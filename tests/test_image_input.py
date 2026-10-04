import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class TestImageInput(unittest.TestCase):
    def test_image_path_validation_primitives(self):
        with TemporaryDirectory() as tmp:
            image = Path(tmp) / "map.png"
            image.write_bytes(b"placeholder")
            self.assertTrue(image.is_file())
            self.assertFalse((Path(tmp) / "missing.png").is_file())


if __name__ == "__main__":
    unittest.main()
