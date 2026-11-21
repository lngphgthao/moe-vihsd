"""CLI help không phụ thuộc code page Windows."""
import contextlib
import io
import unittest
import train
import evaluate

class TestCLI(unittest.TestCase):
    def test_help_works_with_windows_pipe_encoding(self):
        for module in (train, evaluate):
            with self.subTest(module=module.__name__):
                output = io.BytesIO()
                stream = io.TextIOWrapper(output, encoding="cp1252")
                with contextlib.redirect_stdout(stream), self.assertRaises(SystemExit) as result:
                    module.main(["--help"])
                self.assertEqual(result.exception.code, 0)
                stream.flush()
                self.assertIn(b"usage:", output.getvalue())
                stream.close()
