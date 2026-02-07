import pytest


class TestModelscopeUpload:
    def test_import(self):
        import modelscope_upload

        assert modelscope_upload is not None
