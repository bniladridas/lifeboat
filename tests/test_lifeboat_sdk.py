class TestLifeboatSDK:
    def test_import(self):
        import lifeboat_sdk

        assert lifeboat_sdk is not None

    def test_version(self):
        import lifeboat_sdk

        assert hasattr(lifeboat_sdk, "__version__")
