from eurosat_mlops import CLASSES, __version__
from eurosat_mlops.config import Settings


def test_token_map_parses_pairs_and_skips_garbage():
    s = Settings(api_tokens="alice:tok1, bob:tok2,broken,:x,y:", _env_file=None)
    assert s.token_map() == {"tok1": "alice", "tok2": "bob"}


def test_origins_split():
    s = Settings(cors_origins="http://a, http://b", _env_file=None)
    assert s.origins() == ["http://a", "http://b"]


def test_ten_classes_and_version():
    assert len(CLASSES) == 10
    assert __version__
