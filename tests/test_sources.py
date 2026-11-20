from superturn.sources import load_sources, matches, read_url_list

PERMISSIVE_MARKERS = ("CC0", "CC BY 4.0", "Apache 2.0", "Basis Data License 1.0")


def test_every_source_has_a_permissive_licence_and_attribution():
    for source in load_sources().values():
        assert source.attribution
        assert source.licence.startswith(PERMISSIVE_MARKERS), source.name


def test_emilia_only_downloads_the_yodas_subset():
    emilia = load_sources()["emilia_yodas"]
    include, exclude = emilia.get("include"), emilia.get("exclude")
    assert matches("Emilia-YODAS/DE/DE-B000000.tar", include, exclude)
    assert not matches("Emilia/DE/DE-B000000.tar", include, exclude)


def test_voxpopuli_skips_the_combined_copy():
    vox = load_sources()["voxpopuli"]
    include, exclude = vox.get("include"), vox.get("exclude")
    assert matches("de/train-00000-of-00016.parquet", include, exclude)
    assert not matches("multilang/train-00000-of-00103.parquet", include, exclude)


def test_url_lists_are_complete():
    sources = load_sources()
    for name in ("ami", "icsi"):
        rows = read_url_list(sources[name])
        assert rows and all(size > 0 and url.startswith("https://") for size, url in rows)
