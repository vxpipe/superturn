from superturn.sources import expand, load_sources, matches, read_url_list

PERMISSIVE_MARKERS = ("CC0", "CC BY 4.0", "Apache 2.0")


def test_every_source_has_a_permissive_licence_and_attribution():
    for source in load_sources().values():
        assert source.attribution
        assert (source.licence.startswith(PERMISSIVE_MARKERS)
                or source.licence == "Singapore Open Data Licence v1.0"), source.name


def test_emilia_only_downloads_the_yodas_subset():
    emilia = load_sources()["emilia_yodas"]
    for path in expand(emilia.get("files"), emilia.get("languages")):
        assert path.startswith("Emilia-YODAS/"), path


def test_voxpopuli_takes_one_row_group_per_file():
    vox = load_sources()["voxpopuli"]
    assert vox.get("max_bytes_per_file") == 1  # select_row_groups keeps at least one
    assert not any(f.startswith("multilang/") for f in vox.get("files"))


def test_ami_url_list_is_complete():
    rows = read_url_list(load_sources()["ami"])
    assert len(rows) > 600 and all(size > 0 and url.startswith("https://") for size, url in rows)


def test_matches():
    assert matches("Hindi/test-00000-of-00001.parquet", ["*/test-*.parquet"])
    assert not matches("Emilia/DE/x.tar", ["Emilia-YODAS/*"])
