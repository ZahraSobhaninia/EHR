from corebehrt.azure.util import job

INPUTS = {
    "data": {"type": "uri_folder"},
    "vocabulary": {"type": "uri_folder", "optional": True},
    "code_mapping": {"type": "uri_file", "optional": True},
}
OUTPUTS = {
    "features": {"type": "uri_folder"},
    "tokenized": {"type": "uri_folder"},
}


if __name__ == "__main__":
    from corebehrt.main import create_data_oot

    job.run_main("create_data_oot", create_data_oot.main_data, INPUTS, OUTPUTS)
