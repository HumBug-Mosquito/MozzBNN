# DatabaseAccess

# HumBug Database

This repo includes detailed instructions for how to query data from the existing HumBug database, and use the data to generate features.

## Database access

The data is situated on the `rvm7` robots server at `humbug.ac.uk`. To access the machine through `ssh` one requires an account with a password to be set up. Please contact Steve Roberts (MLRG), Christopher Rabson (robots admin), and Adam Galt (robots admin) for more account instructions. Connecting directly to the HumBug server requires a VPN connection to the university if accessed outside of the university network.

## Copying data for local use

For the purpose of working with the data, one may query the database through a Python wrapper for `SQL` commands. Please refer to Section 3.1 in `HumBug_database_documentation.pdf` for query documentation which includes helpful examples.

Once queried, one needs to copy the files from the database that are relevant to the queries. For example, if we are to select all fine labels of mosquito class, while preserving the `label.id`, `audio_id`, `fine_start_time`, `fine_end_time`, `species`, `sound_type` and `path` we would proceed as follows:


1. `ssh` into `humbug.ac.uk`.
2. From any directory, for example the `/home/*user_name*` directory, execute the command
>`python3 /data/access/access.py SELECT label.id,
audio_id, fine_start_time, fine_end_time, species, sound_type, path
FROM label
LEFT JOIN mosquito
ON (label.mosquito_id = mosquito.id)
RIGHT JOIN audio
ON (label.audio_id = audio.id)
WHERE sound_type = 'mosquito' AND type = 'Fine' AND species IS NOT NULL; > fine_mosquito.csv`
3. On Linux/Mac, copy, via `scp` or `rsync` the results of the query, which are stored in `fine_mosquito.csv` to your local machine as follows:
`scp humbug.ac.uk:/home/*user_name*/fine_mosquito.csv .` On Windows you may either use a Linux-like emulator to run unix commands, or use an FTP such as WinSCP to establish a connection to the HumBug server, and then use the GUI for file transfer.

4. Open the `file_io` notebook in `Code/notebooks` to then convert the query into a `pandas` dataframe. Further processing to select and filter data, by fields such as the name of the experiment (determined by conditional statements from the file path names), is documented in the notebook.

## Jupyter notebook documentation

The notebook is split into three themes: data i/o, data filtering and label conversion, and classification and evaluation. Please see the notebook for the latest documentation on why certain label conversion is necessary for a particular experiment.
