# Raw CTG data

The raw waveform files are intentionally not stored in this repository.
Download version 1.0.0 of the CTU-UHB Intrapartum Cardiotocography Database
from PhysioNet:

<https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/>

Place the downloaded WFDB record files (`*.hea` and `*.dat`) directly in this
directory. The expected layout is:

```text
data/raw/
├── 1001.dat
├── 1001.hea
├── 1002.dat
├── 1002.hea
└── ...
```

When publishing work based on the database, follow its attribution terms and
cite:

> Chudáček, V., Spilka, J., Burša, M., Janků, P., Hruban, L., Huptych, M., &
> Lhotská, L. (2014). Open access intrapartum CTG database. *BMC Pregnancy and
> Childbirth, 14*, 16. <https://doi.org/10.1186/1471-2393-14-16>

The PhysioNet page also provides the dataset DOI, license, and requested
PhysioNet citation.
